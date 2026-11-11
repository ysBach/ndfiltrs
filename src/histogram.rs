//! Exact sliding-window median and rank for rectangular windows.
//!
//! The output is split into rectangular blocks. For each block, the kernel
//! gathers the input support (the block plus the window halo, through the
//! boundary modes) and sorts the retained samples in IEEE total order. Each
//! sample is then replaced by its rank inside the block. The window visits the
//! block's outputs in boustrophedon order, so every step moves the window by
//! one position along one axis. A step removes one face of the window and
//! adds the opposite face to a bit histogram over the block ranks. A count per
//! group of 4096 ranks makes each order-statistic query a short scan.
//!
//! A step along `axis` costs `2 * volume / size[axis]` histogram updates. The
//! kernel steps mostly along the axis with the largest window size, which
//! makes this number smallest. For a `k x k` window the cost per output is
//! therefore proportional to `k`, not to `k * k`. The block sort adds about
//! `(support / block) * log2(support)` comparisons per output.

use std::cmp::Ordering;

use rayon::prelude::*;
use reducers::Float;

use crate::boundary::{map_index, BoundaryMode};
use crate::error::Error;
use crate::median_network::NetworkFloat;
use crate::reduction::InvalidPolicy;

/// The order statistic selected from the retained samples of each window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Statistic {
    /// Median. An even count averages the two middle values.
    Median,
    /// Zero-based rank in ascending total order.
    Rank(usize),
}

/// Code of a sample that the policy omits.
const OMITTED: u32 = u32::MAX;
/// Code of a NaN sample under [`InvalidPolicy::Propagate`].
const PROPAGATED_NAN: u32 = u32::MAX - 1;
/// log2 of the number of ranks counted by one group (64 words of 64 bits).
const GROUP_SHIFT: usize = 12;
/// Smallest block length per axis. This constant changes speed and memory
/// only, never results. A larger block spreads the cost of one block sort
/// over more outputs when the window is small.
const MIN_BLOCK: usize = 32;
/// Marks a support coordinate outside a `Constant` boundary.
const OUTSIDE: usize = usize::MAX;

/// Block length per axis.
///
/// Each axis gets the largest number of blocks that are at least
/// `max(size, MIN_BLOCK)` long (or the whole axis, when it is shorter). All
/// blocks but the last have the same length, and the last is not much
/// shorter. A block about as long as the window keeps the support at
/// about `2^rank` times the block volume.
fn block_lengths(shape: &[usize], size: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .zip(size)
        .map(|(&length, &window)| {
            let blocks = (length / window.max(MIN_BLOCK)).max(1);
            length.div_ceil(blocks).max(1)
        })
        .collect()
}

/// Check that every block support can be ranked with `u32` codes.
///
/// # Errors
///
/// Returns [`Error::Overflow`] when a block support has more elements than
/// the rank codes can address.
pub(crate) fn validate_support(shape: &[usize], size: &[usize]) -> Result<(), Error> {
    let support = block_lengths(shape, size).iter().zip(size).try_fold(
        1usize,
        |product, (&block, &window)| {
            block
                .checked_add(window - 1)
                .and_then(|length| product.checked_mul(length))
                .ok_or(Error::Overflow)
        },
    )?;
    if support > PROPAGATED_NAN as usize {
        return Err(Error::Overflow);
    }
    Ok(())
}

/// Position of the `k`-th set bit (zero-based) of `word`, which has more
/// than `k` set bits. Each step keeps the half that holds the bit.
#[inline]
fn select_in_word(mut word: u64, mut k: u32) -> u32 {
    let mut position = 0;
    for width in [32, 16, 8, 4, 2, 1] {
        let low = (word & ((1u64 << width) - 1)).count_ones();
        if k >= low {
            k -= low;
            word >>= width;
            position += width;
        }
    }
    position
}

/// Bit histogram over block ranks, with counts of retained and NaN samples.
struct Histogram<'a> {
    words: &'a mut [u64],
    groups: &'a mut [u32],
    count: usize,
    nan: usize,
}

impl Histogram<'_> {
    #[inline]
    fn add(&mut self, code: u32) {
        match code.cmp(&PROPAGATED_NAN) {
            Ordering::Less => {
                let rank = code as usize;
                self.words[rank >> 6] |= 1 << (rank & 63);
                self.groups[rank >> GROUP_SHIFT] += 1;
                self.count += 1;
            }
            Ordering::Equal => self.nan += 1,
            Ordering::Greater => {}
        }
    }

    #[inline]
    fn remove(&mut self, code: u32) {
        match code.cmp(&PROPAGATED_NAN) {
            Ordering::Less => {
                let rank = code as usize;
                self.words[rank >> 6] &= !(1 << (rank & 63));
                self.groups[rank >> GROUP_SHIFT] -= 1;
                self.count -= 1;
            }
            Ordering::Equal => self.nan -= 1,
            Ordering::Greater => {}
        }
    }

    /// Block rank of the `k`-th smallest retained sample (zero-based).
    /// The caller guarantees `k < self.count`.
    fn select(&self, mut k: usize) -> usize {
        let mut group = 0;
        loop {
            let count = self.groups[group] as usize;
            if k < count {
                break;
            }
            k -= count;
            group += 1;
        }
        let mut index = group << (GROUP_SHIFT - 6);
        loop {
            let count = self.words[index].count_ones() as usize;
            if k < count {
                break;
            }
            k -= count;
            index += 1;
        }
        (index << 6) + select_in_word(self.words[index], k as u32) as usize
    }

    fn finish<T: Float>(&self, values: &[T], policy: InvalidPolicy, statistic: Statistic) -> T {
        match policy {
            InvalidPolicy::Propagate if self.nan > 0 => return T::nan(),
            InvalidPolicy::Omit { min_count, .. } if self.count < min_count => return T::nan(),
            _ => {}
        }
        match statistic {
            // `median_valid_in_place` applies the same midpoint and cast as
            // `Reduction::Median` to the selected middle values.
            Statistic::Median if self.count % 2 == 1 => {
                let mut middle = [values[self.select(self.count / 2)]];
                T::from_f64(reducers::reducers_1d::median_valid_in_place(&mut middle))
            }
            Statistic::Median => {
                let mut middle = [
                    values[self.select(self.count / 2 - 1)],
                    values[self.select(self.count / 2)],
                ];
                T::from_f64(reducers::reducers_1d::median_valid_in_place(&mut middle))
            }
            Statistic::Rank(rank) if rank < self.count => values[self.select(rank)],
            Statistic::Rank(_) => T::nan(),
        }
    }
}

/// Call `visit` with the support index of every element of a box.
///
/// The box starts at `low` and has `lengths` elements per axis. `strides`
/// are the support strides. The inner loop runs along the last axis with
/// more than one element.
#[inline]
fn for_each_in_box(
    low: &[usize],
    lengths: &[usize],
    strides: &[usize],
    counter: &mut [usize],
    mut visit: impl FnMut(usize),
) {
    let rank = low.len();
    let inner = (0..rank)
        .rev()
        .find(|&axis| lengths[axis] > 1)
        .unwrap_or(rank - 1);
    let inner_stride = strides[inner];
    let inner_length = lengths[inner];
    counter.fill(0);
    let mut base: usize = low.iter().zip(strides).map(|(&l, &s)| l * s).sum();
    loop {
        let mut index = base;
        for _ in 0..inner_length {
            visit(index);
            index += inner_stride;
        }
        // Advance the odometer over the other axes.
        let mut axis = rank;
        loop {
            if axis == 0 {
                return;
            }
            axis -= 1;
            if axis == inner {
                continue;
            }
            counter[axis] += 1;
            base += strides[axis];
            if counter[axis] < lengths[axis] {
                break;
            }
            base -= strides[axis] * lengths[axis];
            counter[axis] = 0;
        }
    }
}

/// Reusable per-worker buffers.
struct Workspace<T: NetworkFloat> {
    maps: Vec<Vec<usize>>,
    codes: Vec<u32>,
    pairs: Vec<(T::Key, u32)>,
    values: Vec<T>,
    words: Vec<u64>,
    groups: Vec<u32>,
}

impl<T: NetworkFloat> Workspace<T> {
    fn new(rank: usize) -> Self {
        Self {
            maps: vec![Vec::new(); rank],
            codes: Vec::new(),
            pairs: Vec::new(),
            values: Vec::new(),
            words: Vec::new(),
            groups: Vec::new(),
        }
    }
}

fn refill<V: Clone>(buffer: &mut Vec<V>, length: usize, value: V) -> Result<(), Error> {
    buffer.clear();
    buffer.try_reserve(length).map_err(|_| Error::Memory)?;
    buffer.resize(length, value);
    Ok(())
}

/// Validated geometry shared by all blocks.
struct Geometry<'a, T> {
    input: &'a [T],
    shape: &'a [usize],
    size: &'a [usize],
    modes: &'a [BoundaryMode],
    anchors: Vec<i128>,
    strides: Vec<usize>,
    block: Vec<usize>,
    /// Axes from outermost to innermost traversal; the last moves fastest.
    order: Vec<usize>,
    cval: T,
    policy: InvalidPolicy,
    statistic: Statistic,
}

impl<T: NetworkFloat> Geometry<'_, T> {
    fn retained(&self, value: T) -> bool {
        let ignore_inf = matches!(
            self.policy,
            InvalidPolicy::Omit {
                ignore_inf: true,
                ..
            }
        );
        !value.is_nan() && (!ignore_inf || value.is_finite())
    }

    /// Filter the block whose first output has coordinates `start` and whose
    /// shape is `extent`. `output` receives the results in C order.
    fn block(
        &self,
        start: &[usize],
        extent: &[usize],
        workspace: &mut Workspace<T>,
        output: &mut [T],
    ) -> Result<(), Error> {
        let rank = self.shape.len();
        let lengths: Vec<usize> = (0..rank)
            .map(|axis| extent[axis] + self.size[axis] - 1)
            .collect();
        let mut support_strides = vec![1usize; rank];
        for axis in (0..rank - 1).rev() {
            support_strides[axis] = support_strides[axis + 1] * lengths[axis + 1];
        }
        let total = support_strides[0] * lengths[0];

        // Map each support coordinate to an input offset along its axis.
        for axis in 0..rank {
            let map = &mut workspace.maps[axis];
            map.clear();
            map.try_reserve(lengths[axis]).map_err(|_| Error::Memory)?;
            let first = start[axis] as i128 - self.anchors[axis];
            for step in 0..lengths[axis] {
                map.push(
                    match map_index(first + step as i128, self.shape[axis], self.modes[axis])? {
                        Some(index) => index * self.strides[axis],
                        None => OUTSIDE,
                    },
                );
            }
        }

        // Gather and classify the support, then rank the retained samples.
        refill(&mut workspace.codes, total, OMITTED)?;
        workspace.pairs.clear();
        workspace
            .pairs
            .try_reserve(total)
            .map_err(|_| Error::Memory)?;
        let propagate = matches!(self.policy, InvalidPolicy::Propagate);
        let last = rank - 1;
        let mut counter = vec![0usize; rank];
        let mut support_index = 0usize;
        'rows: loop {
            let mut base = 0usize;
            let mut outside = false;
            for (map, &step) in workspace.maps[..last].iter().zip(&counter) {
                let offset = map[step];
                outside |= offset == OUTSIDE;
                base = base.wrapping_add(offset);
            }
            for &offset in &workspace.maps[last] {
                let value = if outside || offset == OUTSIDE {
                    self.cval
                } else {
                    self.input[base + offset]
                };
                if self.retained(value) {
                    workspace
                        .pairs
                        .push((value.sort_key(), support_index as u32));
                } else if propagate {
                    workspace.codes[support_index] = PROPAGATED_NAN;
                }
                support_index += 1;
            }
            let mut axis = last;
            loop {
                if axis == 0 {
                    break 'rows;
                }
                axis -= 1;
                counter[axis] += 1;
                if counter[axis] < lengths[axis] {
                    break;
                }
                counter[axis] = 0;
            }
        }
        workspace.pairs.sort_unstable();
        workspace.values.clear();
        workspace
            .values
            .try_reserve(workspace.pairs.len())
            .map_err(|_| Error::Memory)?;
        for (rank_index, &(key, index)) in workspace.pairs.iter().enumerate() {
            workspace.codes[index as usize] = rank_index as u32;
            workspace.values.push(T::from_sort_key(key));
        }
        let ranks = workspace.pairs.len();
        refill(&mut workspace.words, ranks.div_ceil(64), 0)?;
        refill(&mut workspace.groups, ranks.div_ceil(1 << GROUP_SHIFT), 0)?;

        let mut output_strides = vec![1usize; rank];
        for axis in (0..rank - 1).rev() {
            output_strides[axis] = output_strides[axis + 1] * extent[axis + 1];
        }
        if output.len() != output_strides[0] * extent[0] {
            return Err(Error::LengthMismatch);
        }

        let codes = &workspace.codes;
        let mut histogram = Histogram {
            words: &mut workspace.words,
            groups: &mut workspace.groups,
            count: 0,
            nan: 0,
        };
        let mut position = vec![0usize; rank];
        let mut direction = vec![true; rank];
        let mut low = vec![0usize; rank];
        let mut box_lengths = self.size.to_vec();
        for_each_in_box(
            &position,
            self.size,
            &support_strides,
            &mut counter,
            |index| {
                histogram.add(codes[index]);
            },
        );
        loop {
            let offset: usize = position
                .iter()
                .zip(&output_strides)
                .map(|(&p, &s)| p * s)
                .sum();
            output[offset] = histogram.finish(&workspace.values, self.policy, self.statistic);

            // Boustrophedon step: move the fastest axis that can still move in
            // its direction, reversing every faster axis that cannot.
            let mut moved = None;
            for &axis in self.order.iter().rev() {
                let can_move = if direction[axis] {
                    position[axis] + 1 < extent[axis]
                } else {
                    position[axis] > 0
                };
                if can_move {
                    moved = Some(axis);
                    break;
                }
                direction[axis] = !direction[axis];
            }
            let Some(axis) = moved else {
                break;
            };
            let (removed, added) = if direction[axis] {
                (position[axis], position[axis] + self.size[axis])
            } else {
                (position[axis] + self.size[axis] - 1, position[axis] - 1)
            };
            low.copy_from_slice(&position);
            box_lengths[axis] = 1;
            low[axis] = removed;
            for_each_in_box(
                &low,
                &box_lengths,
                &support_strides,
                &mut counter,
                |index| {
                    histogram.remove(codes[index]);
                },
            );
            low[axis] = added;
            for_each_in_box(
                &low,
                &box_lengths,
                &support_strides,
                &mut counter,
                |index| {
                    histogram.add(codes[index]);
                },
            );
            box_lengths[axis] = self.size[axis];
            if direction[axis] {
                position[axis] += 1;
            } else {
                position[axis] -= 1;
            }
        }
        Ok(())
    }
}

/// Apply an exact rectangular median or rank filter with block histograms.
///
/// The result equals [`crate::reduction::Reduction::Median`] or
/// [`crate::reduction::Reduction::Rank`] with a direct window traversal, bit
/// for bit, and does not depend on the number of threads.
#[allow(clippy::too_many_arguments)]
pub(crate) fn order_statistic_into<T: NetworkFloat + Send + Sync>(
    input: &[T],
    shape: &[usize],
    size: &[usize],
    origin: &[isize],
    modes: &[BoundaryMode],
    cval: T,
    policy: InvalidPolicy,
    statistic: Statistic,
    output: &mut [T],
) -> Result<(), Error> {
    let rank = shape.len();
    if rank == 0 || size.len() != rank || origin.len() != rank || modes.len() != rank {
        return Err(Error::InvalidConfiguration(
            "array, window, origin, and mode must have equal nonzero rank",
        ));
    }
    if size.contains(&0) {
        return Err(Error::InvalidConfiguration("window sizes must be positive"));
    }
    let output_len = shape.iter().try_fold(1usize, |product, &length| {
        product.checked_mul(length).ok_or(Error::Overflow)
    })?;
    if input.len() != output_len || output.len() != output_len {
        return Err(Error::LengthMismatch);
    }
    if output_len == 0 {
        return Ok(());
    }
    validate_support(shape, size)?;
    let mut anchors = Vec::with_capacity(rank);
    for (&length, &shift) in size.iter().zip(origin) {
        let signed = isize::try_from(length).map_err(|_| Error::Overflow)?;
        if shift < -(signed / 2) || shift > (signed - 1) / 2 {
            return Err(Error::InvalidConfiguration(
                "origin must satisfy -(size // 2) <= origin <= (size - 1) // 2",
            ));
        }
        anchors.push((signed / 2 + shift) as i128);
    }
    let mut strides = vec![1usize; rank];
    for axis in (0..rank - 1).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    // Step mostly along the axis with the largest window size: its faces are
    // the smallest. Ties keep the last axis, which is contiguous.
    // `max_by_key` returns the last of equal maxima.
    let fastest = (0..rank).max_by_key(|&axis| size[axis]).unwrap_or(rank - 1);
    let mut order: Vec<usize> = (0..rank).filter(|&axis| axis != fastest).collect();
    order.push(fastest);
    let geometry = Geometry {
        input,
        shape,
        size,
        modes,
        anchors,
        strides,
        block: block_lengths(shape, size),
        order,
        cval,
        policy,
        statistic,
    };

    let blocks: Vec<usize> = (0..rank)
        .map(|axis| shape[axis].div_ceil(geometry.block[axis]))
        .collect();
    let strip_len = geometry.block[0] * geometry.strides[0];
    output
        .par_chunks_mut(strip_len)
        .enumerate()
        .try_for_each(|(strip, chunk)| geometry.strip(strip, &blocks, chunk))
}

impl<T: NetworkFloat + Send + Sync> Geometry<'_, T> {
    /// Filter the blocks of one strip of outputs along axis 0.
    ///
    /// The blocks run in parallel. Each writes a contiguous part of one
    /// scratch buffer, which is then copied into `chunk`.
    fn strip(&self, strip: usize, blocks: &[usize], chunk: &mut [T]) -> Result<(), Error> {
        let rank = self.shape.len();
        let mut tiles = Vec::new();
        let count: usize = blocks[1..].iter().product();
        tiles.try_reserve_exact(count).map_err(|_| Error::Memory)?;
        for mut index in 0..count {
            let mut start = vec![0usize; rank];
            start[0] = strip * self.block[0];
            for axis in (1..rank).rev() {
                start[axis] = (index % blocks[axis]) * self.block[axis];
                index /= blocks[axis];
            }
            let extent: Vec<usize> = (0..rank)
                .map(|axis| self.block[axis].min(self.shape[axis] - start[axis]))
                .collect();
            tiles.push((start, extent));
        }
        let mut scratch = Vec::new();
        scratch
            .try_reserve_exact(chunk.len())
            .map_err(|_| Error::Memory)?;
        scratch.resize(chunk.len(), T::nan());
        let mut parts = Vec::with_capacity(count);
        let mut rest = scratch.as_mut_slice();
        for (_, extent) in &tiles {
            let (part, tail) = rest.split_at_mut(extent.iter().product());
            parts.push(part);
            rest = tail;
        }
        tiles.par_iter().zip(parts).try_for_each_init(
            || Workspace::<T>::new(rank),
            |workspace, ((start, extent), part)| self.block(start, extent, workspace, part),
        )?;
        // Copy each contiguous block row into the strip.
        let strip_start = strip * self.block[0] * self.strides[0];
        let mut values = scratch.as_slice();
        let mut counter = vec![0usize; rank];
        for (start, extent) in &tiles {
            let row = extent[rank - 1];
            let (block, tail) = values.split_at(extent.iter().product());
            values = tail;
            counter.fill(0);
            for block_row in block.chunks_exact(row) {
                let offset: usize = (0..rank)
                    .map(|axis| (start[axis] + counter[axis]) * self.strides[axis])
                    .sum::<usize>()
                    - strip_start;
                chunk[offset..offset + row].copy_from_slice(block_row);
                for axis in (0..rank - 1).rev() {
                    counter[axis] += 1;
                    if counter[axis] < extent[axis] {
                        break;
                    }
                    counter[axis] = 0;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::boundary::BoundaryMode::{self, Constant, Mirror, Nearest, Reflect, Wrap};
    use crate::plan::{window_filter, FilterFloat, Neighborhood};
    use crate::reduction::{InvalidPolicy, Reduction};

    const MODES: [BoundaryMode; 5] = [Reflect, Constant, Nearest, Mirror, Wrap];

    #[test]
    fn word_selection_finds_every_set_bit() {
        for word in [1u64, u64::MAX, 0x8000_0000_0000_0001, 0x0f0f_00ff_1234_8001] {
            let expected: Vec<u32> = (0..64).filter(|bit| word >> bit & 1 == 1).collect();
            for (k, &bit) in expected.iter().enumerate() {
                assert_eq!(super::select_in_word(word, k as u32), bit);
            }
        }
    }

    /// Values with many ties, both zeros, infinities, and NaN payloads.
    fn sample<T: FilterFloat>(len: usize, seed: u64) -> Vec<T> {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                match state % 23 {
                    0 => T::nan(),
                    1 => T::from_f64(f64::INFINITY),
                    2 => T::from_f64(f64::NEG_INFINITY),
                    3 => T::from_f64(-0.0),
                    4 => T::from_f64(0.0),
                    5 => T::from_f64(f64::from_bits(0x7ff8_0000_0000_0123)),
                    other => T::from_f64((other % 9) as f64 - 4.0 + (state % 3) as f64 / 8.0),
                }
            })
            .collect()
    }

    fn policies(volume: usize) -> Vec<InvalidPolicy> {
        vec![
            InvalidPolicy::Propagate,
            InvalidPolicy::Omit {
                ignore_inf: false,
                min_count: 1,
            },
            InvalidPolicy::Omit {
                ignore_inf: true,
                min_count: volume.div_ceil(2),
            },
            InvalidPolicy::Omit {
                ignore_inf: true,
                min_count: volume,
            },
        ]
    }

    fn assert_matches_direct<T: FilterFloat + std::fmt::Debug>(
        input: &[T],
        shape: &[usize],
        hood: &Neighborhood,
        cval: T,
    ) {
        let volume: usize = hood.size.iter().product();
        for policy in policies(volume) {
            let mut pairs = vec![(
                Reduction::HistogramMedian(policy),
                Reduction::Median(policy),
            )];
            for rank in [0, volume / 3, volume - 1] {
                pairs.push((
                    Reduction::HistogramRank(policy, rank),
                    Reduction::Rank(policy, rank),
                ));
            }
            for (fast, slow) in pairs {
                let actual = window_filter(input, shape, hood, fast, cval).unwrap();
                let expected = window_filter(input, shape, hood, slow, cval).unwrap();
                for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        a.to_f64().to_bits(),
                        e.to_f64().to_bits(),
                        "{fast:?} {hood:?} cval={cval:?} at {index}: {a:?} != {e:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn one_dimensional_windows_match_direct_selection() {
        let shape = [150];
        let input = sample::<f64>(150, 1);
        for size in [1, 2, 5, 8, 33, 64, 151, 300] {
            let signed = size as isize;
            for origin in [-(signed / 2), 0, (signed - 1) / 2] {
                for mode in MODES {
                    let hood = Neighborhood::rectangular([size])
                        .origin([origin])
                        .mode(mode);
                    assert_matches_direct(&input, &shape, &hood, 1.5);
                }
            }
        }
    }

    #[test]
    fn two_dimensional_windows_match_direct_selection() {
        let shape = [70, 75];
        let input32 = sample::<f32>(70 * 75, 2);
        let input64 = sample::<f64>(70 * 75, 3);
        for size in [[3, 3], [4, 6], [7, 2], [1, 9], [35, 3]] {
            for mode in MODES {
                let hood = Neighborhood::rectangular(size).mode(mode);
                assert_matches_direct(&input32, &shape, &hood, 0.0);
                assert_matches_direct(&input64, &shape, &hood, 0.0);
            }
        }
        let shifted = Neighborhood::rectangular([6, 5])
            .origin([-3, 2])
            .modes([Wrap, Constant]);
        for cval in [f64::NAN, f64::INFINITY, -0.0, 2.0] {
            assert_matches_direct(&input64, &shape, &shifted, cval);
        }
    }

    #[test]
    fn three_dimensional_windows_match_direct_selection() {
        let shape = [9, 40, 37];
        let input = sample::<f32>(9 * 40 * 37, 4);
        for (size, origin) in [
            ([3, 3, 3], [0, 0, 0]),
            ([2, 5, 4], [-1, 2, 1]),
            ([12, 1, 2], [5, 0, -1]),
        ] {
            let hood = Neighborhood::rectangular(size)
                .origin(origin)
                .modes([Mirror, Reflect, Constant]);
            assert_matches_direct(&input, &shape, &hood, f32::NAN);
            assert_matches_direct(&input, &shape, &hood.clone().mode(Nearest), -0.0);
        }
    }

    #[test]
    fn supports_with_several_rank_groups_match_direct_selection() {
        // A 33 x 31 window gives block supports of more than 4096 ranks.
        let shape = [70, 75];
        let input = sample::<f64>(70 * 75, 6);
        let hood = Neighborhood::rectangular([33, 31]).mode(Reflect);
        assert_matches_direct(&input, &shape, &hood, 0.0);
    }

    #[test]
    fn results_do_not_depend_on_thread_count() {
        let shape = [130, 140];
        let input = sample::<f32>(130 * 140, 7);
        let hood = Neighborhood::rectangular([9, 12]).mode(Wrap);
        let reduction = Reduction::HistogramMedian(InvalidPolicy::Propagate);
        let pooled = window_filter(&input, &shape, &hood, reduction, 0.0).unwrap();
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| window_filter(&input, &shape, &hood, reduction, 0.0).unwrap());
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&pooled), bits(&serial));
    }

    #[test]
    fn oversized_supports_are_rejected_before_allocation() {
        let hood = Neighborhood::rectangular([70_000, 70_000]);
        let reduction = Reduction::HistogramMedian(InvalidPolicy::Propagate);
        assert_eq!(
            crate::plan::WindowFilter::<f32>::new(&[70_000, 70_000], &hood, reduction, 0.0).err(),
            Some(crate::error::Error::Overflow)
        );
    }

    #[test]
    fn degenerate_axes_match_direct_selection() {
        let input = sample::<f64>(3 * 80, 5);
        for size in [[1, 1, 1], [4, 3, 5], [2, 2, 40]] {
            for mode in MODES {
                let hood = Neighborhood::rectangular(size).mode(mode);
                assert_matches_direct(&input, &[3, 1, 80], &hood, 0.5);
            }
        }
    }
}
