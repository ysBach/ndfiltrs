//! Means over per-pixel rectangular windows with summed-area tables.
//!
//! A summed-area table stores, at table index `p`, the sum of all inputs with
//! every coordinate `i_a < p_a`. A window sum is then a signed combination of
//! a few table entries, so the cost per output does not depend on the window
//! size. Boundary modes are handled without padding: along each axis, the
//! prefix sum of the extended signal at any extended index is an integer
//! combination of prefix sums of the original axis.

use rayon::prelude::*;

use crate::boundary::BoundaryMode;
use crate::error::Error;
use crate::plan::{FilterFloat, Modes};
use crate::reduction::InvalidPolicy;

/// Per-pixel rectangular window sizes and boundary modes for
/// [`variable_mean`].
///
/// Each output pixel has its own window size along each axis. A window of
/// size `s` on an axis covers input indices `center - s / 2` through
/// `center - s / 2 + s - 1` (integer division), as `scipy.ndimage` places a
/// window of size `s` with origin `0`. Odd sizes are centered. Even sizes
/// extend one index further toward lower indices.
///
/// The default boundary mode is [`BoundaryMode::Reflect`] on every axis.
///
/// # Examples
///
/// ```
/// use ndfiltrs::{BoundaryMode, VariableWindow};
///
/// // A 2x3 image: one size per pixel, used on both axes.
/// let isotropic = VariableWindow::isotropic([1, 3, 5, 1, 3, 5]);
///
/// // Separate sizes per axis: all axis-0 sizes, then all axis-1 sizes.
/// let per_axis = VariableWindow::per_axis([1, 1, 1, 1, 1, 1, 3, 3, 3, 5, 5, 5])
///     .mode(BoundaryMode::Nearest);
/// # let _ = (isotropic, per_axis);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VariableWindow {
    sizes: Vec<usize>,
    per_axis: bool,
    modes: Modes,
}

impl VariableWindow {
    /// Use one window size per pixel on every axis.
    ///
    /// `sizes` is in C order and has one entry per array element.
    pub fn isotropic(sizes: impl Into<Vec<usize>>) -> Self {
        Self {
            sizes: sizes.into(),
            per_axis: false,
            modes: Modes::All(BoundaryMode::Reflect),
        }
    }

    /// Use one window size per pixel and per axis.
    ///
    /// `sizes` is a C-ordered array with shape `(ndim, *shape)`: the sizes
    /// for axis 0 of every pixel, then for axis 1, and so on.
    pub fn per_axis(sizes: impl Into<Vec<usize>>) -> Self {
        Self {
            sizes: sizes.into(),
            per_axis: true,
            modes: Modes::All(BoundaryMode::Reflect),
        }
    }

    /// Use one boundary mode on every axis.
    pub fn mode(mut self, mode: BoundaryMode) -> Self {
        self.modes = Modes::All(mode);
        self
    }

    /// Use one boundary mode per axis.
    pub fn modes(mut self, modes: impl Into<Vec<BoundaryMode>>) -> Self {
        self.modes = Modes::PerAxis(modes.into());
        self
    }

    fn size(&self, axis: usize, index: usize, len: usize) -> usize {
        if self.per_axis {
            self.sizes[axis * len + index]
        } else {
            self.sizes[index]
        }
    }
}

/// Mean over a per-pixel rectangular window, with a NaN policy.
///
/// `input` is C-ordered with shape `shape`. Output pixel `i` averages the
/// window that `window` assigns to `i`, read through the boundary modes.
/// Each boundary repetition and each `cval` sample counts as one sample,
/// as in [`Reduction::Mean`](crate::Reduction::Mean). Infinite samples follow
/// the same rules as `Reduction::Mean`. With [`InvalidPolicy::Omit`], a
/// window with fewer than `min_count` retained samples gives NaN, including a
/// window smaller than `min_count`.
///
/// # Accuracy
///
/// The function sums in `f64` with summed-area tables. The tables are built
/// with one compensated prefix pass per axis, and each pass rounds its
/// entries to `f64`. A window sum subtracts table entries, so its absolute
/// error grows with the magnitude of the prefix sums, not with the window
/// sum. Derived first-order bound: the absolute error of a window sum is at
/// most about `2^-53 * K * ndim * A`. Here `K` is the number of table entries
/// combined (`2^ndim` for a window inside the array) and `A` is the sum of
/// `|input|` over the whole array. One huge finite value, such as a `1e30`
/// sentinel, therefore damages windows far from it. Set such values to NaN
/// and use [`InvalidPolicy::Omit`]. The result is not
/// bit-identical to [`Reduction::Mean`](crate::Reduction::Mean) or
/// [`Reduction::RollingMean`](crate::Reduction::RollingMean). Use those
/// for exact fixed-size windows.
///
/// # Cost
///
/// Building the tables costs `O(len * ndim)` time and 8 bytes per table entry
/// (`prod(shape[a] + 1)` entries). Each nonfinite category that occurs in the
/// input (NaN, `+inf`, `-inf`) adds one count table of the same size. Each
/// output costs `O(2^ndim)` for windows inside the array, independent of the
/// window size, and up to `O(5^ndim)` for windows that cross a boundary.
///
/// # Errors
///
/// - [`Error::LengthMismatch`]: `input.len()` or the size map length does not
///   match `shape`.
/// - [`Error::InvalidConfiguration`]: rank zero, a zero window size, a mode
///   count other than the rank, `min_count == 0`, or a finite input whose
///   absolute sum overflows `f64`.
/// - [`Error::Overflow`]: a window volume or the table size overflows
///   `usize`.
/// - [`Error::Memory`]: a table allocation failed.
///
/// # Examples
///
/// ```
/// use ndfiltrs::{variable_mean, BoundaryMode, InvalidPolicy, VariableWindow};
///
/// let row = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let window = VariableWindow::isotropic([1, 3, 3, 5, 1]).mode(BoundaryMode::Nearest);
/// let out = variable_mean(&row, &[5], &window, InvalidPolicy::Propagate, 0.0)?;
/// assert_eq!(out, [1.0, 2.0, 3.0, 3.8, 5.0]);
/// # Ok::<(), ndfiltrs::Error>(())
/// ```
pub fn variable_mean<T: FilterFloat>(
    input: &[T],
    shape: &[usize],
    window: &VariableWindow,
    policy: InvalidPolicy,
    cval: T,
) -> Result<Vec<T>, Error> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(input.len())
        .map_err(|_| Error::Memory)?;
    output.resize(input.len(), T::nan());
    variable_mean_into(input, shape, window, policy, cval, &mut output)?;
    Ok(output)
}

/// Write [`variable_mean`] of `input` into `output`.
///
/// `output` must have the same length as `input`. See [`variable_mean`] for
/// semantics, accuracy, and cost.
///
/// # Errors
///
/// Returns the errors of [`variable_mean`]. A wrong `output` length gives
/// [`Error::LengthMismatch`].
pub fn variable_mean_into<T: FilterFloat>(
    input: &[T],
    shape: &[usize],
    window: &VariableWindow,
    policy: InvalidPolicy,
    cval: T,
    output: &mut [T],
) -> Result<(), Error> {
    let rank = shape.len();
    if rank == 0 {
        return Err(Error::InvalidConfiguration("array rank must be at least 1"));
    }
    let len = shape.iter().try_fold(1usize, |product, &value| {
        product.checked_mul(value).ok_or(Error::Overflow)
    })?;
    if input.len() != len || output.len() != len {
        return Err(Error::LengthMismatch);
    }
    let expected_sizes = if window.per_axis {
        len.checked_mul(rank).ok_or(Error::Overflow)?
    } else {
        len
    };
    if window.sizes.len() != expected_sizes {
        return Err(Error::LengthMismatch);
    }
    let modes = match &window.modes {
        Modes::All(mode) => vec![*mode; rank],
        Modes::PerAxis(modes) if modes.len() == rank => modes.clone(),
        Modes::PerAxis(_) => {
            return Err(Error::InvalidConfiguration(
                "modes must have one entry per axis",
            ))
        }
    };
    if matches!(policy, InvalidPolicy::Omit { min_count: 0, .. }) {
        return Err(Error::InvalidConfiguration("min_count must be at least 1"));
    }
    if window.sizes.contains(&0) {
        return Err(Error::InvalidConfiguration("window sizes must be positive"));
    }
    (0..len).into_par_iter().try_for_each(|index| {
        (0..rank)
            .try_fold(1usize, |volume, axis| {
                volume.checked_mul(window.size(axis, index, len))
            })
            .map(|_| ())
            .ok_or(Error::Overflow)
    })?;
    if len == 0 {
        return Ok(());
    }
    let tables = Tables::build(input, shape)?;
    let query = Query {
        shape,
        modes: &modes,
        window,
        tables: &tables,
        policy,
        cval: cval.to_f64(),
        len,
    };
    let threads = rayon::current_num_threads().max(1);
    let chunk_len = len.div_ceil(threads * TASKS_PER_THREAD);
    output
        .par_chunks_mut(chunk_len)
        .enumerate()
        .for_each(|(chunk, values)| query.run(chunk * chunk_len, values));
    Ok(())
}

/// One running prefix value along an axis.
trait Prefix: Copy + Default + Send + Sync {
    type State: Copy + Default + Send;
    /// Add `value` to the running state and return the stored prefix.
    fn step(state: &mut Self::State, value: Self) -> Self;
}

impl Prefix for f64 {
    type State = (f64, f64);

    fn step(state: &mut (f64, f64), value: f64) -> f64 {
        let (sum, correction) = state;
        let next = *sum + value;
        *correction += if sum.abs() >= value.abs() {
            (*sum - next) + value
        } else {
            (value - next) + *sum
        };
        *sum = next;
        next + *correction
    }
}

impl Prefix for u64 {
    type State = u64;

    fn step(state: &mut u64, value: u64) -> u64 {
        *state = state.wrapping_add(value);
        *state
    }
}

/// Compensated sum with Knuth's branch-free two-sum.
///
/// Table terms alternate in sign with similar magnitudes, so a comparison
/// branch, as in Neumaier summation, is poorly predicted.
#[derive(Default)]
struct TwoSum {
    sum: f64,
    correction: f64,
}

impl TwoSum {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        let part = next - self.sum;
        self.correction += (self.sum - (next - part)) + (value - part);
        self.sum = next;
    }

    fn total(&self) -> f64 {
        self.sum + self.correction
    }
}

/// Most columns per parallel task in the outer-axis prefix passes, and the
/// chunk length of the input scan. It sets only the work split and does not
/// affect results.
const COLUMN_CHUNK: usize = 4096;

/// Output tasks per Rayon thread, for load balance. It sets only the work
/// split and does not affect results.
const TASKS_PER_THREAD: usize = 8;

/// Summed-area tables with shape `shape[a] + 1` per axis.
struct Tables {
    strides: Vec<usize>,
    values: Vec<f64>,
    nan: Option<Vec<u64>>,
    positive: Option<Vec<u64>>,
    negative: Option<Vec<u64>>,
}

impl Tables {
    fn build<T: FilterFloat>(input: &[T], shape: &[usize]) -> Result<Self, Error> {
        // Fixed chunks summed in order keep the overflow check independent
        // of the thread count.
        let (has_nan, has_positive, has_negative, absolute) = input
            .par_chunks(COLUMN_CHUNK)
            .map(|chunk| {
                chunk
                    .iter()
                    .fold((false, false, false, 0.0), |state, value| {
                        let value = value.to_f64();
                        (
                            state.0 | value.is_nan(),
                            state.1 | (value == f64::INFINITY),
                            state.2 | (value == f64::NEG_INFINITY),
                            state.3 + if value.is_finite() { value.abs() } else { 0.0 },
                        )
                    })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .fold((false, false, false, 0.0), |a, b| {
                (a.0 | b.0, a.1 | b.1, a.2 | b.2, a.3 + b.3)
            });
        if !absolute.is_finite() {
            return Err(Error::InvalidConfiguration(
                "the absolute sum of finite inputs overflows f64",
            ));
        }
        let dims: Vec<usize> = shape
            .iter()
            .map(|&length| length.checked_add(1).ok_or(Error::Overflow))
            .collect::<Result<_, _>>()?;
        let mut strides = vec![1usize; dims.len()];
        for axis in (0..dims.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1]
                .checked_mul(dims[axis + 1])
                .ok_or(Error::Overflow)?;
        }
        let total = strides[0].checked_mul(dims[0]).ok_or(Error::Overflow)?;
        let values = table(input, shape, &dims, total, |value: T| {
            let value = value.to_f64();
            if value.is_finite() {
                value
            } else {
                0.0
            }
        })?;
        let count = |wanted: fn(f64) -> bool| {
            table(input, shape, &dims, total, move |value: T| {
                u64::from(wanted(value.to_f64()))
            })
        };
        let nan = has_nan.then(|| count(f64::is_nan)).transpose()?;
        let positive = has_positive
            .then(|| count(|value| value == f64::INFINITY))
            .transpose()?;
        let negative = has_negative
            .then(|| count(|value| value == f64::NEG_INFINITY))
            .transpose()?;
        Ok(Self {
            strides,
            values,
            nan,
            positive,
            negative,
        })
    }
}

/// Build one summed-area table from per-sample values.
fn table<T: FilterFloat, E: Prefix>(
    input: &[T],
    shape: &[usize],
    dims: &[usize],
    total: usize,
    sample: impl Fn(T) -> E + Sync,
) -> Result<Vec<E>, Error> {
    let mut result = Vec::new();
    result.try_reserve_exact(total).map_err(|_| Error::Memory)?;
    result.resize(total, E::default());
    let rank = shape.len();
    let row_len = dims[rank - 1];
    let input_row = shape[rank - 1];
    // Fill each last-axis line and take its prefix sums.
    result
        .par_chunks_mut(row_len)
        .enumerate()
        .for_each(|(row, line)| {
            let mut remaining = row;
            let mut source = 0usize;
            let mut source_stride = input_row;
            for axis in (0..rank - 1).rev() {
                let coordinate = remaining % dims[axis];
                remaining /= dims[axis];
                if coordinate == 0 {
                    return;
                }
                source += (coordinate - 1) * source_stride;
                source_stride *= shape[axis];
            }
            let mut state = E::State::default();
            for (target, &value) in line[1..].iter_mut().zip(&input[source..source + input_row]) {
                *target = E::step(&mut state, sample(value));
            }
        });
    // Prefix sums along every other axis: rows of `inner` entries.
    for axis in (0..rank - 1).rev() {
        let inner: usize = dims[axis + 1..].iter().product();
        let block = inner * dims[axis];
        result
            .par_chunks_mut(block)
            .for_each(|block| prefix_rows(block, inner));
    }
    Ok(result)
}

/// Replace row `k` of `block` (rows of length `inner`) by the sum of rows
/// `0..=k`, column by column.
fn prefix_rows<E: Prefix>(block: &mut [E], inner: usize) {
    // Split columns into independent chunks; each walks all rows.
    let width = COLUMN_CHUNK
        .min(inner.div_ceil(rayon::current_num_threads().max(1)))
        .max(1);
    let mut pieces: Vec<Vec<&mut [E]>> = (0..inner.div_ceil(width)).map(|_| Vec::new()).collect();
    for row in block.chunks_mut(inner) {
        for (piece, part) in pieces.iter_mut().zip(row.chunks_mut(width)) {
            piece.push(part);
        }
    }
    pieces.par_iter_mut().for_each(|piece| {
        let width = piece[0].len();
        let mut state = vec![E::State::default(); width];
        for part in piece.iter_mut() {
            for (value, state) in part.iter_mut().zip(state.iter_mut()) {
                *value = E::step(state, *value);
            }
        }
    });
}

/// Up to this many `(table index, coefficient)` terms per axis.
const MAX_TERMS: usize = 10;

/// Integer combination of axis prefix sums `F(p)`, `p` in `1..=length`.
///
/// Coefficients are built as `i128`, then stored as `f64` factors for the
/// value table and as `u64` weights (two's complement) for the count tables.
struct Terms {
    positions: [usize; MAX_TERMS],
    coefficients: [i128; MAX_TERMS],
    factors: [f64; MAX_TERMS],
    weights: [u64; MAX_TERMS],
    len: usize,
}

impl Terms {
    fn new() -> Self {
        Self {
            positions: [0; MAX_TERMS],
            coefficients: [0; MAX_TERMS],
            factors: [0.0; MAX_TERMS],
            weights: [0; MAX_TERMS],
            len: 0,
        }
    }

    /// Store `F(high) - F(low)` for a window inside the axis.
    fn interior(&mut self, low: usize, high: usize) {
        self.positions[0] = high;
        self.factors[0] = 1.0;
        self.weights[0] = 1;
        self.positions[1] = low;
        self.factors[1] = -1.0;
        self.weights[1] = u64::MAX;
        self.len = if low == 0 { 1 } else { 2 };
    }

    /// Store `E(high) - E(low)` for a window that crosses the boundary.
    fn boundary(&mut self, low: i128, high: i128, length: i128, mode: BoundaryMode) {
        self.len = 0;
        self.extended(high, length, mode, 1);
        self.extended(low, length, mode, -1);
        let mut kept = 0;
        for index in 0..self.len {
            let coefficient = self.coefficients[index];
            if coefficient != 0 {
                self.positions[kept] = self.positions[index];
                self.factors[kept] = coefficient as f64;
                self.weights[kept] = coefficient as u64;
                kept += 1;
            }
        }
        self.len = kept;
    }

    fn push(&mut self, position: i128, coefficient: i128) {
        if position == 0 || coefficient == 0 {
            return;
        }
        let position = position as usize;
        for index in 0..self.len {
            if self.positions[index] == position {
                self.coefficients[index] += coefficient;
                return;
            }
        }
        self.positions[self.len] = position;
        self.coefficients[self.len] = coefficient;
        self.len += 1;
    }

    /// Add `sign` times the prefix sum of the extended axis at index `t`.
    ///
    /// The extended prefix `E(t)` is the sum of extended samples `0..t` for
    /// `t >= 0` and minus the sum of samples `t..0` for `t < 0`, so a window
    /// `[lo, hi)` sums to `E(hi) - E(lo)` in every mode.
    fn extended(&mut self, t: i128, length: i128, mode: BoundaryMode, sign: i128) {
        match mode {
            BoundaryMode::Constant => self.push(t.clamp(0, length), sign),
            BoundaryMode::Nearest => {
                if t < 0 {
                    self.push(1, sign * t);
                } else if t <= length {
                    self.push(t, sign);
                } else {
                    let excess = t - length;
                    self.push(length, sign * (1 + excess));
                    self.push(length - 1, -sign * excess);
                }
            }
            BoundaryMode::Wrap => {
                self.push(length, sign * t.div_euclid(length));
                self.push(t.rem_euclid(length), sign);
            }
            BoundaryMode::Reflect => {
                let period = 2 * length;
                let (cycles, rest) = (t.div_euclid(period), t.rem_euclid(period));
                self.push(length, sign * 2 * cycles);
                if rest <= length {
                    self.push(rest, sign);
                } else {
                    self.push(length, 2 * sign);
                    self.push(period - rest, -sign);
                }
            }
            BoundaryMode::Mirror if length == 1 => self.push(1, sign * t),
            BoundaryMode::Mirror => {
                // One period is x[0..L] followed by x[L-2], ..., x[1].
                let period = 2 * length - 2;
                let (cycles, rest) = (t.div_euclid(period), t.rem_euclid(period));
                self.push(length, sign * cycles);
                self.push(1, -sign * cycles);
                self.push(length - 1, sign * cycles);
                if rest <= length {
                    self.push(rest, sign);
                } else {
                    self.push(length, sign);
                    self.push(length - 1, sign);
                    self.push(2 * length - 1 - rest, -sign);
                }
            }
        }
    }
}

struct Query<'a> {
    shape: &'a [usize],
    modes: &'a [BoundaryMode],
    window: &'a VariableWindow,
    tables: &'a Tables,
    policy: InvalidPolicy,
    cval: f64,
    len: usize,
}

/// Per-chunk scratch space for [`Query::pixel`].
struct Scratch {
    coordinates: Vec<usize>,
    choice: Vec<usize>,
    terms: Vec<Terms>,
    /// Table offsets of the low and high window ends on interior axes.
    lows: Vec<usize>,
    highs: Vec<usize>,
}

impl Query<'_> {
    fn run<T: FilterFloat>(&self, start: usize, output: &mut [T]) {
        let rank = self.shape.len();
        let mut scratch = Scratch {
            coordinates: vec![0; rank],
            choice: vec![0; rank],
            terms: (0..rank).map(|_| Terms::new()).collect(),
            lows: vec![0; rank],
            highs: vec![0; rank],
        };
        let mut remaining = start;
        for axis in (0..rank).rev() {
            scratch.coordinates[axis] = remaining % self.shape[axis];
            remaining /= self.shape[axis];
        }
        for (offset, value) in output.iter_mut().enumerate() {
            *value = self.pixel(start + offset, &mut scratch);
            for axis in (0..rank).rev() {
                scratch.coordinates[axis] += 1;
                if scratch.coordinates[axis] < self.shape[axis] {
                    break;
                }
                scratch.coordinates[axis] = 0;
            }
        }
    }

    fn pixel<T: FilterFloat>(&self, index: usize, scratch: &mut Scratch) -> T {
        let rank = self.shape.len();
        let Scratch {
            coordinates,
            choice,
            terms,
            lows,
            highs,
        } = scratch;
        let tables = self.tables;
        let mut volume = 1usize;
        let mut inside = 1usize;
        let mut interior = true;
        for axis in 0..rank {
            let size = self.window.size(axis, index, self.len);
            let length = self.shape[axis];
            let center = coordinates[axis];
            volume *= size;
            match center.checked_sub(size / 2) {
                Some(low) if size <= length - low => {
                    terms[axis].interior(low, low + size);
                    lows[axis] = low * tables.strides[axis];
                    highs[axis] = (low + size) * tables.strides[axis];
                    inside *= size;
                }
                _ => {
                    interior = false;
                    let low = center as i128 - (size / 2) as i128;
                    let high = low + size as i128;
                    let mode = self.modes[axis];
                    terms[axis].boundary(low, high, length as i128, mode);
                    inside *= if mode == BoundaryMode::Constant {
                        (high.min(length as i128) - low.max(0)).max(0) as usize
                    } else {
                        size
                    };
                }
            }
        }
        let counted =
            tables.nan.is_some() || tables.positive.is_some() || tables.negative.is_some();
        let mut sum = TwoSum::default();
        let (mut nan, mut positive, mut negative) = (0u64, 0u64, 0u64);
        if interior && !counted {
            // Inclusion-exclusion over the 2^rank window corners.
            for corner in 0..1usize << rank {
                let mut position = 0usize;
                let mut lower = false;
                for axis in 0..rank {
                    if corner >> axis & 1 == 1 {
                        position += highs[axis];
                    } else {
                        position += lows[axis];
                        lower = !lower;
                    }
                }
                let value = tables.values[position];
                sum.add(if lower { -value } else { value });
            }
        } else if inside != 0 && terms.iter().all(|terms| terms.len != 0) {
            choice.fill(0);
            'outer: loop {
                let mut position = 0usize;
                let mut factor = 1.0f64;
                for axis in 0..rank {
                    let terms = &terms[axis];
                    position += terms.positions[choice[axis]] * tables.strides[axis];
                    factor *= terms.factors[choice[axis]];
                }
                sum.add(factor * tables.values[position]);
                if counted {
                    let weight = (0..rank).fold(1u64, |weight, axis| {
                        weight.wrapping_mul(terms[axis].weights[choice[axis]])
                    });
                    let count = |table: &Option<Vec<u64>>| {
                        table
                            .as_ref()
                            .map_or(0, |table| weight.wrapping_mul(table[position]))
                    };
                    nan = nan.wrapping_add(count(&tables.nan));
                    positive = positive.wrapping_add(count(&tables.positive));
                    negative = negative.wrapping_add(count(&tables.negative));
                }
                for axis in (0..rank).rev() {
                    choice[axis] += 1;
                    if choice[axis] < terms[axis].len {
                        continue 'outer;
                    }
                    choice[axis] = 0;
                }
                break;
            }
        }
        let outside = (volume - inside) as u64;
        let mut finite = inside as u64 - nan - positive - negative;
        let mut total = sum.total();
        if outside != 0 {
            let cval = self.cval;
            if cval.is_nan() {
                nan += outside;
            } else if cval.is_finite() {
                finite += outside;
                total += cval * outside as f64;
            } else if cval > 0.0 {
                positive += outside;
            } else {
                negative += outside;
            }
        }
        let (nan, retained) = match self.policy {
            InvalidPolicy::Propagate => (nan, finite + positive + negative),
            InvalidPolicy::Omit { ignore_inf, .. } => {
                if ignore_inf {
                    positive = 0;
                    negative = 0;
                }
                (0, finite + positive + negative)
            }
        };
        if nan != 0
            || retained == 0
            || matches!(self.policy, InvalidPolicy::Omit { min_count, .. } if retained < min_count as u64)
            || (positive != 0 && negative != 0)
        {
            T::nan()
        } else if positive != 0 {
            T::from_f64(f64::INFINITY)
        } else if negative != 0 {
            T::from_f64(f64::NEG_INFINITY)
        } else {
            T::from_f64(total / finite as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{variable_mean, VariableWindow};
    use crate::boundary::{map_index, BoundaryMode};
    use crate::error::Error;
    use crate::reduction::InvalidPolicy;

    const MODES: [BoundaryMode; 5] = [
        BoundaryMode::Reflect,
        BoundaryMode::Constant,
        BoundaryMode::Nearest,
        BoundaryMode::Mirror,
        BoundaryMode::Wrap,
    ];

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }
    }

    /// Visit every window sample directly and average with the crate's rules.
    fn brute(
        input: &[f64],
        shape: &[usize],
        sizes: &[usize],
        per_axis: bool,
        modes: &[BoundaryMode],
        policy: InvalidPolicy,
        cval: f64,
    ) -> Vec<f64> {
        let len = input.len();
        let rank = shape.len();
        let mut output = vec![0.0; len];
        for (index, result) in output.iter_mut().enumerate() {
            let mut coordinates = vec![0usize; rank];
            let mut remaining = index;
            for axis in (0..rank).rev() {
                coordinates[axis] = remaining % shape[axis];
                remaining /= shape[axis];
            }
            let size = |axis: usize| {
                if per_axis {
                    sizes[axis * len + index]
                } else {
                    sizes[index]
                }
            };
            let mut mean = crate::mean::Mean::default();
            let mut tap = vec![0usize; rank];
            'taps: loop {
                let mut flat = Some(0usize);
                for axis in 0..rank {
                    let position =
                        coordinates[axis] as i128 + tap[axis] as i128 - (size(axis) / 2) as i128;
                    match map_index(position, shape[axis], modes[axis]).unwrap() {
                        Some(mapped) => {
                            flat = flat.map(|flat| flat * shape[axis] + mapped);
                        }
                        None => flat = None,
                    }
                }
                mean.add(flat.map_or(cval, |flat| input[flat]), policy);
                for axis in (0..rank).rev() {
                    tap[axis] += 1;
                    if tap[axis] < size(axis) {
                        continue 'taps;
                    }
                    tap[axis] = 0;
                }
                break;
            }
            *result = mean.finish::<f64>(policy).unwrap();
        }
        output
    }

    fn same(left: &[f64], right: &[f64]) {
        for (index, (&a, &b)) in left.iter().zip(right).enumerate() {
            let equal = (a.is_nan() && b.is_nan())
                || a == b
                || (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()));
            assert!(equal, "index {index}: {a} != {b}");
        }
    }

    fn random_case(shape: &[usize], seed: u64, special: bool) {
        let mut rng = Lcg(seed);
        let len: usize = shape.iter().product();
        let input: Vec<f64> = (0..len)
            .map(|_| match rng.next() % 23 {
                0 if special => f64::NAN,
                1 if special => f64::INFINITY,
                2 if special => f64::NEG_INFINITY,
                _ => (rng.next() % 2001) as f64 / 7.0 - 100.0,
            })
            .collect();
        let max_size = shape.iter().max().unwrap() * 2 + 3;
        for per_axis in [false, true] {
            let count = if per_axis { len * shape.len() } else { len };
            let sizes: Vec<usize> = (0..count)
                .map(|_| 1 + (rng.next() as usize) % max_size)
                .collect();
            for mode in MODES {
                for (modes, cval) in [
                    (vec![mode; shape.len()], 2.5),
                    (
                        (0..shape.len()).map(|axis| MODES[(axis + 1) % 5]).collect(),
                        f64::NAN,
                    ),
                ] {
                    let window = if per_axis {
                        VariableWindow::per_axis(sizes.clone())
                    } else {
                        VariableWindow::isotropic(sizes.clone())
                    }
                    .modes(modes.clone());
                    for policy in [
                        InvalidPolicy::Propagate,
                        InvalidPolicy::Omit {
                            ignore_inf: false,
                            min_count: 1,
                        },
                        InvalidPolicy::Omit {
                            ignore_inf: true,
                            min_count: 3,
                        },
                    ] {
                        let fast = variable_mean(&input, shape, &window, policy, cval).unwrap();
                        let slow = brute(&input, shape, &sizes, per_axis, &modes, policy, cval);
                        same(&fast, &slow);
                    }
                }
            }
        }
    }

    #[test]
    fn matches_direct_windows_in_one_to_three_dimensions() {
        for (seed, shape) in [
            (1, vec![1]),
            (2, vec![2]),
            (3, vec![9]),
            (4, vec![4, 5]),
            (5, vec![1, 6]),
            (6, vec![3, 2, 4]),
        ] {
            random_case(&shape, seed, false);
            random_case(&shape, seed + 100, true);
        }
    }

    #[test]
    fn wide_rows_use_several_column_chunks() {
        let shape = [3, super::COLUMN_CHUNK + 7];
        let len = shape[0] * shape[1];
        let input: Vec<f64> = (0..len).map(|index| (index % 97) as f64 - 40.0).collect();
        let sizes: Vec<usize> = (0..len).map(|index| 1 + index % 4).collect();
        let modes = [BoundaryMode::Reflect, BoundaryMode::Wrap];
        let window = VariableWindow::isotropic(sizes.clone()).modes(modes);
        let policy = InvalidPolicy::Propagate;
        let fast = variable_mean(&input, &shape, &window, policy, 0.0).unwrap();
        same(
            &fast,
            &brute(&input, &shape, &sizes, false, &modes, policy, 0.0),
        );
    }

    #[test]
    fn results_do_not_depend_on_the_thread_count() {
        let shape = [37, 2 * super::COLUMN_CHUNK + 5];
        let len = shape[0] * shape[1];
        let mut rng = Lcg(9);
        let input: Vec<f64> = (0..len)
            .map(|_| (rng.next() % 100_003) as f64 * 1.0e-3 + 1.0e4)
            .collect();
        let sizes: Vec<usize> = (0..len).map(|_| 1 + (rng.next() as usize) % 40).collect();
        let window = VariableWindow::isotropic(sizes).mode(BoundaryMode::Mirror);
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    variable_mean(&input, &shape, &window, InvalidPolicy::Propagate, 0.0).unwrap()
                })
        };
        let serial = run(1);
        for threads in [2, 3, 8] {
            let parallel = run(threads);
            assert!(serial
                .iter()
                .zip(&parallel)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    #[test]
    fn rejects_invalid_configurations() {
        let input = [1.0_f64; 4];
        let policy = InvalidPolicy::Propagate;
        let window = VariableWindow::isotropic([1, 1, 0, 1]);
        assert!(matches!(
            variable_mean(&input, &[4], &window, policy, 0.0),
            Err(Error::InvalidConfiguration(_))
        ));
        let window = VariableWindow::isotropic([1, 1, 1]);
        assert_eq!(
            variable_mean(&input, &[4], &window, policy, 0.0),
            Err(Error::LengthMismatch)
        );
        let window = VariableWindow::per_axis([1; 4]).modes([BoundaryMode::Wrap; 2]);
        assert!(matches!(
            variable_mean(&input, &[2, 2], &window, policy, 0.0),
            Err(Error::LengthMismatch)
        ));
        let window = VariableWindow::isotropic([1; 4]).modes([BoundaryMode::Wrap; 1]);
        assert!(matches!(
            variable_mean(&input, &[2, 2], &window, policy, 0.0),
            Err(Error::InvalidConfiguration(_))
        ));
        let omit = InvalidPolicy::Omit {
            ignore_inf: false,
            min_count: 0,
        };
        let window = VariableWindow::isotropic([1; 4]);
        assert!(matches!(
            variable_mean(&input, &[4], &window, omit, 0.0),
            Err(Error::InvalidConfiguration(_))
        ));
        let window = VariableWindow::isotropic([usize::MAX; 4]);
        assert_eq!(
            variable_mean(&input, &[2, 2], &window, policy, 0.0),
            Err(Error::Overflow)
        );
        let huge = [f64::MAX, f64::MAX];
        let window = VariableWindow::isotropic([1, 1]);
        assert!(matches!(
            variable_mean(&huge, &[2], &window, policy, 0.0),
            Err(Error::InvalidConfiguration(_))
        ));
        let empty: [f64; 0] = [];
        let window = VariableWindow::isotropic([]);
        assert_eq!(
            variable_mean(&empty, &[0, 3], &window, policy, 0.0),
            Ok(vec![])
        );
    }

    #[test]
    fn single_precision_rounds_the_f64_mean() {
        let input = [1.0_f32, 2.0, 4.0];
        let window = VariableWindow::isotropic([3, 1, 2]).mode(BoundaryMode::Nearest);
        let out = variable_mean(&input, &[3], &window, InvalidPolicy::Propagate, 0.0).unwrap();
        assert_eq!(out, [4.0 / 3.0, 2.0, 3.0]);
    }
}
