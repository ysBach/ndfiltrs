//! Rectangular means with exact, removable integer sums.

use rayon::prelude::*;

use crate::boundary::{map_index, BoundaryMode};
use crate::error::KernelError;
use crate::exact_sum::{ExactFloat, ExactSum};
use crate::reduction::InvalidPolicy;

#[derive(Clone, Copy, Default)]
struct Counts {
    finite: usize,
    nan: usize,
    positive: usize,
    negative: usize,
}

impl Counts {
    fn sample(value: f64, policy: InvalidPolicy) -> Self {
        let mut result = Self::default();
        if value.is_nan() {
            result.nan = usize::from(matches!(policy, InvalidPolicy::Propagate));
        } else if value.is_finite() {
            result.finite = 1;
        } else if !matches!(
            policy,
            InvalidPolicy::Omit {
                ignore_inf: true,
                ..
            }
        ) {
            result.positive = usize::from(value.is_sign_positive());
            result.negative = usize::from(value.is_sign_negative());
        }
        result
    }

    fn add(&mut self, other: Self) {
        self.finite += other.finite;
        self.nan += other.nan;
        self.positive += other.positive;
        self.negative += other.negative;
    }

    fn subtract(&mut self, other: Self) {
        self.finite -= other.finite;
        self.nan -= other.nan;
        self.positive -= other.positive;
        self.negative -= other.negative;
    }

    fn multiply(&mut self, factor: usize) {
        self.finite *= factor;
        self.nan *= factor;
        self.positive *= factor;
        self.negative *= factor;
    }

    fn value<T: ExactFloat>(self, words: &[u64], low_word: usize, policy: InvalidPolicy) -> T {
        let retained = self.finite + self.positive + self.negative;
        if self.nan != 0
            || retained == 0
            || matches!(policy, InvalidPolicy::Omit { min_count, .. } if retained < min_count)
            || (self.positive != 0 && self.negative != 0)
        {
            T::nan()
        } else if self.positive != 0 {
            T::from_f64(f64::INFINITY)
        } else if self.negative != 0 {
            T::from_f64(f64::NEG_INFINITY)
        } else {
            ExactSum::mean_words::<T>(words, low_word, self.finite)
        }
    }
}

fn zeros<T: Clone + Default>(length: usize) -> Result<Vec<T>, KernelError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(length)
        .map_err(|_| KernelError::Memory)?;
    result.resize(length, T::default());
    Ok(result)
}

/// Add or remove a two's-complement integer within a proven nonoverflowing width.
fn combine(target: &mut [u64], source: &[u64], subtract: bool) {
    let mut carry = subtract;
    for (target, &source) in target.iter_mut().zip(source) {
        let source = if subtract { !source } else { source };
        let (sum, first) = target.overflowing_add(source);
        let (sum, second) = sum.overflowing_add(u64::from(carry));
        *target = sum;
        carry = first || second;
    }
}

fn multiply(words: &mut [u64], factor: usize) {
    let mut carry = 0u128;
    for word in words {
        let product = u128::from(*word) * factor as u128 + carry;
        *word = product as u64;
        carry = product >> u64::BITS;
    }
}

/// Find the minimum exact signed word span for every intermediate window sum.
fn word_span<T: ExactFloat>(
    input: &[T],
    cval: T,
    include_padding: bool,
    volume: usize,
) -> (usize, usize) {
    let mut lowest = usize::MAX;
    let mut highest = 0usize;
    for value in input.iter().copied().chain(include_padding.then_some(cval)) {
        let value = value.to_f64();
        if !value.is_finite() || value == 0.0 {
            continue;
        }
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as usize;
        let fraction = bits & ((1u64 << 52) - 1);
        let mantissa = fraction | if exponent == 0 { 0 } else { 1u64 << 52 };
        let shift = exponent.saturating_sub(1);
        lowest = lowest.min(shift + mantissa.trailing_zeros() as usize);
        highest = highest.max(shift + (63 - mantissa.leading_zeros()) as usize);
    }
    if lowest == usize::MAX {
        return (0, 1);
    }
    let count_bits = (usize::BITS - (volume - 1).leading_zeros()) as usize;
    // A sum of at most volume terms has magnitude below 2^(highest+1+count_bits).
    // One more bit stores the sign. All preceding axis supports are smaller.
    let low_word = lowest / u64::BITS as usize;
    let end = (highest + 1 + count_bits + 1).div_ceil(u64::BITS as usize);
    (low_word, end - low_word)
}

/// Average rectangular neighborhoods with no rounding between axis passes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mean_into<T: ExactFloat>(
    input: &[T],
    shape: &[usize],
    size: &[usize],
    origin: &[isize],
    modes: &[BoundaryMode],
    cval: T,
    policy: InvalidPolicy,
    output: &mut [T],
) -> Result<(), KernelError> {
    let rank = shape.len();
    if rank == 0
        || size.len() != rank
        || origin.len() != rank
        || modes.len() != rank
        || size.contains(&0)
    {
        return Err(KernelError::InvalidConfiguration(
            "invalid rectangular mean geometry",
        ));
    }
    let product = |items: &[usize]| {
        items.iter().try_fold(1usize, |a, &b| {
            a.checked_mul(b).ok_or(KernelError::Overflow)
        })
    };
    let length = product(shape)?;
    let volume = product(size)?;
    if length != input.len() || length != output.len() {
        return Err(KernelError::LengthMismatch);
    }
    if matches!(policy, InvalidPolicy::Omit { min_count, .. } if min_count == 0 || min_count > volume)
    {
        return Err(KernelError::InvalidConfiguration(
            "min_count must be within window support",
        ));
    }
    let mut anchors = zeros::<i128>(rank)?;
    for axis in 0..rank {
        anchors[axis] = size[axis] as i128 / 2 + origin[axis] as i128;
        if !(0..size[axis] as i128).contains(&anchors[axis]) {
            return Err(KernelError::InvalidConfiguration(
                "origin must keep the anchor in the window",
            ));
        }
    }
    if length == 0 {
        return Ok(());
    }
    let (low_word, words) = word_span(input, cval, modes.contains(&BoundaryMode::Constant), volume);
    let storage = length.checked_mul(words).ok_or(KernelError::Overflow)?;
    let mut previous = zeros::<u64>(storage)?;
    let mut next = zeros::<u64>(storage)?;
    let mut previous_counts = zeros::<Counts>(length)?;
    let mut next_counts = zeros::<Counts>(length)?;
    previous
        .par_chunks_mut(words)
        .zip(previous_counts.par_iter_mut())
        .zip(input.par_iter())
        .for_each(|((state, counts), &value)| {
            *counts = Counts::sample(value.to_f64(), policy);
            if value.is_finite() {
                ExactSum::encode_finite_words(value.to_f64(), low_word, state);
            }
        });
    let mut padding = zeros::<u64>(words)?;
    let mut padding_counts = Counts::sample(cval.to_f64(), policy);
    if cval.is_finite() && modes.contains(&BoundaryMode::Constant) {
        ExactSum::encode_finite_words(cval.to_f64(), low_word, &mut padding);
    }
    let mut strides = zeros::<usize>(rank)?;
    let mut stride = 1usize;
    for axis in (0..rank).rev() {
        strides[axis] = stride;
        stride *= shape[axis];
    }
    for axis in 0..rank {
        let axis_length = shape[axis];
        let axis_stride = strides[axis];
        // Every parallel line has the same boundary geometry. Map each
        // entering/leaving coordinate once per axis, not once per line.
        let mut updates = zeros::<(Option<usize>, Option<usize>)>(axis_length - 1)?;
        for (coordinate, update) in updates.iter_mut().enumerate() {
            *update = (
                map_index(coordinate as i128 - anchors[axis], axis_length, modes[axis])?,
                map_index(
                    coordinate as i128 + size[axis] as i128 - anchors[axis],
                    axis_length,
                    modes[axis],
                )?,
            );
        }
        next.par_chunks_mut(axis_length * words)
            .zip(next_counts.par_chunks_mut(axis_length))
            .enumerate()
            .try_for_each(|(line, (destinations, counts))| {
                let mut remaining = line;
                let mut base = 0;
                for other in (0..rank).rev().filter(|&other| other != axis) {
                    base += (remaining % shape[other]) * strides[other];
                    remaining /= shape[other];
                }
                let sample = |mapped: Option<usize>| -> (&[u64], Counts) {
                    match mapped {
                        Some(index) => {
                            let source = base + index * axis_stride;
                            (
                                &previous[source * words..(source + 1) * words],
                                previous_counts[source],
                            )
                        }
                        None => (&padding, padding_counts),
                    }
                };
                let mut sum = zeros::<u64>(words)?;
                let mut count = Counts::default();
                for tap in 0..size[axis] {
                    let (state, retained) = sample(map_index(
                        tap as i128 - anchors[axis],
                        axis_length,
                        modes[axis],
                    )?);
                    combine(&mut sum, state, false);
                    count.add(retained);
                }
                for (coordinate, (destination, retained)) in
                    destinations.chunks_exact_mut(words).zip(counts).enumerate()
                {
                    destination.copy_from_slice(&sum);
                    *retained = count;
                    if let Some(&(leaving_index, entering_index)) = updates.get(coordinate) {
                        let (leaving, removed) = sample(leaving_index);
                        combine(&mut sum, leaving, true);
                        count.subtract(removed);
                        let (entering, added) = sample(entering_index);
                        combine(&mut sum, entering, false);
                        count.add(added);
                    }
                }
                Ok::<(), KernelError>(())
            })?;
        std::mem::swap(&mut previous, &mut next);
        std::mem::swap(&mut previous_counts, &mut next_counts);
        // Intermediate passes store the filtered axis last, making lines disjoint.
        // The final pass restores the original last axis and C order.
        strides[axis] = 1;
        let mut stride = axis_length;
        for other in (0..rank).rev().filter(|&other| other != axis) {
            strides[other] = stride;
            stride *= shape[other];
        }
        multiply(&mut padding, size[axis]);
        padding_counts.multiply(size[axis]);
    }
    output
        .par_iter_mut()
        .zip(previous.par_chunks(words))
        .zip(previous_counts.par_iter())
        .for_each(|((destination, state), &counts)| {
            *destination = counts.value(state, low_word, policy);
        });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::window::Window;

    fn check_axis_passes<T: ExactFloat>() {
        let all_modes = [
            BoundaryMode::Reflect,
            BoundaryMode::Constant,
            BoundaryMode::Mirror,
            BoundaryMode::Nearest,
            BoundaryMode::Wrap,
        ];
        for shape in [vec![5], vec![3, 5], vec![2, 3, 5], vec![2, 2, 3, 5]] {
            let rank = shape.len();
            let size: Vec<_> = (0..rank).map(|axis| 2 + axis % 2).collect();
            let origin: Vec<_> = size.iter().map(|&n| if n == 2 { -1 } else { 1 }).collect();
            let length = shape.iter().product();
            let input: Vec<T> = (0..length)
                .map(|index| {
                    T::from_f64(match index % 11 {
                        0 => f64::NAN,
                        1 => f64::INFINITY,
                        2 => f64::NEG_INFINITY,
                        3 => -0.0,
                        4 => 1.0e20,
                        5 => -1.0e20,
                        _ => index as f64 - 7.0,
                    })
                })
                .collect();
            for mode_offset in 0..all_modes.len() {
                let modes: Vec<_> = (0..rank)
                    .map(|axis| all_modes[(axis + mode_offset) % all_modes.len()])
                    .collect();
                let window =
                    Window::new(shape.clone(), &size, None, &origin, modes.clone()).unwrap();
                for policy in [
                    InvalidPolicy::Propagate,
                    InvalidPolicy::Omit {
                        ignore_inf: false,
                        min_count: 1,
                    },
                    InvalidPolicy::Omit {
                        ignore_inf: true,
                        min_count: 2,
                    },
                ] {
                    for cval in [T::from_f64(-0.0), T::from_f64(4.0), T::nan()] {
                        let mut actual = vec![T::nan(); length];
                        mean_into(
                            &input,
                            &shape,
                            &size,
                            &origin,
                            &modes,
                            cval,
                            policy,
                            &mut actual,
                        )
                        .unwrap();
                        let mut index = vec![0; rank];
                        let mut samples = Vec::new();
                        for (center, value) in actual.iter().enumerate() {
                            // Independent direct N-D gathering, not a sequence
                            // of rolling passes, is the traversal oracle.
                            samples.clear();
                            window
                                .gather(&input, center, cval, &mut index, &mut samples, |x| x)
                                .unwrap();
                            let mut sum = ExactSum::default();
                            let mut counts = Counts::default();
                            for &sample in &samples {
                                counts.add(Counts::sample(sample.to_f64(), policy));
                                if sample.is_finite() {
                                    sum.add_f64(sample.to_f64());
                                }
                            }
                            let expected = counts.value::<T>(sum.words(), 0, policy);
                            assert_eq!(value.to_f64().to_bits(), expected.to_f64().to_bits(),
                                "shape={shape:?}, center={center}, modes={modes:?}, policy={policy:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn axis_passes_match_direct_exact_windows() {
        check_axis_passes::<f32>();
        check_axis_passes::<f64>();
    }
}
