use rayon::prelude::*;
use reducers::Float;

use crate::boundary::{map_index, BoundaryMode};
use crate::convolution::{finite_weighted, WeightedSum};
use crate::error::KernelError;
use crate::mean::Sum;

#[derive(Clone, Copy, Default)]
struct Sample {
    mean: f64,
    mass: f64,
    count: usize,
    nan: bool,
    positive_inf: bool,
    negative_inf: bool,
}

impl Sample {
    fn new(value: f64, omit_nan: bool, ignore_inf: bool) -> Self {
        if value.is_nan() || (ignore_inf && value.is_infinite()) {
            return Self {
                nan: !omit_nan,
                ..Self::default()
            };
        }
        Self {
            mean: if value.is_finite() { value } else { 0.0 },
            mass: if value.is_finite() { 1.0 } else { 0.0 },
            count: 1,
            nan: false,
            positive_inf: value == f64::INFINITY,
            negative_inf: value == f64::NEG_INFINITY,
        }
    }

    fn value(self, min_count: usize) -> f64 {
        if self.nan || self.count < min_count || (self.positive_inf && self.negative_inf) {
            f64::NAN
        } else if self.positive_inf {
            f64::INFINITY
        } else if self.negative_inf {
            f64::NEG_INFINITY
        } else if self.mass > 0.0 {
            self.mean
        } else {
            f64::NAN
        }
    }
}

/// Smooth along selected axes while carrying retained mass and exact counts.
/// Means stay within each retained neighborhood's range, even near f64::MAX.
#[allow(clippy::too_many_arguments)]
pub(crate) fn smooth_into<T: Float + Send + Sync>(
    input: &[T],
    shape: &[usize],
    axes: &[usize],
    kernels: &[Vec<f64>],
    modes: &[BoundaryMode],
    cval: T,
    omit_nan: bool,
    ignore_inf: bool,
    min_count: usize,
    output: &mut [T],
) -> Result<(), KernelError> {
    let invalid = || KernelError::InvalidConfiguration("invalid separable smoothing parameters");
    if shape.is_empty()
        || axes.is_empty()
        || axes.len() != kernels.len()
        || modes.len() != shape.len()
    {
        return Err(invalid());
    }
    if !omit_nan && (ignore_inf || min_count != 1) {
        return Err(invalid());
    }
    let mut strides = vec![1usize; shape.len()];
    let mut length = 1usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = length;
        length = length
            .checked_mul(shape[axis])
            .ok_or(KernelError::Overflow)?;
    }
    if input.len() != length || output.len() != length {
        return Err(KernelError::LengthMismatch);
    }
    let mut support = 1usize;
    let mut smallest_product = 1.0;
    for (index, (&axis, kernel)) in axes.iter().zip(kernels).enumerate() {
        if axis >= shape.len()
            || axes[..index].contains(&axis)
            || kernel.is_empty()
            || kernel.len() % 2 == 0
        {
            return Err(invalid());
        }
        if kernel.iter().any(|&w| !(0.0..=1.0).contains(&w)) {
            return Err(invalid());
        }
        let count = kernel.iter().filter(|&&w| w > 0.0).count();
        if count == 0 {
            return Err(invalid());
        }
        support = support.checked_mul(count).ok_or(KernelError::Overflow)?;
        smallest_product *= kernel
            .iter()
            .copied()
            .filter(|&w| w > 0.0)
            .fold(1.0, f64::min);
    }
    if smallest_product < f64::MIN_POSITIVE {
        return Err(KernelError::InvalidConfiguration(
            "separable product weights underflow; use method='direct'",
        ));
    }
    if min_count == 0 || min_count > support {
        return Err(KernelError::InvalidConfiguration(
            "min_count exceeds positive kernel support",
        ));
    }
    if input.is_empty() {
        return Ok(());
    }
    let mut previous = Vec::new();
    previous
        .try_reserve_exact(length)
        .map_err(|_| KernelError::Memory)?;
    previous.extend(
        input
            .iter()
            .map(|&value| Sample::new(value.to_f64(), omit_nan, ignore_inf)),
    );
    let mut next = Vec::new();
    next.try_reserve_exact(length)
        .map_err(|_| KernelError::Memory)?;
    next.resize(length, Sample::default());
    let mut padding = Sample::new(cval.to_f64(), omit_nan, ignore_inf);
    for (&axis, kernel) in axes.iter().zip(kernels) {
        let mut total = Sum::default();
        for &weight in kernel {
            total.add(weight);
        }
        let total = total.total();
        let stride = strides[axis];
        let axis_length = shape[axis];
        let radius = kernel.len() / 2;
        let mode = modes[axis];
        next.par_iter_mut().enumerate().try_for_each_init(
            Vec::new,
            |pairs, (flat, destination)| {
                let coordinate = (flat / stride) % axis_length;
                // The checked shape product and these bounds ensure both the
                // starting index and every tap offset fit within the array.
                let interior_start = (coordinate >= radius && radius < axis_length - coordinate)
                    .then(|| flat + radius * stride);
                let sample_at = |tap: usize| -> Result<Sample, KernelError> {
                    if let Some(start) = interior_start {
                        return Ok(previous[start - tap * stride]);
                    }
                    let source = coordinate as i128 + radius as i128 - tap as i128;
                    Ok(match map_index(source, axis_length, mode)? {
                        Some(mapped) => {
                            let offset = if mapped >= coordinate {
                                flat + (mapped - coordinate) * stride
                            } else {
                                flat - (coordinate - mapped) * stride
                            };
                            previous[offset]
                        }
                        None => padding,
                    })
                };
                let mut result = Sample::default();
                let mut accumulator = WeightedSum::new(true);
                for (tap, &weight) in kernel.iter().enumerate() {
                    if weight == 0.0 {
                        continue;
                    }
                    let sample = sample_at(tap)?;
                    result.count = result
                        .count
                        .checked_add(sample.count)
                        .ok_or(KernelError::Overflow)?;
                    result.nan |= sample.nan;
                    result.positive_inf |= sample.positive_inf;
                    result.negative_inf |= sample.negative_inf;
                    let retained = weight * sample.mass;
                    if retained > 0.0 {
                        accumulator.add(sample.mean, retained);
                    }
                }
                let retained_mass = accumulator.weight_total();
                result.mass = (retained_mass / total).clamp(0.0, 1.0);
                result.mean = if let Some(mean) = accumulator.finish() {
                    mean
                } else {
                    // Replay only exceptional neighborhoods. The retained
                    // pairs and their order are identical to the original path.
                    pairs.clear();
                    pairs
                        .try_reserve(kernel.len())
                        .map_err(|_| KernelError::Memory)?;
                    for (tap, &weight) in kernel.iter().enumerate() {
                        if weight == 0.0 {
                            continue;
                        }
                        let sample = sample_at(tap)?;
                        let retained = weight * sample.mass;
                        if retained > 0.0 {
                            pairs.push((sample.mean, retained));
                        }
                    }
                    finite_weighted(pairs, true)
                };
                *destination = result;
                Ok::<(), KernelError>(())
            },
        )?;
        std::mem::swap(&mut previous, &mut next);
        padding.count = padding
            .count
            .checked_mul(kernel.iter().filter(|&&w| w > 0.0).count())
            .ok_or(KernelError::Overflow)?;
    }
    output
        .par_iter_mut()
        .zip(previous)
        .for_each(|(destination, sample)| {
            *destination = T::from_f64(sample.value(min_count));
        });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::smooth_into;
    use crate::boundary::BoundaryMode;

    #[test]
    fn asymmetric_weights_keep_orientation_across_strided_interiors_and_edges() {
        let mut output = [0.0; 6];
        smooth_into(
            &[0.0, 7.0, 14.0, 21.0, 28.0, 35.0],
            &[3, 2],
            &[0],
            &[vec![0.25, 0.5, 1.0]],
            &[BoundaryMode::Reflect; 2],
            0.0,
            false,
            false,
            1,
            &mut output,
        )
        .unwrap();
        // Each column uses (next + 2 * center + 4 * previous) / 7,
        // repeating the edge value at reflected boundaries.
        assert_eq!(output, [2.0, 9.0, 8.0, 15.0, 20.0, 27.0]);
    }

    #[test]
    fn finite_averages_preserve_cancellation_and_extreme_constants() {
        let tiny = f64::from_bits(1);
        for (input, kernel, expected) in [
            ([1e16, 1.0, -1e16], vec![1.0; 3], 1.0 / 3.0),
            (
                [f64::MAX, f64::MAX, -f64::MAX],
                vec![1.0; 3],
                f64::MAX / 3.0,
            ),
            ([tiny; 3], vec![0.25, 0.5, 0.25], tiny),
            ([-tiny; 3], vec![0.25, 0.5, 0.25], -tiny),
            ([f64::MAX; 3], vec![0.25, 0.5, 0.25], f64::MAX),
        ] {
            let mut output = [0.0; 3];
            smooth_into(
                &input,
                &[3],
                &[0],
                &[kernel],
                &[BoundaryMode::Wrap],
                0.0,
                false,
                false,
                1,
                &mut output,
            )
            .unwrap();
            // Every wrapped window contains the same three values. The
            // finite mean must recover the small term and preserve constants.
            for value in output {
                assert_eq!(value.to_bits(), expected.to_bits(), "{input:?}");
            }
        }
    }

    #[test]
    fn missing_samples_and_infinities_keep_distinct_counts_and_signs() {
        let input = [f64::NAN, 2.0, f64::INFINITY, 4.0, f64::NEG_INFINITY];
        for (omit_nan, ignore_inf, min_count, expected) in [
            (true, true, 2, [f64::NAN, f64::NAN, 3.0, f64::NAN, f64::NAN]),
            (
                true,
                false,
                1,
                [
                    2.0,
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NAN,
                    f64::NEG_INFINITY,
                ],
            ),
            (
                false,
                false,
                1,
                [
                    f64::NAN,
                    f64::NAN,
                    f64::INFINITY,
                    f64::NAN,
                    f64::NEG_INFINITY,
                ],
            ),
        ] {
            let mut output = [0.0; 5];
            smooth_into(
                &input,
                &[5],
                &[0],
                &[vec![0.25, 0.5, 0.25]],
                &[BoundaryMode::Reflect],
                0.0,
                omit_nan,
                ignore_inf,
                min_count,
                &mut output,
            )
            .unwrap();
            for (value, expected) in output.into_iter().zip(expected) {
                if expected.is_nan() {
                    assert!(value.is_nan());
                } else {
                    assert_eq!(value.to_bits(), expected.to_bits());
                }
            }
        }
    }
}
