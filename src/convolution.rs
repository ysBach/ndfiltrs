use rayon::prelude::*;
use reducers::Float;

use crate::boundary::{map_index, BoundaryMode};
use crate::error::KernelError;
use crate::mean::Sum;

#[derive(Clone, Copy, Debug)]
pub(crate) enum WeightPolicy {
    Sum,
    Normalize {
        omit_nan: bool,
        ignore_inf: bool,
        min_count: usize,
    },
}

/// Validated geometry with zero-weight taps removed before sampling.
pub(crate) struct ConvolutionConfig<T> {
    shape: Vec<usize>,
    strides: Vec<usize>,
    modes: Vec<BoundaryMode>,
    cval: T,
    weights: Vec<f64>,
    offsets: Vec<i128>,
    flat_offsets: Vec<isize>,
    low: Vec<i128>,
    high: Vec<i128>,
    policy: WeightPolicy,
    output_len: usize,
}

fn checked_product(shape: &[usize]) -> Result<usize, KernelError> {
    shape.iter().try_fold(1usize, |product, &length| {
        product.checked_mul(length).ok_or(KernelError::Overflow)
    })
}

fn strides(shape: &[usize]) -> Result<Vec<usize>, KernelError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(shape.len())
        .map_err(|_| KernelError::Memory)?;
    result.resize(shape.len(), 1usize);
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        result[axis] = result[axis + 1]
            .checked_mul(shape[axis + 1])
            .ok_or(KernelError::Overflow)?;
    }
    Ok(result)
}

impl<T: Copy> ConvolutionConfig<T> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        shape: Vec<usize>,
        kernel_shape: Vec<usize>,
        weights: Vec<f64>,
        modes: Vec<BoundaryMode>,
        cval: T,
        origin: Vec<isize>,
        correlate: bool,
        policy: WeightPolicy,
    ) -> Result<Self, KernelError> {
        let rank = shape.len();
        if rank == 0 {
            return Err(KernelError::InvalidConfiguration(
                "convolution requires at least one dimension",
            ));
        }
        if kernel_shape.len() != rank || modes.len() != rank || origin.len() != rank {
            return Err(KernelError::InvalidConfiguration(
                "array, weights, mode, and origin must have equal rank",
            ));
        }
        if kernel_shape.contains(&0) {
            return Err(KernelError::InvalidConfiguration(
                "weights must have nonempty dimensions",
            ));
        }
        if checked_product(&kernel_shape)? != weights.len() {
            return Err(KernelError::LengthMismatch);
        }
        if weights.iter().any(|weight| !weight.is_finite()) {
            return Err(KernelError::InvalidConfiguration(
                "weights must contain only finite values",
            ));
        }
        let count = weights.iter().filter(|&&weight| weight != 0.0).count();
        if let WeightPolicy::Normalize {
            omit_nan,
            ignore_inf,
            min_count,
        } = policy
        {
            if !omit_nan && (ignore_inf || min_count != 1) {
                return Err(KernelError::InvalidConfiguration(
                    "ignore_inf and min_count require NaN omission",
                ));
            }
            if weights.iter().any(|&weight| weight < 0.0) || count == 0 {
                return Err(KernelError::InvalidConfiguration(
                    "normalized weights must be nonnegative with a positive entry",
                ));
            }
            if min_count == 0 || min_count > count {
                return Err(KernelError::InvalidConfiguration(
                    "min_count must be between one and the number of positive weights",
                ));
            }
        }
        let mut anchors = Vec::new();
        anchors
            .try_reserve_exact(rank)
            .map_err(|_| KernelError::Memory)?;
        for (&length, &shift) in kernel_shape.iter().zip(&origin) {
            let length = i128::try_from(length).map_err(|_| KernelError::Overflow)?;
            let anchor = length / 2 + shift as i128;
            if !(0..length).contains(&anchor) {
                return Err(KernelError::InvalidConfiguration(
                    "origin must keep the kernel center inside its dimensions",
                ));
            }
            anchors.push(anchor);
        }
        let kernel_strides = strides(&kernel_shape)?;
        let mut active_weights = Vec::new();
        active_weights
            .try_reserve_exact(count)
            .map_err(|_| KernelError::Memory)?;
        let mut offsets = Vec::new();
        offsets
            .try_reserve_exact(count.checked_mul(rank).ok_or(KernelError::Overflow)?)
            .map_err(|_| KernelError::Memory)?;
        for (index, weight) in weights.into_iter().enumerate() {
            if weight == 0.0 {
                continue;
            }
            active_weights.push(weight);
            let mut remainder = index;
            for axis in 0..rank {
                let coordinate = remainder / kernel_strides[axis];
                remainder %= kernel_strides[axis];
                let offset = coordinate as i128 - anchors[axis];
                offsets.push(if correlate { offset } else { -offset });
            }
        }
        let array_strides = strides(&shape)?;
        let mut flat_offsets = Vec::new();
        flat_offsets
            .try_reserve_exact(count)
            .map_err(|_| KernelError::Memory)?;
        let mut low = vec![0; rank];
        let mut high = vec![0; rank];
        for tap in offsets.chunks_exact(rank) {
            let mut flat = 0i128;
            for axis in 0..rank {
                low[axis] = low[axis].min(tap[axis]);
                high[axis] = high[axis].max(tap[axis]);
                flat = tap[axis]
                    .checked_mul(array_strides[axis] as i128)
                    .and_then(|step| flat.checked_add(step))
                    .ok_or(KernelError::Overflow)?;
            }
            // Huge supports can have no interior but still valid mapped edges.
            // An unrepresentable displacement disables only the interior path.
            if let Ok(offset) = isize::try_from(flat) {
                flat_offsets.push(offset);
            }
        }
        if flat_offsets.len() != count {
            flat_offsets.clear();
        }
        Ok(Self {
            output_len: checked_product(&shape)?,
            strides: array_strides,
            shape,
            modes,
            cval,
            weights: active_weights,
            offsets,
            flat_offsets,
            low,
            high,
            policy,
        })
    }
}

/// Split finite nonzero values into a signed mantissa in [1, 2) and exponent.
fn split(value: f64) -> (f64, i32) {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    if exponent == 0 {
        // The subnormal fraction is an exact integer with at most 52 bits.
        let leading = 63 - fraction.leading_zeros();
        let mantissa = fraction as f64 / (1u64 << leading) as f64;
        (mantissa.copysign(value), leading as i32 - 1074)
    } else {
        (
            f64::from_bits((bits & ((1u64 << 63) | ((1u64 << 52) - 1))) | (1023u64 << 52)),
            exponent - 1023,
        )
    }
}

/// Apply a binary scale without forming an overflowing or zero scale factor.
fn scale_binary(value: f64, exponent: i32) -> f64 {
    if value == 0.0 {
        return value;
    }
    let (mantissa, value_exponent) = split(value);
    let target = value_exponent + exponent;
    if target > 1023 {
        return f64::INFINITY.copysign(value);
    }
    if target < -1075 {
        return 0.0_f64.copysign(value);
    }
    if target >= -1022 {
        mantissa * f64::from_bits(((target + 1023) as u64) << 52)
    } else {
        // A normal intermediate keeps the mantissa intact until final rounding.
        let remaining = f64::from_bits(((target + 1022 + 1023) as u64) << 52);
        (mantissa * f64::MIN_POSITIVE) * remaining
    }
}

/// Compensated scaled dot product, returned before restoring its binary scale.
fn scaled_dot(pairs: &[(f64, f64)]) -> (f64, i32) {
    let maximum = pairs
        .iter()
        .filter(|(value, _)| *value != 0.0)
        .map(|&(value, weight)| split(value).1 + split(weight).1)
        .max();
    let Some(maximum) = maximum else {
        return (0.0, 0);
    };
    // Mantissa products are below 4. These count-derived bits bound the sum
    // of absolute scaled products below 1, including cancellation-heavy sums.
    let count_bits = usize::BITS - (pairs.len() - 1).leading_zeros();
    let exponent = maximum + count_bits as i32 + 2;
    let mut sum = Sum::default();
    for &(value, weight) in pairs {
        if value == 0.0 {
            continue;
        }
        let (value_mantissa, value_exponent) = split(value);
        let (weight_mantissa, weight_exponent) = split(weight);
        sum.add(scale_binary(
            value_mantissa * weight_mantissa,
            value_exponent + weight_exponent - exponent,
        ));
    }
    (sum.total(), exponent)
}

fn scaled_weight_sum(pairs: &[(f64, f64)]) -> (f64, i32) {
    let maximum = pairs
        .iter()
        .map(|&(_, weight)| split(weight).1)
        .max()
        .unwrap();
    let count_bits = usize::BITS - (pairs.len() - 1).leading_zeros();
    let exponent = maximum + count_bits as i32 + 1;
    let mut sum = Sum::default();
    for &(_, weight) in pairs {
        sum.add(scale_binary(weight, -exponent));
    }
    (sum.total(), exponent)
}

/// Accumulate finite products in tap order; request replay on exceptional arithmetic.
pub(crate) struct WeightedSum {
    sum: Sum,
    weight_sum: Sum,
    normalize: bool,
    fallback: bool,
    empty: bool,
    lower: f64,
    upper: f64,
}

impl WeightedSum {
    pub(crate) fn new(normalize: bool) -> Self {
        Self {
            sum: Sum::default(),
            weight_sum: Sum::default(),
            normalize,
            fallback: false,
            empty: true,
            lower: f64::INFINITY,
            upper: f64::NEG_INFINITY,
        }
    }

    pub(crate) fn add(&mut self, value: f64, weight: f64) {
        self.empty = false;
        if self.normalize {
            // Separable passes need the full retained mass even after fallback.
            self.weight_sum.add(weight);
        }
        if self.fallback {
            return;
        }
        if self.normalize {
            self.lower = self.lower.min(value);
            self.upper = self.upper.max(value);
        }
        let product = value * weight;
        // Recover multiplication overflow or underflow before it corrupts the
        // compensated sum. No data-size or workload-dependent cutoff is used.
        if !product.is_finite() || (value != 0.0 && product.abs() < f64::MIN_POSITIVE) {
            self.fallback = true;
            return;
        }
        self.sum.add(product);
    }

    pub(crate) fn weight_total(&self) -> f64 {
        self.weight_sum.total()
    }

    /// Return None when the caller must replay samples with binary scaling.
    pub(crate) fn finish(&self) -> Option<f64> {
        if self.empty {
            return Some(0.0);
        }
        let total = self.sum.total();
        if self.fallback || !total.is_finite() {
            return None;
        }
        if self.normalize {
            let denominator = self.weight_total();
            let result = total / denominator;
            if !denominator.is_finite() || !result.is_finite() {
                return None;
            }
            Some(result.clamp(self.lower, self.upper))
        } else {
            Some(total)
        }
    }
}

/// Sum finite products, or normalize by positive retained weights.
///
/// Binary scaling recovers overflowing products and intermediate sums. Terms
/// below the scaled subnormal range can underflow; exact rounding is not promised.
pub(crate) fn finite_weighted(pairs: &[(f64, f64)], normalize: bool) -> f64 {
    let mut accumulator = WeightedSum::new(normalize);
    for &(value, weight) in pairs {
        accumulator.add(value, weight);
    }
    if let Some(result) = accumulator.finish() {
        return result;
    }
    let (total, exponent) = scaled_dot(pairs);
    if normalize {
        let (denominator, denominator_exponent) = scaled_weight_sum(pairs);
        // The exact positive-weight average lies within the retained range.
        // Recompute bounds because the fast pass may have stopped early.
        let (lower, upper) = pairs.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(lower, upper), &(value, _)| (lower.min(value), upper.max(value)),
        );
        scale_binary(total / denominator, exponent - denominator_exponent).clamp(lower, upper)
    } else {
        scale_binary(total, exponent)
    }
}

/// Per-thread buffers reused across output evaluations.
struct Scratch {
    coordinates: Vec<usize>,
    pairs: Vec<(f64, f64)>,
}

impl Scratch {
    fn new<T>(config: &ConvolutionConfig<T>) -> Result<Self, KernelError> {
        let rank = config.shape.len();
        let mut coordinates = Vec::new();
        coordinates
            .try_reserve_exact(rank)
            .map_err(|_| KernelError::Memory)?;
        coordinates.resize(rank, 0usize);
        Ok(Self {
            coordinates,
            pairs: Vec::new(),
        })
    }
}

/// Evaluate one output at C-order flat index `flat`.
///
/// Dense and masked-position evaluation both call this function, so they
/// visit the same taps in the same order and share all arithmetic.
fn convolve_at<T: Float, const BUFFERED: bool>(
    input: &[T],
    config: &ConvolutionConfig<T>,
    flat: usize,
    scratch: &mut Scratch,
) -> Result<T, KernelError> {
    let rank = config.shape.len();
    let Scratch { coordinates, pairs } = scratch;
    let mut remainder = flat;
    let mut interior = config.flat_offsets.len() == config.weights.len();
    for (axis, coordinate) in coordinates.iter_mut().enumerate() {
        *coordinate = remainder / config.strides[axis];
        remainder %= config.strides[axis];
        let position = *coordinate as i128;
        interior &= position + config.low[axis] >= 0
            && position + config.high[axis] < config.shape[axis] as i128;
    }
    pairs.clear();
    if BUFFERED {
        pairs
            .try_reserve(config.weights.len())
            .map_err(|_| KernelError::Memory)?;
    }
    let normalize = matches!(config.policy, WeightPolicy::Normalize { .. });
    let mut accumulator = WeightedSum::new(normalize);
    let mut retained = 0usize;
    let mut nan = false;
    let mut positive_inf = false;
    let mut negative_inf = false;
    for (tap, (&weight, offsets)) in config
        .weights
        .iter()
        .zip(config.offsets.chunks_exact(rank))
        .enumerate()
    {
        let value = if interior {
            // Each axis is in bounds, so this signed displacement stays in input.
            input[flat.wrapping_add_signed(config.flat_offsets[tap])]
        } else {
            let mut source = 0usize;
            let mut constant = false;
            for axis in 0..rank {
                let coordinate = (coordinates[axis] as i128)
                    .checked_add(offsets[axis])
                    .ok_or(KernelError::Overflow)?;
                match map_index(coordinate, config.shape[axis], config.modes[axis])? {
                    Some(mapped) => {
                        source = source
                            .checked_add(
                                mapped
                                    .checked_mul(config.strides[axis])
                                    .ok_or(KernelError::Overflow)?,
                            )
                            .ok_or(KernelError::Overflow)?;
                    }
                    None => {
                        constant = true;
                        break;
                    }
                }
            }
            if constant {
                config.cval
            } else {
                input[source]
            }
        }
        .to_f64();
        if value.is_nan() {
            if matches!(
                config.policy,
                WeightPolicy::Sum
                    | WeightPolicy::Normalize {
                        omit_nan: false,
                        ..
                    }
            ) {
                nan = true;
                break;
            }
            continue;
        }
        if value.is_infinite()
            && matches!(
                config.policy,
                WeightPolicy::Normalize {
                    ignore_inf: true,
                    ..
                }
            )
        {
            continue;
        }
        retained += 1;
        if value.is_infinite() {
            if value.is_sign_negative() == weight.is_sign_negative() {
                positive_inf = true;
            } else {
                negative_inf = true;
            }
        } else if BUFFERED {
            pairs.push((value, weight));
        } else {
            accumulator.add(value, weight);
        }
    }
    if let WeightPolicy::Normalize { min_count, .. } = config.policy {
        nan |= retained < min_count;
    }
    Ok(if nan || (positive_inf && negative_inf) {
        T::nan()
    } else if positive_inf {
        T::from_f64(f64::INFINITY)
    } else if negative_inf {
        T::from_f64(f64::NEG_INFINITY)
    } else {
        let value = if BUFFERED {
            finite_weighted(pairs, normalize)
        } else if let Some(value) = accumulator.finish() {
            value
        } else {
            // Replay only exceptional arithmetic, keeping identical tap order.
            return convolve_at::<T, true>(input, config, flat, scratch);
        };
        T::from_f64(value)
    })
}

fn convolve_range<T: Float>(
    input: &[T],
    config: &ConvolutionConfig<T>,
    start: usize,
    output: &mut [T],
) -> Result<(), KernelError> {
    let mut scratch = Scratch::new(config)?;
    for (local, destination) in output.iter_mut().enumerate() {
        let flat = start.checked_add(local).ok_or(KernelError::Overflow)?;
        *destination = convolve_at::<T, false>(input, config, flat, &mut scratch)?;
    }
    Ok(())
}

/// Replace masked samples by the convolution of the unmasked samples.
///
/// `positions` are strictly increasing C-order flat indices. Every masked
/// sample is set to NaN before any output is evaluated, and all estimates are
/// computed before any is written back, so masked values never feed each
/// other. With a NaN-omitting normalized policy, each repaired value equals
/// the dense `convolve_into` output of the data with masked samples set to
/// NaN, bit for bit; only the masked positions are evaluated.
pub(crate) fn repair_masked<T: Float>(
    data: &mut [T],
    config: &ConvolutionConfig<T>,
    positions: &[usize],
) -> Result<(), KernelError> {
    if data.len() != config.output_len
        || positions.last().is_some_and(|&last| last >= data.len())
        || positions.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(KernelError::LengthMismatch);
    }
    if positions.is_empty() {
        return Ok(());
    }
    let mut estimates = Vec::new();
    estimates
        .try_reserve_exact(positions.len())
        .map_err(|_| KernelError::Memory)?;
    estimates.resize(positions.len(), T::nan());
    for &offset in positions {
        data[offset] = T::nan();
    }
    let snapshot: &[T] = data;
    // Each estimate depends only on its position, so the partition cannot
    // change results; it only divides the work among the pool's threads.
    let chunk_len = positions
        .len()
        .div_ceil(rayon::current_num_threads().max(1));
    estimates
        .par_chunks_mut(chunk_len)
        .zip(positions.par_chunks(chunk_len))
        .try_for_each(|(chunk, offsets)| {
            let mut scratch = Scratch::new(config)?;
            for (destination, &offset) in chunk.iter_mut().zip(offsets) {
                *destination = convolve_at::<T, false>(snapshot, config, offset, &mut scratch)?;
            }
            Ok(())
        })?;
    for (&offset, value) in positions.iter().zip(estimates) {
        data[offset] = value;
    }
    Ok(())
}

pub(crate) fn convolve_into<T: Float>(
    input: &[T],
    config: &ConvolutionConfig<T>,
    output: &mut [T],
) -> Result<(), KernelError> {
    if input.len() != config.output_len || output.len() != config.output_len {
        return Err(KernelError::LengthMismatch);
    }
    if output.is_empty() {
        return Ok(());
    }
    let threads = rayon::current_num_threads().max(1);
    let chunk_len = output.len().div_ceil(threads);
    output
        .par_chunks_mut(chunk_len)
        .enumerate()
        .try_for_each(|(chunk_index, chunk)| {
            let start = chunk_index
                .checked_mul(chunk_len)
                .ok_or(KernelError::Overflow)?;
            convolve_range(input, config, start, chunk)
        })
}

#[cfg(test)]
mod tests {
    use super::{
        convolve_into, finite_weighted, repair_masked, scale_binary, ConvolutionConfig,
        WeightPolicy,
    };
    use crate::boundary::BoundaryMode;
    use reducers::Float;

    fn run(input: &[f64], weights: &[f64], origin: isize, correlate: bool) -> Vec<f64> {
        let config = ConvolutionConfig::new(
            vec![input.len()],
            vec![weights.len()],
            weights.to_vec(),
            vec![BoundaryMode::Constant],
            0.0,
            vec![origin],
            correlate,
            WeightPolicy::Sum,
        )
        .unwrap();
        let mut output = vec![0.0; input.len()];
        convolve_into(input, &config, &mut output).unwrap();
        output
    }

    #[test]
    fn even_kernel_convolution_and_correlation_have_explicit_anchors() {
        assert_eq!(
            run(&[1.0, 2.0, 3.0], &[1.0, 10.0], 0, false),
            [12.0, 23.0, 30.0]
        );
        assert_eq!(
            run(&[1.0, 2.0, 3.0], &[1.0, 10.0], 0, true),
            [10.0, 21.0, 32.0]
        );
        assert_eq!(
            run(&[1.0, 2.0, 3.0], &[1.0, 10.0], -1, false),
            [1.0, 12.0, 23.0]
        );
    }

    #[test]
    fn zero_taps_do_not_sample_nonfinite_values() {
        let result = run(&[f64::NAN, 2.0, f64::INFINITY], &[0.0, 1.0, 0.0], 0, false);
        assert_eq!(result[0].to_bits(), f64::NAN.to_bits());
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], f64::INFINITY);
        assert!(run(&[f64::NAN, f64::INFINITY], &[0.0], 0, false)
            .iter()
            .all(|value| value.to_bits() == 0.0_f64.to_bits()));
    }

    #[test]
    fn signed_weights_determine_infinity_signs() {
        assert_eq!(
            run(&[f64::INFINITY], &[-1.0], 0, false),
            [f64::NEG_INFINITY]
        );
        assert_eq!(
            run(&[f64::INFINITY, f64::INFINITY], &[-1.0, 1.0], 0, true)[1].to_bits(),
            f64::NAN.to_bits()
        );
    }

    #[test]
    fn scaling_recovers_overflowing_products_and_sums() {
        assert_eq!(
            finite_weighted(&[(f64::MAX, 2.0), (-f64::MAX, 2.0)], false),
            0.0
        );
        assert_eq!(
            finite_weighted(&[(f64::MAX, 1.0), (f64::MAX, 1.0), (-f64::MAX, 1.0)], false),
            f64::MAX
        );
        assert_eq!(finite_weighted(&[(f64::MAX, 2.0)], false), f64::INFINITY);
        let smallest = f64::from_bits(1);
        assert_eq!(
            finite_weighted(&[(smallest, 0.5), (smallest, 0.5)], false).to_bits(),
            1
        );
    }

    #[test]
    fn weighted_mean_handles_extreme_weights_and_constant_values() {
        for weight in [f64::MAX, f64::MIN_POSITIVE, f64::from_bits(1)] {
            assert_eq!(finite_weighted(&[(f64::MAX, weight); 3], true), f64::MAX);
            assert_eq!(
                finite_weighted(&[(f64::from_bits(1), weight); 3], true).to_bits(),
                1
            );
            assert_eq!(finite_weighted(&[(2.0, weight), (4.0, weight)], true), 3.0);
        }
        assert_eq!(
            finite_weighted(&[(1e16, 1.0), (1.0, 1.0), (-1e16, 1.0)], false),
            1.0
        );
    }

    #[test]
    fn normalized_policy_counts_only_retained_positive_taps() {
        let config = ConvolutionConfig::new(
            vec![3],
            vec![3],
            vec![1.0, 0.0, 3.0],
            vec![BoundaryMode::Constant],
            f64::NAN,
            vec![0],
            true,
            WeightPolicy::Normalize {
                omit_nan: true,
                ignore_inf: true,
                min_count: 1,
            },
        )
        .unwrap();
        let mut output = vec![0.0; 3];
        convolve_into(&[2.0, f64::INFINITY, 4.0], &config, &mut output).unwrap();
        assert!(output[0].is_nan());
        assert_eq!(output[1], 3.5);
        assert!(output[2].is_nan());
    }

    #[test]
    fn plain_normalization_preserves_finite_maxima_and_propagates_nan() {
        let config = ConvolutionConfig::new(
            vec![3],
            vec![3],
            vec![0.25, 0.5000000000000001, 0.25],
            vec![BoundaryMode::Reflect],
            0.0,
            vec![0],
            false,
            WeightPolicy::Normalize {
                omit_nan: false,
                ignore_inf: false,
                min_count: 1,
            },
        )
        .unwrap();
        let mut output = vec![0.0; 3];
        convolve_into(&[f64::MAX; 3], &config, &mut output).unwrap();
        assert_eq!(output, [f64::MAX; 3]);
        convolve_into(&[2.0, f64::NAN, 4.0], &config, &mut output).unwrap();
        assert!(output
            .iter()
            .all(|value| value.to_bits() == f64::NAN.to_bits()));
    }

    #[test]
    fn mixed_axis_boundaries_and_thread_counts_preserve_results() {
        let config = ConvolutionConfig::new(
            vec![2, 3],
            vec![2, 2],
            vec![1.0, -2.0, 3.0, 0.0],
            vec![BoundaryMode::Wrap, BoundaryMode::Constant],
            5.0,
            vec![-1, 0],
            true,
            WeightPolicy::Sum,
        )
        .unwrap();
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut results = Vec::new();
        for threads in [1, 3] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut output = vec![0.0; input.len()];
            pool.install(|| convolve_into(&input, &config, &mut output))
                .unwrap();
            results.push(output);
        }
        // Correlation offsets are row [0, 1], column [-1, 0].
        assert_eq!(results[0], [18.0, 9.0, 11.0, 12.0, -3.0, -1.0]);
        assert_eq!(results[0], results[1]);
    }

    #[test]
    fn empty_arrays_return_empty_and_lengths_are_checked() {
        let config = ConvolutionConfig::new(
            vec![0, 2],
            vec![1, 1],
            vec![1.0],
            vec![BoundaryMode::Reflect; 2],
            0.0,
            vec![0, 0],
            false,
            WeightPolicy::Sum,
        )
        .unwrap();
        convolve_into::<f64>(&[], &config, &mut []).unwrap();
        assert!(convolve_into(&[1.0], &config, &mut []).is_err());
    }

    /// Compare masked-position repair with dense evaluation of NaN-masked data.
    fn check_repair_matches_dense<T: Float + std::fmt::Debug>(to_bits: fn(T) -> u64) {
        let shape = vec![7, 9];
        let len = 63;
        // Deterministic signed values with NaN, infinities, and signed zero.
        let mut input: Vec<T> = (0..len)
            .map(|index| T::from_f64(((index * 37 % 23) as f64 - 11.0) / 3.0))
            .collect();
        input[4] = T::nan();
        input[10] = T::from_f64(f64::INFINITY);
        input[30] = T::from_f64(f64::NEG_INFINITY);
        input[40] = T::from_f64(-0.0);
        // Adjacent pixels, a full row, corners, and an unmasked NaN neighbor.
        let mut positions = vec![0, 1, 5, 11, 12, 20, 62];
        positions.extend(27..36);
        positions.sort_unstable();
        let weights = vec![0.5, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0, 0.25, 0.0, 1.0, 1.5, 0.5];
        for (modes, cval, ignore_inf, min_count) in [
            (
                vec![BoundaryMode::Wrap, BoundaryMode::Constant],
                f64::NAN,
                false,
                1,
            ),
            (
                vec![BoundaryMode::Reflect, BoundaryMode::Mirror],
                0.0,
                true,
                3,
            ),
            (
                vec![BoundaryMode::Constant, BoundaryMode::Nearest],
                2.5,
                false,
                4,
            ),
        ] {
            let config = ConvolutionConfig::new(
                shape.clone(),
                vec![3, 4],
                weights.clone(),
                modes,
                T::from_f64(cval),
                vec![-1, 1],
                true,
                WeightPolicy::Normalize {
                    omit_nan: true,
                    ignore_inf,
                    min_count,
                },
            )
            .unwrap();
            let mut masked = input.clone();
            for &offset in &positions {
                masked[offset] = T::nan();
            }
            let mut dense = vec![T::from_f64(0.0); len];
            convolve_into(&masked, &config, &mut dense).unwrap();
            for threads in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let mut repaired = input.clone();
                pool.install(|| repair_masked(&mut repaired, &config, &positions))
                    .unwrap();
                for index in 0..len {
                    let expected = if positions.contains(&index) {
                        dense[index]
                    } else {
                        input[index]
                    };
                    assert_eq!(to_bits(repaired[index]), to_bits(expected), "{index}");
                }
            }
        }
    }

    #[test]
    fn masked_repair_matches_dense_evaluation_bitwise() {
        check_repair_matches_dense::<f64>(f64::to_bits);
        check_repair_matches_dense::<f32>(|value| u64::from(value.to_bits()));
    }

    #[test]
    fn masked_repair_rejects_unsorted_or_out_of_range_positions() {
        let config = ConvolutionConfig::new(
            vec![3],
            vec![3],
            vec![1.0; 3],
            vec![BoundaryMode::Reflect],
            f64::NAN,
            vec![0],
            true,
            WeightPolicy::Normalize {
                omit_nan: true,
                ignore_inf: false,
                min_count: 1,
            },
        )
        .unwrap();
        for positions in [&[1, 1][..], &[2, 1], &[3]] {
            assert!(repair_masked(&mut [1.0, 2.0, 3.0], &config, positions).is_err());
        }
        assert!(repair_masked(&mut [1.0, 2.0], &config, &[0]).is_err());
        let mut data = [1.0, 99.0, 99.0];
        repair_masked(&mut data, &config, &[1, 2]).unwrap();
        // Masked values never feed each other: index 1 keeps only 1.0, and
        // index 2 sees masked indices 1 and 2 (3 reflects to 2), so it is NaN.
        assert_eq!(data[..2], [1.0, 1.0]);
        assert!(data[2].is_nan());
    }

    #[test]
    fn binary_scaling_preserves_subnormal_and_overflow_boundaries() {
        assert_eq!(scale_binary(1.0, -1074).to_bits(), 1);
        assert_eq!(scale_binary(1.0, -1075).to_bits(), 0);
        assert_eq!(scale_binary(1.5, -1075).to_bits(), 1);
        assert_eq!(scale_binary(f64::from_bits(1), 1074), 1.0);
        assert_eq!(scale_binary(f64::MAX, 0), f64::MAX);
        assert_eq!(scale_binary(-1.0, 1024), f64::NEG_INFINITY);
    }

    #[test]
    fn malformed_configuration_is_rejected_before_traversal() {
        for (weights, origin, policy) in [
            (vec![1.0, 1.0], vec![1], WeightPolicy::Sum),
            (vec![f64::NAN, 1.0], vec![0], WeightPolicy::Sum),
            (
                vec![-1.0, 1.0],
                vec![0],
                WeightPolicy::Normalize {
                    omit_nan: true,
                    ignore_inf: false,
                    min_count: 1,
                },
            ),
            (
                vec![0.0, 0.0],
                vec![0],
                WeightPolicy::Normalize {
                    omit_nan: true,
                    ignore_inf: false,
                    min_count: 1,
                },
            ),
            (
                vec![0.0, 1.0],
                vec![0],
                WeightPolicy::Normalize {
                    omit_nan: true,
                    ignore_inf: false,
                    min_count: 2,
                },
            ),
        ] {
            assert!(ConvolutionConfig::new(
                vec![3],
                vec![2],
                weights,
                vec![BoundaryMode::Reflect],
                0.0,
                origin,
                false,
                policy
            )
            .is_err());
        }
    }
}
