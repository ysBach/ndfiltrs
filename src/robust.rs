use rayon::prelude::*;
use reducers::Float;

#[cfg(test)]
use crate::boundary::BoundaryMode;
use crate::error::KernelError;
use crate::location::{biweight_location, biweight_scale, mode_estimate, ModeEstimator};
use crate::mean::Sum;
use crate::window::Window;

// Phi^-1(0.75): the median absolute deviation of a standard normal variable.
const NORMAL_UPPER_QUARTILE: f64 = 0.674_489_750_196_081_7;

#[derive(Clone, Copy, Debug)]
pub(crate) enum SamplePolicy {
    Propagate,
    Omit { ignore_inf: bool, min_count: usize },
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Center {
    Mean,
    Median,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Spread {
    Std,
    Mad,
}

/// Statistic reported from the samples that survive clipping.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Statistic {
    Mean,
    Median,
    Mode(ModeEstimator, f64),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ClipConfig {
    pub(crate) lower: f64,
    pub(crate) upper: f64,
    pub(crate) maxiters: Option<usize>,
    pub(crate) center: Center,
    pub(crate) spread: Spread,
    pub(crate) statistic: Statistic,
    pub(crate) min_count: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RobustReduction {
    Percentile {
        q: f64,
        policy: SamplePolicy,
    },
    Mad {
        scale: f64,
        policy: SamplePolicy,
    },
    SigmaClip(ClipConfig),
    Mode {
        estimator: ModeEstimator,
        pearson: f64,
        policy: SamplePolicy,
    },
    BiweightLocation {
        c: f64,
        policy: SamplePolicy,
    },
    BiweightScale {
        c: f64,
        modify_sample_size: bool,
        policy: SamplePolicy,
    },
}

impl RobustReduction {
    fn validate(self, volume: usize) -> Result<(), KernelError> {
        let min_count = match self {
            Self::Percentile { q, policy } => {
                if !q.is_finite() || !(0.0..=100.0).contains(&q) {
                    return Err(KernelError::InvalidConfiguration(
                        "percentile must be finite and between zero and 100",
                    ));
                }
                policy_min_count(policy)
            }
            Self::Mad { scale, policy } => {
                if !scale.is_finite() || scale < 0.0 {
                    return Err(KernelError::InvalidConfiguration(
                        "MAD scale must be finite and nonnegative",
                    ));
                }
                policy_min_count(policy)
            }
            Self::Mode {
                pearson, policy, ..
            } => {
                validate_pearson(pearson)?;
                policy_min_count(policy)
            }
            Self::BiweightLocation { c, policy } | Self::BiweightScale { c, policy, .. } => {
                if !c.is_finite() || c <= 0.0 {
                    return Err(KernelError::InvalidConfiguration(
                        "biweight tuning constant must be finite and positive",
                    ));
                }
                policy_min_count(policy)
            }
            Self::SigmaClip(config) => {
                if !config.lower.is_finite()
                    || !config.upper.is_finite()
                    || config.lower < 0.0
                    || config.upper < 0.0
                {
                    return Err(KernelError::InvalidConfiguration(
                        "clipping limits must be finite and nonnegative",
                    ));
                }
                if let Statistic::Mode(_, pearson) = config.statistic {
                    validate_pearson(pearson)?;
                }
                config.min_count
            }
        };
        if min_count == 0 || min_count > volume {
            return Err(KernelError::InvalidConfiguration(
                "min_count must be between one and the number of window elements",
            ));
        }
        Ok(())
    }

    fn finish(self, values: &mut Vec<f64>, scratch: &mut Vec<f64>) -> f64 {
        let policy = match self {
            Self::Percentile { policy, .. }
            | Self::Mad { policy, .. }
            | Self::Mode { policy, .. }
            | Self::BiweightLocation { policy, .. }
            | Self::BiweightScale { policy, .. } => policy,
            Self::SigmaClip(config) => {
                values.retain(|value| value.is_finite());
                return clipped(values, scratch, config);
            }
        };
        match policy {
            SamplePolicy::Propagate if values.iter().any(|value| value.is_nan()) => {
                return f64::NAN;
            }
            SamplePolicy::Omit {
                ignore_inf,
                min_count,
            } => {
                values.retain(|value| !value.is_nan() && (!ignore_inf || value.is_finite()));
                if values.len() < min_count {
                    return f64::NAN;
                }
            }
            _ => {}
        }
        if values.is_empty() {
            return f64::NAN;
        }
        match self {
            Self::Percentile { q, .. } => percentile(values, q),
            Self::Mad { scale, .. } => {
                let deviation = mad_parts(values, scratch);
                multiply_scaled(deviation, scale)
            }
            // These estimators are defined for finite samples only; a retained
            // infinity makes the window undefined, as for variance.
            Self::Mode { .. } | Self::BiweightLocation { .. } | Self::BiweightScale { .. }
                if values.iter().any(|value| value.is_infinite()) =>
            {
                f64::NAN
            }
            Self::Mode {
                estimator, pearson, ..
            } => mode_estimate(values, estimator, pearson),
            Self::BiweightLocation { c, .. } => biweight_location(values, scratch, c),
            Self::BiweightScale {
                c,
                modify_sample_size,
                ..
            } => biweight_scale(values, scratch, c, modify_sample_size),
            Self::SigmaClip(_) => unreachable!(),
        }
    }
}

/// SExtractor accepts `BACK_PEARSON` from 1 upward; 1 makes the mode the median.
fn validate_pearson(pearson: f64) -> Result<(), KernelError> {
    if pearson.is_finite() && pearson >= 1.0 {
        Ok(())
    } else {
        Err(KernelError::InvalidConfiguration(
            "pearson must be finite and at least one",
        ))
    }
}

fn policy_min_count(policy: SamplePolicy) -> usize {
    match policy {
        SamplePolicy::Propagate => 1,
        SamplePolicy::Omit { min_count, .. } => min_count,
    }
}

/// Interpolate two ordered samples without overflowing finite differences.
fn interpolate(lower: f64, upper: f64, fraction: f64) -> f64 {
    if fraction == 0.0 || lower.to_bits() == upper.to_bits() {
        return lower;
    }
    if fraction == 1.0 {
        return upper;
    }
    if lower.is_infinite() || upper.is_infinite() {
        return if lower == f64::NEG_INFINITY && upper == f64::INFINITY {
            f64::NAN
        } else if lower.is_infinite() {
            lower
        } else {
            upper
        };
    }
    if lower.is_sign_negative() != upper.is_sign_negative() {
        if !(upper - lower).is_finite() {
            // Halving makes the difference representable without rounding
            // both weighted endpoints separately near f64::MAX.
            let half_lower = lower / 2.0;
            let half_upper = upper / 2.0;
            return ((half_lower + (half_upper - half_lower) * fraction) * 2.0).clamp(lower, upper);
        }
        crate::convolution::finite_weighted(&[(lower, 1.0 - fraction), (upper, fraction)], false)
    } else {
        // With equal signs the difference is finite. Interpolate from the
        // nearer endpoint to reduce cancellation near fractions zero and one.
        let difference = upper - lower;
        let result = if fraction <= 0.5 {
            lower + difference * fraction
        } else {
            upper - difference * (1.0 - fraction)
        };
        result.clamp(lower, upper)
    }
}

fn percentile(values: &mut [f64], q: f64) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    if q == 100.0 {
        return values[values.len() - 1];
    }
    let rank = (q / 100.0) * (values.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = (lower + 1).min(values.len() - 1);
    interpolate(values[lower], values[upper], rank - lower as f64)
}

/// A nonnegative magnitude whose final power of two has not been applied.
#[derive(Clone, Copy)]
struct Scaled {
    value: f64,
    exponent: i32,
}

fn mad_parts(values: &[f64], scratch: &mut Vec<f64>) -> Scaled {
    scratch.clear();
    scratch.extend_from_slice(values);
    scratch.sort_unstable_by(f64::total_cmp);
    let lower = scratch[(scratch.len() - 1) / 2];
    let upper = scratch[scratch.len() / 2];
    let center = if !lower.is_finite() || !upper.is_finite() {
        Scaled {
            value: interpolate(lower, upper, 0.5),
            exponent: 0,
        }
    } else if (lower + upper).is_finite() {
        // Keeping the division by two separate preserves half a subnormal.
        Scaled {
            value: lower + upper,
            exponent: -1,
        }
    } else {
        Scaled {
            value: lower / 2.0 + upper / 2.0,
            exponent: 0,
        }
    };
    if !center.value.is_finite() {
        return Scaled {
            value: f64::NAN,
            exponent: 0,
        };
    }
    // Absolute deviations increase away from the median on both sides of the
    // sorted samples. Merge those two orders without rounding their scales or
    // allocating another per-window buffer.
    let mut left = scratch.len() / 2;
    let mut right = left;
    let mut previous = Scaled {
        value: 0.0,
        exponent: 0,
    };
    let mut selected = previous;
    for _ in 0..=scratch.len() / 2 {
        previous = selected;
        let take_left = left > 0
            && (right == scratch.len()
                || compare_scaled(
                    absolute_deviation(scratch[left - 1], center),
                    absolute_deviation(scratch[right], center),
                )
                .is_le());
        selected = if take_left {
            left -= 1;
            absolute_deviation(scratch[left], center)
        } else {
            let result = absolute_deviation(scratch[right], center);
            right += 1;
            result
        };
    }
    if scratch.len() % 2 == 1 {
        selected
    } else {
        scaled_midpoint(previous, selected)
    }
}

fn absolute_deviation(value: f64, center: Scaled) -> Scaled {
    if center.exponent == -1 && (value * 2.0).is_finite() {
        let difference = value * 2.0 - center.value;
        if difference.is_finite() {
            return Scaled {
                value: difference.abs(),
                exponent: -1,
            };
        }
    }
    // This fallback is used only outside the range where doubling is safe.
    // A subnormal center cannot change a deviation on that scale.
    let midpoint = if center.exponent == -1 {
        center.value / 2.0
    } else {
        center.value
    };
    let difference = value - midpoint;
    if difference.is_finite() || value.is_infinite() {
        Scaled {
            value: difference.abs(),
            exponent: 0,
        }
    } else {
        Scaled {
            value: (value / 2.0 - midpoint / 2.0).abs(),
            exponent: 1,
        }
    }
}

fn compare_scaled(left: Scaled, right: Scaled) -> std::cmp::Ordering {
    if left.value == 0.0
        || right.value == 0.0
        || left.value.is_infinite()
        || right.value.is_infinite()
    {
        return f64::total_cmp(&left.value, &right.value);
    }
    let (left_value, left_exp) = binary_parts(left.value);
    let (right_value, right_exp) = binary_parts(right.value);
    (left_exp + left.exponent)
        .cmp(&(right_exp + right.exponent))
        .then_with(|| f64::total_cmp(&left_value, &right_value))
}

fn scaled_midpoint(left: Scaled, right: Scaled) -> Scaled {
    if right.value.is_infinite() {
        return right;
    }
    if right.value == 0.0 {
        return right;
    }
    let (right_value, right_exp) = binary_parts(right.value);
    let exponent = right_exp + right.exponent;
    let left_value = if left.value == 0.0 {
        0.0
    } else {
        let (value, left_exp) = binary_parts(left.value);
        value * 2.0_f64.powi(left_exp + left.exponent - exponent)
    };
    Scaled {
        value: left_value + right_value,
        exponent: exponent - 1,
    }
}

/// Split a positive finite number into a mantissa in [1, 2) and binary exponent.
fn binary_parts(value: f64) -> (f64, i32) {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    if exponent == 0 {
        let leading = 63 - fraction.leading_zeros();
        (
            fraction as f64 / (1u64 << leading) as f64,
            leading as i32 - 1074,
        )
    } else {
        (f64::from_bits(fraction | (1023u64 << 52)), exponent - 1023)
    }
}

fn multiply_scaled(value: Scaled, multiplier: f64) -> f64 {
    if !value.value.is_finite() {
        return value.value * multiplier;
    }
    if value.value == 0.0 || multiplier == 0.0 {
        return 0.0;
    }
    let (left, left_exp) = binary_parts(value.value);
    let (right, right_exp) = binary_parts(multiplier);
    let (mantissa, adjustment) = binary_parts(left * right);
    let exponent = left_exp + right_exp + adjustment + value.exponent;
    if exponent > 1023 {
        return f64::INFINITY;
    }
    if exponent < -1075 {
        return 0.0;
    }
    if exponent >= -1022 {
        mantissa * f64::from_bits(((exponent + 1023) as u64) << 52)
    } else {
        let scale = f64::from_bits(((exponent + 2045) as u64) << 52);
        (mantissa * f64::MIN_POSITIVE) * scale
    }
}

/// Compare a finite deviation with a scaled clipping limit without overflow.
fn within_limit(value: f64, center: f64, spread: Scaled, sigma: f64) -> bool {
    if value == center {
        return true;
    }
    if spread.value == 0.0 || sigma == 0.0 {
        return false;
    }
    let difference = value - center;
    let (distance, distance_exp) = if difference.is_finite() {
        binary_parts(difference.abs())
    } else {
        let (mantissa, exponent) = binary_parts((value / 2.0 - center / 2.0).abs());
        (mantissa, exponent + 1)
    };
    let (scale, scale_exp) = binary_parts(spread.value);
    let (multiplier, multiplier_exp) = binary_parts(sigma);
    let (limit, adjustment) = binary_parts(scale * multiplier);
    let limit_exp = scale_exp + multiplier_exp + adjustment + spread.exponent;
    distance_exp < limit_exp || (distance_exp == limit_exp && distance <= limit)
}

fn location(values: &[f64], scratch: &mut Vec<f64>, method: Center) -> f64 {
    match method {
        Center::Mean => finite_mean(values),
        Center::Median => {
            scratch.clear();
            scratch.extend_from_slice(values);
            percentile(scratch, 50.0)
        }
    }
}

/// Average finite samples, retrying on overflow with a count-derived scale.
fn finite_mean(values: &[f64]) -> f64 {
    let mut sum = Sum::default();
    for &value in values {
        sum.add(value);
    }
    let total = sum.total();
    if total.is_finite() {
        return total / values.len() as f64;
    }
    let exponent = usize::BITS - (values.len() - 1).leading_zeros() + 1;
    let scale = 2.0_f64.powi(exponent as i32);
    let mut sum = Sum::default();
    let mut lower = f64::INFINITY;
    let mut upper = f64::NEG_INFINITY;
    for &value in values {
        sum.add(value / scale);
        lower = lower.min(value);
        upper = upper.max(value);
    }
    // A finite arithmetic mean lies in the sample range. This bound removes
    // only rounding excursions when the final division approaches MAX.
    (sum.total() / (values.len() as f64 / scale)).clamp(lower, upper)
}

/// Population scatter of finite samples, keeping an overflowed scale separate.
fn std_parts(values: &[f64]) -> Scaled {
    let reference = values[0];
    let halve = values.iter().any(|&value| !(value - reference).is_finite());
    let delta = |value: f64| {
        if halve {
            value / 2.0 - reference / 2.0
        } else {
            value - reference
        }
    };
    let scale = values
        .iter()
        .map(|&value| delta(value).abs())
        .fold(0.0, f64::max);
    if scale == 0.0 {
        return Scaled {
            value: 0.0,
            exponent: 0,
        };
    }
    let mut sum = Sum::default();
    for &value in values {
        sum.add(delta(value) / scale);
    }
    let center = sum.total() / values.len() as f64;
    let mut squares = Sum::default();
    for &value in values {
        let residual = delta(value) / scale - center;
        squares.add(residual * residual);
    }
    let (mantissa, exponent) = binary_parts(scale);
    Scaled {
        value: mantissa * (squares.total() / values.len() as f64).sqrt(),
        exponent: exponent + i32::from(halve),
    }
}

fn clipped(values: &mut Vec<f64>, scratch: &mut Vec<f64>, config: ClipConfig) -> f64 {
    let mut iteration = 0usize;
    while values.len() >= config.min_count && config.maxiters.is_none_or(|limit| iteration < limit)
    {
        let center = location(values, scratch, config.center);
        let spread = match config.spread {
            Spread::Mad => {
                let deviation = mad_parts(values, scratch);
                if deviation.value > 0.0 && deviation.value.is_finite() {
                    // Keep the Gaussian consistency factor in the mantissa:
                    // the corrected spread can exceed MAX while a subsequent
                    // sigma multiplier still gives a representable limit.
                    let (mantissa, exponent) = binary_parts(deviation.value);
                    Scaled {
                        value: mantissa / NORMAL_UPPER_QUARTILE,
                        exponent: exponent + deviation.exponent,
                    }
                } else {
                    deviation
                }
            }
            Spread::Std => std_parts(values),
        };
        let previous_count = values.len();
        values.retain(|&value| {
            let sigma = if value < center {
                config.lower
            } else {
                config.upper
            };
            within_limit(value, center, spread, sigma)
        });
        iteration += 1;
        if values.len() == previous_count {
            break;
        }
    }
    if values.len() < config.min_count {
        return f64::NAN;
    }
    match config.statistic {
        Statistic::Mean => location(values, scratch, Center::Mean),
        Statistic::Median => location(values, scratch, Center::Median),
        Statistic::Mode(estimator, pearson) => mode_estimate(values, estimator, pearson),
    }
}

pub(crate) struct RobustConfig<T> {
    window: Window,
    cval: T,
    reduction: RobustReduction,
}

impl<T: Copy> RobustConfig<T> {
    /// Configure a centered rectangular robust filter.
    #[cfg(test)]
    pub(crate) fn new(
        shape: Vec<usize>,
        size: Vec<usize>,
        modes: Vec<BoundaryMode>,
        cval: T,
        reduction: RobustReduction,
    ) -> Result<Self, KernelError> {
        Self::from_window(Window::rectangular(shape, &size, modes)?, cval, reduction)
    }

    pub(crate) fn from_window(
        window: Window,
        cval: T,
        reduction: RobustReduction,
    ) -> Result<Self, KernelError> {
        reduction.validate(window.count())?;
        Ok(Self {
            window,
            cval,
            reduction,
        })
    }
}

fn filter_range<T: Float>(
    input: &[T],
    config: &RobustConfig<T>,
    start: usize,
    output: &mut [T],
) -> Result<(), KernelError> {
    let mut index = Vec::new();
    index
        .try_reserve_exact(config.window.rank())
        .map_err(|_| KernelError::Memory)?;
    index.resize(config.window.rank(), 0usize);
    let mut values = Vec::new();
    values
        .try_reserve_exact(config.window.count())
        .map_err(|_| KernelError::Memory)?;
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(config.window.count())
        .map_err(|_| KernelError::Memory)?;
    for (position, destination) in output.iter_mut().enumerate() {
        let center = start.checked_add(position).ok_or(KernelError::Overflow)?;
        values.clear();
        config.window.gather(
            input,
            center,
            config.cval,
            &mut index,
            &mut values,
            |value| value.to_f64(),
        )?;
        let value = config.reduction.finish(&mut values, &mut scratch);
        *destination = if value.is_nan() {
            T::nan()
        } else {
            T::from_f64(value)
        };
    }
    Ok(())
}

pub(crate) fn robust_into<T: Float>(
    input: &[T],
    config: &RobustConfig<T>,
    output: &mut [T],
) -> Result<(), KernelError> {
    if input.len() != config.window.output_len() || output.len() != config.window.output_len() {
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
        .try_for_each(|(index, chunk)| {
            let start = index.checked_mul(chunk_len).ok_or(KernelError::Overflow)?;
            filter_range(input, config, start, chunk)
        })
}

#[cfg(test)]
mod tests {
    use super::SamplePolicy;
    use super::{
        interpolate, robust_into, Center, ClipConfig, RobustConfig, RobustReduction, Spread,
        Statistic,
    };
    use crate::boundary::BoundaryMode;
    use crate::location::ModeEstimator;

    fn evaluate(reduction: RobustReduction, values: &[f64]) -> f64 {
        reduction.finish(&mut values.to_vec(), &mut Vec::new())
    }

    fn clipping(maxiters: Option<usize>) -> RobustReduction {
        RobustReduction::SigmaClip(ClipConfig {
            lower: 1.0,
            upper: 1.0,
            maxiters,
            center: Center::Median,
            spread: Spread::Std,
            statistic: Statistic::Mean,
            min_count: 1,
        })
    }

    #[test]
    fn mode_and_biweight_follow_sample_policies() {
        let omit = SamplePolicy::Omit {
            ignore_inf: false,
            min_count: 1,
        };
        let mode = |estimator| RobustReduction::Mode {
            estimator,
            pearson: if estimator == ModeEstimator::Sex {
                2.5
            } else {
                3.0
            },
            policy: omit,
        };
        // mean 4, median 2.5, population std sqrt(12.5): skew 0.42 >= 0.3.
        let skewed = [1.0, 2.0, f64::NAN, 3.0, 10.0];
        assert_eq!(evaluate(mode(ModeEstimator::Sex), &skewed), 2.5);
        assert_eq!(evaluate(mode(ModeEstimator::Mmm), &skewed), -0.5);
        assert_eq!(evaluate(mode(ModeEstimator::Iraf), &skewed), -0.5);
        assert!(evaluate(
            RobustReduction::Mode {
                estimator: ModeEstimator::Mmm,
                pearson: 3.0,
                policy: SamplePolicy::Propagate,
            },
            &skewed,
        )
        .is_nan());
        assert!(evaluate(mode(ModeEstimator::Mmm), &[1.0, f64::INFINITY]).is_nan());
        let symmetric = [1.0, 2.0, 3.0, f64::NAN];
        assert_eq!(
            evaluate(
                RobustReduction::BiweightLocation {
                    c: 6.0,
                    policy: omit
                },
                &symmetric
            ),
            2.0
        );
        assert!(evaluate(
            RobustReduction::BiweightScale {
                c: 9.0,
                modify_sample_size: false,
                policy: omit,
            },
            &[1.0, 2.0, f64::NEG_INFINITY],
        )
        .is_nan());
        let clipped_mode = RobustReduction::SigmaClip(ClipConfig {
            lower: 10.0,
            upper: 10.0,
            maxiters: Some(1),
            center: Center::Median,
            spread: Spread::Std,
            statistic: Statistic::Mode(ModeEstimator::Mmm, 3.0),
            min_count: 1,
        });
        assert_eq!(evaluate(clipped_mode, &skewed), -0.5);
    }

    #[test]
    fn percentile_interpolation_preserves_extremes_infinities_and_endpoints() {
        assert_eq!(interpolate(-f64::MAX, f64::MAX, 0.5), 0.0);
        assert_eq!(interpolate(f64::MAX, f64::MAX, 0.75), f64::MAX);
        assert_eq!(interpolate(2.0, f64::INFINITY, 0.0), 2.0);
        assert_eq!(interpolate(2.0, f64::INFINITY, 0.5), f64::INFINITY);
        assert_eq!(interpolate(f64::NEG_INFINITY, 2.0, 1.0), 2.0);
        assert!(interpolate(f64::NEG_INFINITY, f64::INFINITY, 0.5).is_nan());
        for (q, expected) in [(0.0, 1.0), (25.0, 2.5), (50.0, 4.0), (100.0, 9.0)] {
            assert_eq!(
                evaluate(
                    RobustReduction::Percentile {
                        q,
                        policy: SamplePolicy::Propagate,
                    },
                    &[9.0, 1.0, 4.0],
                ),
                expected
            );
        }
        assert_eq!(interpolate(-0.0, 0.0, 0.5).to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn omission_counts_after_dropping_invalid_samples() {
        let statistic = RobustReduction::Percentile {
            q: 50.0,
            policy: SamplePolicy::Omit {
                ignore_inf: true,
                min_count: 2,
            },
        };
        assert_eq!(
            evaluate(statistic, &[1.0, f64::NAN, 3.0, f64::INFINITY]),
            2.0
        );
        assert!(evaluate(statistic, &[1.0, f64::NAN, f64::INFINITY]).is_nan());
    }

    #[test]
    fn mad_is_centered_on_the_median_and_scaled_after_safe_deviations() {
        let statistic = RobustReduction::Mad {
            scale: 2.0,
            policy: SamplePolicy::Propagate,
        };
        assert_eq!(evaluate(statistic, &[1.0, 2.0, 100.0]), 2.0);
        assert_eq!(evaluate(statistic, &[1.0, 2.0, f64::INFINITY]), 2.0);
        assert!(evaluate(statistic, &[1.0, f64::INFINITY, f64::INFINITY]).is_nan());
        let large = evaluate(
            RobustReduction::Mad {
                scale: 0.25,
                policy: SamplePolicy::Propagate,
            },
            &[-f64::MAX, 0.5 * f64::MAX, f64::MAX],
        );
        assert_eq!(large, f64::MAX * 0.125);
        assert_eq!(
            evaluate(
                RobustReduction::Mad {
                    scale: f64::MAX,
                    policy: SamplePolicy::Propagate,
                },
                &[-f64::from_bits(1), 0.0, f64::from_bits(1)],
            ),
            f64::from_bits(1) * f64::MAX,
        );
    }

    #[test]
    fn clipping_iterations_and_nonfinite_omission_have_literal_results() {
        let values = [0.0, 0.0, 1.0, 2.0, 10.0, f64::NAN, f64::INFINITY];
        assert_eq!(evaluate(clipping(Some(0)), &values), 2.6);
        assert_eq!(evaluate(clipping(Some(1)), &values), 0.75);
        assert_eq!(evaluate(clipping(None), &values), 0.0);
    }

    #[test]
    fn clipping_zero_mad_and_insufficient_count_are_defined() {
        let config = ClipConfig {
            lower: 3.0,
            upper: 3.0,
            maxiters: None,
            center: Center::Median,
            spread: Spread::Mad,
            statistic: Statistic::Median,
            min_count: 1,
        };
        assert_eq!(
            evaluate(RobustReduction::SigmaClip(config), &[0.0, 0.0, 9.0]),
            0.0
        );
        assert!(evaluate(
            RobustReduction::SigmaClip(ClipConfig {
                min_count: 3,
                ..config
            }),
            &[0.0, 0.0, 9.0],
        )
        .is_nan());
    }

    #[test]
    fn clipping_mad_uses_gaussian_consistency_without_overflowing_the_spread() {
        let config = ClipConfig {
            lower: 2.0,
            upper: 2.0,
            maxiters: Some(1),
            center: Center::Median,
            spread: Spread::Mad,
            statistic: Statistic::Mean,
            min_count: 1,
        };
        // Raw MAD is one. Gaussian normalization retains the sample at 4.5.
        assert_eq!(
            evaluate(
                RobustReduction::SigmaClip(config),
                &[0.0, 1.0, 2.0, 3.0, 4.5]
            ),
            2.1,
        );
        // Corrected MAD exceeds MAX, but half that spread is below MAX.
        assert_eq!(
            evaluate(
                RobustReduction::SigmaClip(ClipConfig {
                    lower: 0.5,
                    upper: 0.5,
                    ..config
                }),
                &[-f64::MAX, 0.0, f64::MAX],
            ),
            0.0,
        );
    }

    #[test]
    fn invalid_robust_outputs_use_canonical_nan() {
        for reduction in [
            RobustReduction::Percentile {
                q: 50.0,
                policy: SamplePolicy::Propagate,
            },
            RobustReduction::Mad {
                scale: 0.0,
                policy: SamplePolicy::Propagate,
            },
            clipping(None),
        ] {
            let config = RobustConfig::new(
                vec![1],
                vec![1],
                vec![BoundaryMode::Reflect],
                0.0,
                reduction,
            )
            .unwrap();
            let mut output = [0.0];
            robust_into(
                &[f64::from_bits(0xfff8_0000_0000_0001)],
                &config,
                &mut output,
            )
            .unwrap();
            assert_eq!(output[0].to_bits(), f64::NAN.to_bits());
        }
    }

    #[test]
    fn percentile_midpoints_keep_subnormal_cancellation_until_final_rounding() {
        let tiny = f64::from_bits(1);
        let statistic = RobustReduction::Percentile {
            q: 50.0,
            policy: SamplePolicy::Omit {
                ignore_inf: false,
                min_count: 1,
            },
        };
        assert_eq!(evaluate(statistic, &[-tiny, 2.0 * tiny, f64::NAN]), 0.0);
    }

    #[test]
    fn mad_retains_fractional_subnormal_centers_and_deviations() {
        let tiny = f64::from_bits(1);
        let statistic = RobustReduction::Mad {
            scale: 2.0,
            policy: SamplePolicy::Omit {
                ignore_inf: false,
                min_count: 1,
            },
        };
        assert_eq!(evaluate(statistic, &[0.0, tiny, f64::NAN]), tiny);
        assert_eq!(
            evaluate(statistic, &[0.0, tiny, 2.0 * tiny, 4.0 * tiny]),
            2.0 * tiny,
        );
        // Keeping a tiny center must not overflow a distant finite sample.
        assert_eq!(
            evaluate(statistic, &[-f64::MAX, -tiny, 0.0, f64::MAX]),
            f64::MAX,
        );
    }

    #[test]
    fn clipping_std_keeps_its_scale_until_the_sigma_multiplier() {
        let tiny = f64::from_bits(1);
        let statistic = RobustReduction::SigmaClip(ClipConfig {
            lower: 3.0,
            upper: 3.0,
            maxiters: None,
            center: Center::Median,
            spread: Spread::Std,
            statistic: Statistic::Mean,
            min_count: 3,
        });
        assert_eq!(evaluate(statistic, &[0.0, tiny, tiny]), tiny);
    }

    #[test]
    fn clipping_does_not_confuse_overflowed_distances_with_infinite_limits() {
        let config = ClipConfig {
            lower: 1.5,
            upper: 1.5,
            maxiters: Some(1),
            center: Center::Median,
            spread: Spread::Std,
            statistic: Statistic::Mean,
            min_count: 1,
        };
        assert_eq!(
            evaluate(
                RobustReduction::SigmaClip(config),
                &[-f64::MAX, -f64::MAX, f64::MAX]
            ),
            -f64::MAX
        );
    }

    #[test]
    fn geometry_uses_boundaries_and_parallel_results_are_deterministic() {
        let config = RobustConfig::new(
            vec![3],
            vec![3],
            vec![BoundaryMode::Reflect],
            0.0,
            RobustReduction::Percentile {
                q: 25.0,
                policy: SamplePolicy::Propagate,
            },
        )
        .unwrap();
        let mut expected = [0.0; 3];
        robust_into(&[1.0, 5.0, 9.0], &config, &mut expected).unwrap();
        assert_eq!(expected, [1.0, 3.0, 7.0]);
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut actual = [0.0; 3];
            pool.install(|| robust_into(&[1.0, 5.0, 9.0], &config, &mut actual))
                .unwrap();
            assert_eq!(actual.map(f64::to_bits), expected.map(f64::to_bits));
        }
    }

    #[test]
    fn validation_precedes_empty_output_and_length_is_checked() {
        let make = |q| {
            RobustConfig::new(
                vec![0],
                vec![3],
                vec![BoundaryMode::Reflect],
                0.0,
                RobustReduction::Percentile {
                    q,
                    policy: SamplePolicy::Propagate,
                },
            )
        };
        assert!(make(f64::NAN).is_err());
        assert!(make(101.0).is_err());
        let config = make(50.0).unwrap();
        robust_into::<f64>(&[], &config, &mut []).unwrap();
        assert!(robust_into(&[1.0], &config, &mut []).is_err());
    }
}
