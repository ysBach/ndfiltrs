use reducers::Float;

use crate::reduction::InvalidPolicy;

/// Neumaier accumulation in input order. Nonfinite terms are handled separately.
#[derive(Default)]
pub(crate) struct Sum {
    value: f64,
    correction: f64,
}

impl Sum {
    pub(crate) fn add(&mut self, value: f64) {
        let next = self.value + value;
        self.correction += if self.value.abs() >= value.abs() {
            (self.value - next) + value
        } else {
            (value - next) + self.value
        };
        self.value = next;
    }

    pub(crate) fn total(&self) -> f64 {
        self.value + self.correction
    }
}

/// Accumulate a window in tap order without storing its samples.
#[derive(Default)]
pub(crate) struct Mean {
    sum: Sum,
    count: usize,
    nan: bool,
    positive_inf: bool,
    negative_inf: bool,
}

impl Mean {
    pub(crate) fn add<T: Float>(&mut self, value: T, policy: InvalidPolicy) {
        if value.is_nan() {
            self.nan |= matches!(policy, InvalidPolicy::Propagate);
            return;
        }
        if matches!(
            policy,
            InvalidPolicy::Omit {
                ignore_inf: true,
                ..
            }
        ) && !value.is_finite()
        {
            return;
        }
        self.count += 1;
        let value = value.to_f64();
        if value.is_infinite() {
            self.positive_inf |= value.is_sign_positive();
            self.negative_inf |= value.is_sign_negative();
        } else {
            self.sum.add(value);
        }
    }

    /// Return `None` only when finite summation overflow needs scaled reevaluation.
    pub(crate) fn finish<T: Float>(&self, policy: InvalidPolicy) -> Option<T> {
        if self.nan
            || self.count == 0
            || matches!(policy, InvalidPolicy::Omit { min_count, .. } if self.count < min_count)
        {
            return Some(T::nan());
        }
        if self.positive_inf || self.negative_inf {
            return Some(match (self.positive_inf, self.negative_inf) {
                (true, true) => T::nan(),
                (true, false) => T::from_f64(f64::INFINITY),
                _ => T::from_f64(f64::NEG_INFINITY),
            });
        }
        let total = self.sum.total();
        total
            .is_finite()
            .then(|| T::from_f64(total / self.count as f64))
    }
}

/// Average a retained, non-NaN window, casting only the final result.
pub(crate) fn mean_valid<T: Float>(values: &[T]) -> T {
    let mut mean = Mean::default();
    for &value in values {
        mean.add(value, InvalidPolicy::Propagate);
    }
    if let Some(result) = mean.finish(InvalidPolicy::Propagate) {
        return result;
    }
    let count = values.len() as f64;
    // Retry only on arithmetic overflow. A power-of-two divisor at least
    // twice the count bounds the absolute exact sum by MAX/2, leaving
    // rounding headroom. Scaling is exact unless it enters subnormal range.
    let exponent = usize::BITS - (values.len() - 1).leading_zeros() + 1;
    let scale = 2.0_f64.powi(exponent as i32);
    let mut scaled = Sum::default();
    let mut lower = f64::INFINITY;
    let mut upper = f64::NEG_INFINITY;
    for &value in values {
        let value = value.to_f64();
        lower = lower.min(value);
        upper = upper.max(value);
        scaled.add(value / scale);
    }
    // A finite arithmetic mean lies in the sample range. Clamp only
    // roundoff that escapes those bounds (including final overflow).
    T::from_f64(((scaled.value + scaled.correction) / (count / scale)).clamp(lower, upper))
}

#[cfg(test)]
mod tests {
    use super::mean_valid;

    #[test]
    fn compensation_recovers_small_terms_after_cancellation() {
        assert_eq!(mean_valid(&[1e16, 1.0, -1e16]), 1.0 / 3.0);
        assert_eq!(mean_valid(&[1e20_f32, 3.0, -1e20]), 1.0);
    }

    #[test]
    fn finite_overflow_does_not_make_the_mean_infinite() {
        assert_eq!(mean_valid(&[f64::MAX; 3]), f64::MAX);
        assert_eq!(mean_valid(&[-f64::MAX; 4]), -f64::MAX);
        assert_eq!(mean_valid(&[f64::MAX, f64::MAX, -f64::MAX]), f64::MAX / 3.0);
        assert_eq!(mean_valid(&[f32::MAX; 9]), f32::MAX);
    }

    #[test]
    fn infinity_signs_are_independent_of_finite_accumulator_overflow() {
        assert_eq!(
            mean_valid(&[f64::MAX, f64::MAX, f64::NEG_INFINITY]),
            f64::NEG_INFINITY
        );
        assert_eq!(
            mean_valid(&[f64::INFINITY, f64::NEG_INFINITY]).to_bits(),
            f64::NAN.to_bits()
        );
    }

    #[test]
    fn subnormal_constants_are_not_scaled_unnecessarily() {
        assert_eq!(mean_valid(&[f64::from_bits(1); 3]).to_bits(), 1);
        assert_eq!(mean_valid(&[f32::from_bits(1); 3]).to_bits(), 1);
    }
}
