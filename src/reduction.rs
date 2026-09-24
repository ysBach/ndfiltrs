use reducers::Float;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InvalidPolicy {
    Propagate,
    Omit { ignore_inf: bool, min_count: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
// The benchmark-only build exercises the median path alone.
#[cfg_attr(
    all(feature = "benchmark", not(any(test, feature = "python"))),
    allow(dead_code)
)]
pub(crate) enum Reduction {
    Median(InvalidPolicy),
    NetworkMedian(InvalidPolicy),
    Mean(InvalidPolicy),
    // The rolling traversal is exposed only by the Python extension.
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    RollingMean(InvalidPolicy),
    Minimum(InvalidPolicy),
    Maximum(InvalidPolicy),
    Variance(InvalidPolicy, usize),
    Std(InvalidPolicy, usize),
    /// Zero-based rank in ascending total order (`-0.0` sorts before `+0.0`).
    Rank(InvalidPolicy, usize),
}

impl Reduction {
    pub(crate) fn policy(self) -> InvalidPolicy {
        match self {
            Self::Median(policy)
            | Self::NetworkMedian(policy)
            | Self::Mean(policy)
            | Self::RollingMean(policy)
            | Self::Minimum(policy)
            | Self::Maximum(policy)
            | Self::Variance(policy, _)
            | Self::Std(policy, _)
            | Self::Rank(policy, _) => policy,
        }
    }

    pub(crate) fn finish<T: Float>(self, values: &mut Vec<T>) -> T {
        match self.policy() {
            InvalidPolicy::Propagate => {
                if values.iter().any(|value| value.is_nan()) {
                    return T::nan();
                }
            }
            InvalidPolicy::Omit {
                ignore_inf,
                min_count,
            } => {
                let mut write = 0;
                for read in 0..values.len() {
                    let value = values[read];
                    if !value.is_nan() && (!ignore_inf || value.is_finite()) {
                        values[write] = value;
                        write += 1;
                    }
                }
                values.truncate(write);
                if write < min_count {
                    return T::nan();
                }
            }
        }
        match self {
            Self::Median(_) | Self::NetworkMedian(_) => {
                T::from_f64(reducers::reducers_1d::median_valid_in_place(values))
            }
            Self::Mean(_) | Self::RollingMean(_) => crate::mean::mean_valid(values),
            Self::Minimum(_) => values
                .iter()
                .copied()
                .min_by(|a, b| a.total_cmp(*b))
                .unwrap_or_else(T::nan),
            Self::Maximum(_) => values
                .iter()
                .copied()
                .max_by(|a, b| a.total_cmp(*b))
                .unwrap_or_else(T::nan),
            Self::Variance(_, ddof) => crate::dispersion::dispersion_valid(values, ddof, false),
            Self::Std(_, ddof) => crate::dispersion::dispersion_valid(values, ddof, true),
            Self::Rank(_, rank) => {
                if rank >= values.len() {
                    return T::nan();
                }
                *values
                    .select_nth_unstable_by(rank, |a, b| a.total_cmp(*b))
                    .1
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{InvalidPolicy, Reduction};

    #[test]
    fn extrema_have_total_signed_zero_order() {
        let mut values = vec![3.0_f64, -0.0, 0.0, 1.0];
        assert_eq!(
            Reduction::Rank(InvalidPolicy::Propagate, 1)
                .finish(&mut values)
                .to_bits(),
            0.0_f64.to_bits()
        );
        let mut values = vec![3.0_f64, 1.0];
        assert!(Reduction::Rank(InvalidPolicy::Propagate, 2)
            .finish(&mut values)
            .is_nan());
        let mut values = vec![0.0_f64, -0.0, 1.0];
        assert_eq!(
            Reduction::Minimum(InvalidPolicy::Propagate)
                .finish(&mut values)
                .to_bits(),
            (-0.0_f64).to_bits()
        );
        let mut values = vec![-0.0_f64, 0.0, -1.0];
        assert_eq!(
            Reduction::Maximum(InvalidPolicy::Propagate)
                .finish(&mut values)
                .to_bits(),
            0.0_f64.to_bits()
        );
    }

    #[test]
    fn dispersion_checks_count_after_omission() {
        let policy = InvalidPolicy::Omit {
            ignore_inf: true,
            min_count: 2,
        };
        let mut values = vec![1.0_f64, f64::NAN, f64::INFINITY, 3.0];
        assert_eq!(Reduction::Variance(policy, 0).finish(&mut values), 1.0);
        assert_eq!(
            Reduction::Std(policy, 1).finish(&mut values),
            2.0_f64.sqrt()
        );
        assert!(Reduction::Std(policy, 2).finish(&mut values).is_nan());
    }

    #[test]
    fn plain_policy_propagates_canonical_nan() {
        let mut values = vec![1.0_f64, f64::from_bits(0x7ff0_0000_0000_0001), 3.0];
        assert_eq!(
            Reduction::Median(InvalidPolicy::Propagate)
                .finish(&mut values)
                .to_bits(),
            f64::NAN.to_bits()
        );
    }

    #[test]
    fn omit_policy_compacts_values_and_enforces_min_count() {
        let mut values = vec![1.0_f64, f64::NAN, f64::INFINITY, 3.0];
        assert_eq!(
            Reduction::Median(InvalidPolicy::Omit {
                ignore_inf: true,
                min_count: 2,
            })
            .finish(&mut values),
            2.0
        );
        let mut insufficient = vec![1.0_f64, f64::NAN, f64::INFINITY];
        assert_eq!(
            Reduction::Median(InvalidPolicy::Omit {
                ignore_inf: true,
                min_count: 2,
            })
            .finish(&mut insufficient)
            .to_bits(),
            f64::NAN.to_bits()
        );
    }

    #[test]
    fn reducer_contract_preserves_total_order_and_safe_midpoints() {
        let mut signed_zeros = vec![0.0_f64, -0.0, -0.0];
        assert_eq!(
            Reduction::Median(InvalidPolicy::Propagate)
                .finish(&mut signed_zeros)
                .to_bits(),
            (-0.0_f64).to_bits()
        );

        let mut extremes = vec![f64::MAX, f64::from_bits(f64::MAX.to_bits() - 1)];
        assert!(Reduction::Median(InvalidPolicy::Propagate)
            .finish(&mut extremes)
            .is_finite());

        let mut mixed_inf = vec![f64::NEG_INFINITY, f64::INFINITY];
        assert_eq!(
            Reduction::Median(InvalidPolicy::Propagate)
                .finish(&mut mixed_inf)
                .to_bits(),
            f64::NAN.to_bits()
        );
    }

    #[test]
    fn f32_results_are_cast_once_after_f64_midpoint() {
        let upper = f32::from_bits(1.0_f32.to_bits() + 1);
        let mut values = vec![1.0_f32, upper];
        let expected = ((f64::from(1.0_f32) + f64::from(upper)) / 2.0) as f32;
        assert_eq!(
            Reduction::Median(InvalidPolicy::Propagate)
                .finish(&mut values)
                .to_bits(),
            expected.to_bits()
        );
    }
}
