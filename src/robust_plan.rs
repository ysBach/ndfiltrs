//! Public robust window-filter API.

use crate::error::Error;
use crate::plan::{FilterFloat, Neighborhood};
use crate::robust::{robust_into, RobustConfig, RobustReduction};

/// A validated robust window filter for one array shape.
///
/// Statistics are computed in `f64` and cast once to `T`.
///
/// # Examples
///
/// ```
/// use ndfiltrs::{ClipConfig, ClipSpread, ClipStatistic, InvalidPolicy, Neighborhood,
///     RobustFilter, RobustReduction};
///
/// let skip_nan = InvalidPolicy::Omit { ignore_inf: true, min_count: 1 };
/// let hood = Neighborhood::rectangular([5]);
///
/// // 16th percentile of each 5-sample window: rank 0.16 * (5 - 1) = 0.64.
/// let q16 = RobustReduction::Percentile { q: 16.0, policy: skip_nan };
/// let low = RobustFilter::new(&[6], &hood, q16, 0.0)?.apply(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
/// assert!((low[2] - 1.64).abs() < 1e-12);
///
/// // 3-sigma clipped mean with a MAD spread. With 5 samples, one outlier is
/// // at most 1.79 population standard deviations from the mean, so a `Std`
/// // spread could never reject it.
/// let clip = ClipConfig::new(3.0).spread(ClipSpread::Mad).statistic(ClipStatistic::Mean);
/// let clipped = RobustFilter::new(&[6], &hood, RobustReduction::SigmaClip(clip), 0.0)?
///     .apply(&[1.0_f64, 1.0, 100.0, 1.0, 1.0, 1.0])?;
/// assert_eq!(clipped[2], 1.0);
/// # Ok::<(), ndfiltrs::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct RobustFilter<T> {
    config: RobustConfig<T>,
    len: usize,
}

impl<T: FilterFloat> RobustFilter<T> {
    /// Validate the geometry and statistic for inputs of shape `shape`.
    ///
    /// `cval` is the value outside the array for
    /// [`BoundaryMode::Constant`](crate::BoundaryMode::Constant).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfiguration`] for invalid geometry, a
    /// parameter outside its documented range, or a `min_count` outside
    /// `1..=window size`. Returns [`Error::LengthMismatch`] when a footprint
    /// mask length differs from the product of its shape.
    ///
    /// [`Error::InvalidConfiguration`]: crate::Error::InvalidConfiguration
    /// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
    pub fn new(
        shape: &[usize],
        neighborhood: &Neighborhood,
        reduction: RobustReduction,
        cval: T,
    ) -> Result<Self, Error> {
        let window = neighborhood.window(shape)?;
        let len = window.output_len();
        Ok(Self {
            config: RobustConfig::from_window(window, cval, reduction)?,
            len,
        })
    }

    /// Number of elements in each input and output buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` when the plan's shape has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Filter `input` into a new buffer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] for a wrong input length.
    ///
    /// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
    pub fn apply(&self, input: &[T]) -> Result<Vec<T>, Error> {
        let mut output = Vec::new();
        output
            .try_reserve_exact(input.len())
            .map_err(|_| Error::Memory)?;
        output.resize(input.len(), T::nan());
        self.apply_into(input, &mut output)?;
        Ok(output)
    }

    /// Filter `input` into `output`. The buffers must not overlap.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] for a wrong buffer length.
    ///
    /// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
    pub fn apply_into(&self, input: &[T], output: &mut [T]) -> Result<(), Error> {
        robust_into(input, &self.config, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduction::InvalidPolicy;
    use crate::robust::ClipConfig;

    #[test]
    fn invalid_parameters_are_errors() {
        let hood = Neighborhood::rectangular([3]);
        let policy = InvalidPolicy::Propagate;
        for reduction in [
            RobustReduction::Percentile { q: 101.0, policy },
            RobustReduction::Percentile {
                q: f64::NAN,
                policy,
            },
            RobustReduction::Mad {
                scale: -1.0,
                policy,
            },
            RobustReduction::BiweightLocation { c: 0.0, policy },
            RobustReduction::SigmaClip(ClipConfig::new(-1.0)),
            RobustReduction::SigmaClip(ClipConfig::new(3.0).min_count(4)),
        ] {
            assert!(
                matches!(
                    RobustFilter::<f64>::new(&[5], &hood, reduction, 0.0),
                    Err(Error::InvalidConfiguration(_))
                ),
                "{reduction:?}"
            );
        }
    }
}

#[cfg(test)]
mod edge_tests {
    use super::*;
    use crate::reduction::InvalidPolicy;
    use crate::robust::{ClipConfig, ClipStatistic};

    #[test]
    fn zero_length_axis_is_a_no_op() {
        let plan = RobustFilter::<f64>::new(
            &[0, 4],
            &Neighborhood::rectangular([3, 3]),
            RobustReduction::Percentile {
                q: 16.0,
                policy: InvalidPolicy::Propagate,
            },
            0.0,
        )
        .unwrap();
        assert!(plan.is_empty());
        assert_eq!(plan.apply(&[]).unwrap(), Vec::<f64>::new());
    }

    #[test]
    fn sigma_clip_drops_nan_and_infinity() {
        let clip = ClipConfig::new(3.0).statistic(ClipStatistic::Median);
        let plan = RobustFilter::<f64>::new(
            &[5],
            &Neighborhood::rectangular([5]),
            RobustReduction::SigmaClip(clip),
            0.0,
        )
        .unwrap();
        let out = plan
            .apply(&[2.0, f64::NAN, 4.0, f64::INFINITY, f64::NEG_INFINITY])
            .unwrap();
        // The centered window keeps only 2 and 4 after dropping nonfinite
        // samples, so the median is 3.
        assert_eq!(out[2], 3.0);
    }
}
