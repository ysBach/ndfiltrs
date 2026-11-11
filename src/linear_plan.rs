//! Public convolution and separable-smoothing API.

use crate::boundary::BoundaryMode;
use crate::convolution::{convolve_into, ConvolutionConfig, WeightPolicy};
use crate::error::Error;
use crate::plan::{FilterFloat, Modes};
use crate::reduction::InvalidPolicy;
use crate::variance::{
    convolve_noise_into, Estimator, MatchedWeighting, NoiseMap, NoiseOutputs, NoiseSpec,
};

/// Convolution weights with origin, orientation, and boundary modes.
///
/// `weights` is in C order with shape `shape`. With
/// [`WeightKernel::correlate`] set, weight `k` multiplies the sample at
/// offset `k - center`, as `scipy.ndimage.correlate`. Otherwise the kernel is
/// flipped first, as `scipy.ndimage.convolve`. Zero weights are skipped.
/// The defaults are convolution, origin `0`, and [`BoundaryMode::Reflect`].
///
/// # Examples
///
/// ```
/// use ndfiltrs::{BoundaryMode, WeightKernel};
///
/// let smooth = WeightKernel::new([3, 3], [1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0])
///     .mode(BoundaryMode::Nearest);
/// # let _ = smooth;
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct WeightKernel {
    pub(crate) shape: Vec<usize>,
    pub(crate) weights: Vec<f64>,
    origin: Option<Vec<isize>>,
    modes: Modes,
    correlate: bool,
}

impl WeightKernel {
    /// Use finite `weights` in C order with shape `shape`.
    pub fn new(shape: impl Into<Vec<usize>>, weights: impl Into<Vec<f64>>) -> Self {
        Self {
            shape: shape.into(),
            weights: weights.into(),
            origin: None,
            modes: Modes::All(BoundaryMode::Reflect),
            correlate: false,
        }
    }

    /// Set the per-axis origin shift (one value per axis).
    pub fn origin(mut self, origin: impl Into<Vec<isize>>) -> Self {
        self.origin = Some(origin.into());
        self
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

    /// Correlate (`true`) instead of convolve (`false`, the default).
    pub fn correlate(mut self, correlate: bool) -> Self {
        self.correlate = correlate;
        self
    }

    pub(crate) fn config<T: Copy>(
        &self,
        shape: &[usize],
        policy: WeightPolicy,
        cval: T,
    ) -> Result<ConvolutionConfig<T>, Error> {
        let rank = self.shape.len();
        let modes = match &self.modes {
            Modes::All(mode) => vec![*mode; rank],
            Modes::PerAxis(modes) => modes.clone(),
        };
        ConvolutionConfig::new(
            shape.to_vec(),
            self.shape.clone(),
            self.weights.clone(),
            modes,
            cval,
            self.origin.clone().unwrap_or_else(|| vec![0; rank]),
            self.correlate,
            policy,
        )
    }
}

/// A validated convolution for one array shape.
///
/// # Examples
///
/// ```
/// use ndfiltrs::{BoundaryMode, Convolution, WeightKernel, WeightPolicy};
///
/// // NaN-skipping normalized box smoothing, like Astropy's `convolve`.
/// let skip_nan = WeightPolicy::Normalize { omit_nan: true, ignore_inf: false, min_count: 1 };
/// let plan = Convolution::new(
///     &[5],
///     &WeightKernel::new([3], [1.0, 1.0, 1.0]).mode(BoundaryMode::Nearest),
///     skip_nan,
///     0.0,
/// )?;
/// let out = plan.apply(&[1.0_f64, f64::NAN, 3.0, 4.0, 5.0])?;
/// assert_eq!(out, [1.0, 2.0, 3.5, 4.0, 14.0 / 3.0]);
/// # Ok::<(), ndfiltrs::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Convolution<T> {
    config: ConvolutionConfig<T>,
    len: usize,
}

impl<T: FilterFloat> Convolution<T> {
    /// Validate `kernel` and `policy` for inputs of shape `shape`.
    ///
    /// `cval` is the value outside the array for [`BoundaryMode::Constant`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfiguration`] for a rank mismatch, an empty
    /// kernel axis, a nonfinite weight, an origin outside the kernel, or a
    /// policy that the weights do not satisfy. Returns
    /// [`Error::LengthMismatch`] when the weight count differs from the
    /// product of the kernel shape.
    ///
    /// [`Error::InvalidConfiguration`]: crate::Error::InvalidConfiguration
    /// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
    pub fn new(
        shape: &[usize],
        kernel: &WeightKernel,
        policy: WeightPolicy,
        cval: T,
    ) -> Result<Self, Error> {
        let config = kernel.config(shape, policy, cval)?;
        let len = shape.iter().try_fold(1usize, |product, &value| {
            product.checked_mul(value).ok_or(Error::Overflow)
        })?;
        Ok(Self { config, len })
    }

    /// Number of elements in each input and output buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` when the plan's shape has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Convolve `input` into a new buffer.
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

    /// Convolve `input` into `output`. The buffers must not overlap.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] for a wrong buffer length.
    ///
    /// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
    pub fn apply_into(&self, input: &[T], output: &mut [T]) -> Result<(), Error> {
        convolve_into(input, &self.config, output)
    }

    /// Convolve `input` and propagate its independent per-pixel noise.
    ///
    /// Returns `(output, variance)`. See [`Convolution::apply_with_variance_into`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] for a wrong buffer length.
    ///
    /// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
    ///
    /// # Examples
    ///
    /// ```
    /// use ndfiltrs::{BoundaryMode, Convolution, NoiseMap, WeightKernel, WeightPolicy};
    ///
    /// let skip_nan = WeightPolicy::Normalize { omit_nan: true, ignore_inf: false, min_count: 1 };
    /// let plan = Convolution::new(
    ///     &[3],
    ///     &WeightKernel::new([3], [1.0, 1.0, 1.0]).mode(BoundaryMode::Constant),
    ///     skip_nan,
    ///     f64::NAN,
    /// )?;
    /// let (mean, variance) =
    ///     plan.apply_with_variance(&[1.0, 2.0, f64::NAN], &[1.0, 4.0, 1.0], NoiseMap::Variance)?;
    /// // Center: (1 + 2) / 2 with variance (1 + 4) / 2^2.
    /// assert_eq!((mean[1], variance[1]), (1.5, 1.25));
    /// # Ok::<(), ndfiltrs::Error>(())
    /// ```
    pub fn apply_with_variance(
        &self,
        input: &[T],
        noise: &[T],
        noise_map: NoiseMap,
    ) -> Result<(Vec<T>, Vec<T>), Error> {
        let mut output = filled(input.len())?;
        let mut variance = filled(input.len())?;
        self.apply_with_variance_into(input, noise, noise_map, &mut output, &mut variance)?;
        Ok((output, variance))
    }

    /// Convolve `input` into `output` and its propagated variance into
    /// `variance`, in one pass.
    ///
    /// `noise` holds one variance or inverse variance per input sample, as
    /// `noise_map` states. Noise is assumed independent between pixels. With
    /// coefficients `c_p`, the summed weights of all taps that read pixel `p`
    /// (boundary modes can read one pixel through several taps):
    ///
    /// - [`WeightPolicy::Sum`]: `y = sum_j w_j x_j` and
    ///   `var = sum_p c_p^2 v_p`.
    /// - [`WeightPolicy::Normalize`]: `y = sum_j w_j x_j / W` and
    ///   `var = sum_p c_p^2 v_p / W^2`, with `W = sum_j w_j` over retained taps.
    ///
    /// `output` is bitwise equal to [`Convolution::apply_into`] whenever the
    /// noise omits no sample. Constant-boundary samples are exact (zero
    /// variance). Invalid noise (NaN or negative) and infinite variance
    /// (inverse variance `0`) follow the policy. With NaN omission, such a
    /// sample is dropped from both outputs and from the `min_count` total.
    /// Otherwise the data result is unchanged, invalid noise gives a NaN
    /// variance, and infinite variance gives an infinite variance. A NaN
    /// output has a NaN variance. An infinite output keeps its finite
    /// variance.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] for a wrong buffer length.
    ///
    /// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
    pub fn apply_with_variance_into(
        &self,
        input: &[T],
        noise: &[T],
        noise_map: NoiseMap,
        output: &mut [T],
        variance: &mut [T],
    ) -> Result<(), Error> {
        let spec = match self.config.policy {
            WeightPolicy::Normalize {
                omit_nan,
                ignore_inf,
                min_count,
            } => NoiseSpec {
                noise: noise_map,
                inverse_weighting: false,
                estimator: Estimator::Mean,
                omit_nan,
                ignore_inf,
                min_count,
            },
            WeightPolicy::Sum => NoiseSpec {
                noise: noise_map,
                inverse_weighting: false,
                estimator: Estimator::Sum,
                omit_nan: false,
                ignore_inf: false,
                min_count: 1,
            },
        };
        convolve_noise_into(
            input,
            noise,
            &self.config,
            &spec,
            NoiseOutputs {
                value: output,
                variance,
                snr: None,
            },
        )
    }
}

fn filled<T: FilterFloat>(len: usize) -> Result<Vec<T>, Error> {
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(len).map_err(|_| Error::Memory)?;
    buffer.resize(len, T::nan());
    Ok(buffer)
}

/// Matched-filter maps returned by [`MatchedFilter::apply`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct MatchedMaps<T> {
    /// Signal-to-noise ratio `N / sqrt(Var(N))` of the filtered sum `N`.
    pub snr: Vec<T>,
    /// Template amplitude `N / S`.
    pub amplitude: Vec<T>,
    /// Variance of `amplitude`.
    pub variance: Vec<T>,
}

/// A validated matched filter with noise propagation for one array shape.
///
/// For a template `w` placed at each output, the filter forms
/// `N = sum_j a_j x_j` over retained samples, with `a_j = w_j` for
/// [`MatchedWeighting::Kernel`] or `a_j = w_j / v_j` for
/// [`MatchedWeighting::InverseVariance`]. The outputs are the amplitude
/// `A = N / S` with `S = sum_j w_j a_j`, its variance `Var(N) / S^2`, and the
/// signal-to-noise ratio `N / sqrt(Var(N))`. If the data are `A w` plus
/// independent zero-mean noise, `A` is unbiased. With inverse-variance
/// weighting, it is the minimum-variance linear unbiased estimate and its
/// variance is `1 / S` away from boundary repeats.
///
/// A matched filter correlates: set [`WeightKernel::correlate`] to keep the
/// template orientation. Template weights may have any sign.
///
/// Samples follow `policy`. NaN data are invalid. Noise that is NaN or
/// negative is invalid, and zero variance is also invalid under
/// inverse-variance weighting (its weight is infinite). Constant-boundary
/// samples have zero variance. An infinite variance (inverse variance `0`)
/// gives the sample no weight under inverse-variance weighting and is dropped
/// under [`InvalidPolicy::Omit`]. Under [`InvalidPolicy::Propagate`], an
/// output that depends on an invalid sample is NaN.
///
/// # Examples
///
/// ```
/// use ndfiltrs::{
///     BoundaryMode, InvalidPolicy, MatchedFilter, MatchedWeighting, NoiseMap, WeightKernel,
/// };
///
/// let template = WeightKernel::new([3], [0.25, 0.5, 0.25])
///     .correlate(true)
///     .mode(BoundaryMode::Reflect);
/// let filter = MatchedFilter::new(
///     &[5],
///     &template,
///     MatchedWeighting::InverseVariance,
///     InvalidPolicy::Omit { ignore_inf: false, min_count: 1 },
///     0.0,
/// )?;
/// let star = [0.0, 1.0, 2.0, 1.0, 0.0_f64];
/// let maps = filter.apply(&star, &[1.0; 5], NoiseMap::Variance)?;
/// assert_eq!(maps.amplitude[2], 4.0);
/// assert_eq!(maps.variance[2], 1.0 / 0.375);
/// # Ok::<(), ndfiltrs::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct MatchedFilter<T> {
    config: ConvolutionConfig<T>,
    weighting: MatchedWeighting,
    policy: InvalidPolicy,
    len: usize,
}

impl<T: FilterFloat> MatchedFilter<T> {
    /// Validate `template`, `weighting`, and `policy` for inputs of shape
    /// `shape`.
    ///
    /// `cval` is the value outside the array for [`BoundaryMode::Constant`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfiguration`] for an invalid template (as
    /// [`Convolution::new`] with [`WeightPolicy::Sum`]), a template without a
    /// nonzero weight, or a `min_count` outside one to the number of nonzero
    /// weights. Returns [`Error::LengthMismatch`] when the weight count
    /// differs from the product of the template shape.
    ///
    /// [`Error::InvalidConfiguration`]: crate::Error::InvalidConfiguration
    /// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
    pub fn new(
        shape: &[usize],
        template: &WeightKernel,
        weighting: MatchedWeighting,
        policy: InvalidPolicy,
        cval: T,
    ) -> Result<Self, Error> {
        let config = template.config(shape, WeightPolicy::Sum, cval)?;
        let filter = Self {
            len: config.output_len,
            config,
            weighting,
            policy,
        };
        filter.spec().validate(&filter.config)?;
        Ok(filter)
    }

    fn spec(&self) -> NoiseSpec {
        let (omit_nan, ignore_inf, min_count) = match self.policy {
            InvalidPolicy::Propagate => (false, false, 1),
            InvalidPolicy::Omit {
                ignore_inf,
                min_count,
            } => (true, ignore_inf, min_count),
        };
        NoiseSpec {
            noise: NoiseMap::Variance,
            inverse_weighting: self.weighting == MatchedWeighting::InverseVariance,
            estimator: Estimator::Amplitude,
            omit_nan,
            ignore_inf,
            min_count,
        }
    }

    /// Number of elements in each input and output buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` when the filter's shape has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Filter `input` with noise `noise` into new buffers.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] for a wrong buffer length.
    ///
    /// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
    pub fn apply(
        &self,
        input: &[T],
        noise: &[T],
        noise_map: NoiseMap,
    ) -> Result<MatchedMaps<T>, Error> {
        let mut maps = MatchedMaps {
            snr: filled(input.len())?,
            amplitude: filled(input.len())?,
            variance: filled(input.len())?,
        };
        self.apply_into(
            input,
            noise,
            noise_map,
            &mut maps.snr,
            &mut maps.amplitude,
            &mut maps.variance,
        )?;
        Ok(maps)
    }

    /// Filter `input` into `snr`, `amplitude`, and `variance` in one pass.
    /// The buffers must not overlap.
    ///
    /// `noise` holds one variance or inverse variance per input sample, as
    /// `noise_map` states.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] for a wrong buffer length.
    ///
    /// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
    pub fn apply_into(
        &self,
        input: &[T],
        noise: &[T],
        noise_map: NoiseMap,
        snr: &mut [T],
        amplitude: &mut [T],
        variance: &mut [T],
    ) -> Result<(), Error> {
        let spec = NoiseSpec {
            noise: noise_map,
            ..self.spec()
        };
        convolve_noise_into(
            input,
            noise,
            &self.config,
            &spec,
            NoiseOutputs {
                value: amplitude,
                variance,
                snr: Some(snr),
            },
        )
    }
}

/// Smooth a C-ordered `input` with one normalized 1-D kernel per listed axis.
///
/// Each output is the weighted mean of the retained samples in the product
/// window. Weights per axis must be in `0..=1`, with an odd count and at
/// least one positive entry. Axes not listed are not smoothed. `modes` has
/// one entry per array axis. `cval` is the value outside the array for
/// [`BoundaryMode::Constant`].
///
/// # Errors
///
/// Returns [`Error::InvalidConfiguration`] for invalid axes or weights, a
/// `min_count` above the positive product support, or product weights that
/// underflow. Returns [`Error::LengthMismatch`] for wrong buffer lengths.
///
/// [`Error::InvalidConfiguration`]: crate::Error::InvalidConfiguration
/// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
///
/// # Examples
///
/// ```
/// use ndfiltrs::{separable_smooth, BoundaryMode, InvalidPolicy};
///
/// let image = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let mut out = [0.0; 6];
/// // Smooth along the last axis of a 2x3 array only.
/// separable_smooth(
///     &image,
///     &[2, 3],
///     &[(1, &[0.25, 0.5, 0.25][..])],
///     &[BoundaryMode::Nearest; 2],
///     0.0,
///     InvalidPolicy::Propagate,
///     &mut out,
/// )?;
/// assert_eq!(out, [1.25, 2.0, 2.75, 4.25, 5.0, 5.75]);
/// # Ok::<(), ndfiltrs::Error>(())
/// ```
pub fn separable_smooth<T: FilterFloat>(
    input: &[T],
    shape: &[usize],
    axis_kernels: &[(usize, &[f64])],
    modes: &[BoundaryMode],
    cval: T,
    policy: InvalidPolicy,
    output: &mut [T],
) -> Result<(), Error> {
    let axes: Vec<usize> = axis_kernels.iter().map(|&(axis, _)| axis).collect();
    let kernels: Vec<Vec<f64>> = axis_kernels
        .iter()
        .map(|&(_, kernel)| kernel.to_vec())
        .collect();
    let (omit_nan, ignore_inf, min_count) = match policy {
        InvalidPolicy::Propagate => (false, false, 1),
        InvalidPolicy::Omit {
            ignore_inf,
            min_count,
        } => (true, ignore_inf, min_count),
    };
    crate::separable::smooth_into(
        input, shape, &axes, &kernels, modes, cval, omit_nan, ignore_inf, min_count, output,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_kernels_are_errors() {
        let sum = WeightPolicy::Sum;
        let normalize = WeightPolicy::Normalize {
            omit_nan: true,
            ignore_inf: false,
            min_count: 1,
        };
        let new =
            |kernel: &WeightKernel, policy| Convolution::<f64>::new(&[5], kernel, policy, 0.0);
        assert_eq!(
            new(&WeightKernel::new([3], [1.0, 1.0]), sum).err(),
            Some(Error::LengthMismatch)
        );
        for kernel in [
            WeightKernel::new([3], [1.0, f64::NAN, 1.0]),
            WeightKernel::new([0], [0.0; 0]),
            WeightKernel::new([3, 1], [1.0; 3]),
            WeightKernel::new([3], [1.0; 3]).origin([2]),
        ] {
            assert!(matches!(
                new(&kernel, sum),
                Err(Error::InvalidConfiguration(_))
            ));
        }
        assert!(matches!(
            new(&WeightKernel::new([3], [1.0, -1.0, 1.0]), normalize),
            Err(Error::InvalidConfiguration(_))
        ));
        assert!(new(&WeightKernel::new([3], [1.0, -1.0, 1.0]), sum).is_ok());
    }

    #[test]
    fn convolve_flips_and_correlate_does_not() {
        let input = [0.0_f64, 0.0, 1.0, 0.0, 0.0];
        let kernel = WeightKernel::new([3], [1.0, 2.0, 3.0]).mode(BoundaryMode::Constant);
        let convolved = Convolution::new(&[5], &kernel, WeightPolicy::Sum, 0.0)
            .unwrap()
            .apply(&input)
            .unwrap();
        let correlated = Convolution::new(&[5], &kernel.correlate(true), WeightPolicy::Sum, 0.0)
            .unwrap()
            .apply(&input)
            .unwrap();
        assert_eq!(convolved, [0.0, 1.0, 2.0, 3.0, 0.0]);
        assert_eq!(correlated, [0.0, 3.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn separable_rejects_invalid_axes_and_weights() {
        let input = [1.0_f32; 6];
        let mut out = [0.0_f32; 6];
        let modes = [BoundaryMode::Reflect; 2];
        for axis_kernels in [
            vec![(2, &[1.0_f64][..])],
            vec![(0, &[0.5, 0.5][..])],
            vec![(0, &[2.0][..])],
            vec![(0, &[1.0][..]), (0, &[1.0][..])],
        ] {
            assert!(matches!(
                separable_smooth(
                    &input,
                    &[2, 3],
                    &axis_kernels,
                    &modes,
                    0.0,
                    InvalidPolicy::Propagate,
                    &mut out
                ),
                Err(Error::InvalidConfiguration(_))
            ));
        }
    }
}

#[cfg(test)]
mod edge_tests {
    use super::*;

    #[test]
    fn zero_length_axis_is_a_no_op() {
        let plan = Convolution::<f32>::new(
            &[0, 4],
            &WeightKernel::new([3, 3], [1.0; 9]),
            WeightPolicy::Sum,
            0.0,
        )
        .unwrap();
        assert!(plan.is_empty());
        assert_eq!(plan.apply(&[]).unwrap(), Vec::<f32>::new());
        let mut out: [f32; 0] = [];
        separable_smooth(
            &[],
            &[0, 4],
            &[(1, &[0.5, 1.0, 0.5][..])],
            &[BoundaryMode::Reflect; 2],
            0.0,
            InvalidPolicy::Propagate,
            &mut out,
        )
        .unwrap();
    }
}
