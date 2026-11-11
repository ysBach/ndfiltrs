//! Public convolution and separable-smoothing API.

use crate::boundary::BoundaryMode;
use crate::convolution::{convolve_into, ConvolutionConfig, WeightPolicy};
use crate::error::Error;
use crate::plan::{FilterFloat, Modes};
use crate::reduction::InvalidPolicy;

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
