//! Public window-filter API: validated plans that run on C-ordered buffers.

use crate::boundary::BoundaryMode;
use crate::error::Error;
use crate::exact_sum::ExactFloat;
use crate::filter_nd::FilterConfig;
use crate::histogram::Statistic;
use crate::median_network::NetworkFloat;
use crate::reduction::{InvalidPolicy, Reduction};
use crate::window::Window;

/// Floating-point element types accepted by the filters: `f32` and `f64`.
///
/// This trait is sealed. Other crates cannot implement it.
pub trait FilterFloat: ExactFloat + NetworkFloat + Send + Sync {}

impl FilterFloat for f32 {}
impl FilterFloat for f64 {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Modes {
    All(BoundaryMode),
    PerAxis(Vec<BoundaryMode>),
}

/// Neighborhood geometry: window shape, optional footprint, origin, and
/// boundary modes.
///
/// The conventions follow `scipy.ndimage`. Tap `k` along an axis reads
/// input index `center + k - (size / 2 + origin)`. The defaults are origin
/// `0` on every axis and [`BoundaryMode::Reflect`] on every axis.
///
/// # Examples
///
/// ```
/// use ndfiltrs::{BoundaryMode, Neighborhood};
///
/// // A 3x5 rectangle that repeats the edge value outside the array.
/// let rect = Neighborhood::rectangular([3, 5]).mode(BoundaryMode::Nearest);
///
/// // A plus-shaped footprint (C order) with per-axis modes.
/// let plus = Neighborhood::footprint(
///     [3, 3],
///     [false, true, false, true, true, true, false, true, false],
/// )
/// .modes([BoundaryMode::Wrap, BoundaryMode::Mirror]);
/// # let _ = (rect, plus);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Neighborhood {
    pub(crate) size: Vec<usize>,
    pub(crate) footprint: Option<Vec<bool>>,
    origin: Option<Vec<isize>>,
    modes: Modes,
}

impl Neighborhood {
    /// Select every element of a rectangular window of shape `size`.
    pub fn rectangular(size: impl Into<Vec<usize>>) -> Self {
        Self {
            size: size.into(),
            footprint: None,
            origin: None,
            modes: Modes::All(BoundaryMode::Reflect),
        }
    }

    /// Select the `true` elements of a boolean `mask` of shape `shape`.
    ///
    /// `mask` is in C (row-major) order and must have `shape.iter().product()`
    /// elements. Plan construction rejects a mask with no `true` element.
    pub fn footprint(shape: impl Into<Vec<usize>>, mask: impl Into<Vec<bool>>) -> Self {
        Self {
            size: shape.into(),
            footprint: Some(mask.into()),
            origin: None,
            modes: Modes::All(BoundaryMode::Reflect),
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

    pub(crate) fn resolved_origin(&self) -> Vec<isize> {
        self.origin
            .clone()
            .unwrap_or_else(|| vec![0; self.size.len()])
    }

    /// Build the window for `shape`, materializing every tap.
    pub(crate) fn window(&self, shape: &[usize]) -> Result<Window, Error> {
        Window::new(
            shape.to_vec(),
            &self.size,
            self.footprint.as_deref(),
            &self.resolved_origin(),
            self.resolved_modes(),
        )
    }

    pub(crate) fn resolved_modes(&self) -> Vec<BoundaryMode> {
        match &self.modes {
            Modes::All(mode) => vec![*mode; self.size.len()],
            Modes::PerAxis(modes) => modes.clone(),
        }
    }
}

#[derive(Clone, Debug)]
enum Kernel<T> {
    Direct(FilterConfig<T>),
    Extrema {
        size: Vec<usize>,
        origin: Vec<isize>,
        modes: Vec<BoundaryMode>,
        maximum: bool,
        policy: InvalidPolicy,
    },
    Rolling {
        size: Vec<usize>,
        origin: Vec<isize>,
        modes: Vec<BoundaryMode>,
        policy: InvalidPolicy,
    },
    Histogram {
        size: Vec<usize>,
        origin: Vec<isize>,
        modes: Vec<BoundaryMode>,
        policy: InvalidPolicy,
        statistic: Statistic,
    },
}

/// Check rectangular geometry without building per-tap offsets.
///
/// The separable kernels repeat these checks when they run. Checking here
/// makes plan construction reject the same inputs as the direct path, with
/// memory independent of the window volume.
fn validate_rectangular(
    shape: &[usize],
    size: &[usize],
    origin: &[isize],
    modes: &[BoundaryMode],
    policy: InvalidPolicy,
) -> Result<(), Error> {
    let rank = shape.len();
    if rank == 0 {
        return Err(Error::InvalidConfiguration(
            "filtering requires at least one dimension",
        ));
    }
    if size.len() != rank || origin.len() != rank || modes.len() != rank {
        return Err(Error::InvalidConfiguration(
            "array, window, origin, and mode must have equal rank",
        ));
    }
    if size.contains(&0) {
        return Err(Error::InvalidConfiguration("window sizes must be positive"));
    }
    shape.iter().try_fold(1usize, |product, &value| {
        product.checked_mul(value).ok_or(Error::Overflow)
    })?;
    let volume = size.iter().try_fold(1usize, |product, &value| {
        product.checked_mul(value).ok_or(Error::Overflow)
    })?;
    for (&length, &shift) in size.iter().zip(origin) {
        let length = isize::try_from(length).map_err(|_| Error::Overflow)?;
        if shift < -(length / 2) || shift > (length - 1) / 2 {
            return Err(Error::InvalidConfiguration(
                "origin must satisfy -(size // 2) <= origin <= (size - 1) // 2",
            ));
        }
    }
    if let InvalidPolicy::Omit { min_count, .. } = policy {
        if min_count == 0 || min_count > volume {
            return Err(Error::InvalidConfiguration(
                "min_count must be between 1 and the number of window elements",
            ));
        }
    }
    Ok(())
}

/// A validated window filter for one array shape.
///
/// Build the plan once with [`WindowFilter::new`], then apply it to any
/// number of C-ordered inputs with that shape. Work runs on the current
/// Rayon thread pool.
///
/// # Algorithms
///
/// - [`Reduction::Minimum`] and [`Reduction::Maximum`] with a rectangular
///   neighborhood use exact separable sliding extrema. The result equals a
///   direct window traversal bit for bit.
/// - [`Reduction::RollingMean`] needs a rectangular neighborhood. It uses
///   exact running sums and rounds once to `T`.
/// - [`Reduction::HistogramMedian`] and [`Reduction::HistogramRank`] need a
///   rectangular neighborhood. They rank the samples of each output block and
///   slide a histogram of those ranks. The result equals
///   [`Reduction::Median`] or [`Reduction::Rank`] bit for bit.
/// - All other cases visit each window directly.
///
/// # Examples
///
/// ```
/// use ndfiltrs::{InvalidPolicy, Neighborhood, Reduction, WindowFilter};
///
/// let image = [1.0_f64, 2.0, f64::NAN, 4.0, 5.0, 6.0];
/// let skip_nan = InvalidPolicy::Omit { ignore_inf: false, min_count: 1 };
/// let plan = WindowFilter::new(
///     &[2, 3],
///     &Neighborhood::rectangular([1, 3]),
///     Reduction::Median(skip_nan),
///     0.0,
/// )?;
/// let smoothed = plan.apply(&image)?;
/// assert_eq!(smoothed, [1.0, 1.5, 2.0, 4.0, 5.0, 6.0]);
/// # Ok::<(), ndfiltrs::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct WindowFilter<T> {
    shape: Vec<usize>,
    cval: T,
    kernel: Kernel<T>,
}

impl<T: FilterFloat> WindowFilter<T> {
    /// Validate the geometry and reduction for inputs of shape `shape`.
    ///
    /// `cval` is the value outside the array for [`BoundaryMode::Constant`].
    /// Other modes ignore it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfiguration`] for a rank mismatch, an empty
    /// window or footprint, an origin outside the window, a `min_count` outside `1..=window size`, or a rolling mean or
    /// histogram reduction with a footprint. Returns [`Error::LengthMismatch`] when a footprint mask
    /// length differs from the product of its shape. Returns
    /// [`Error::Overflow`] or [`Error::Memory`] when the shape is too large.
    ///
    /// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
    /// [`Error::InvalidConfiguration`]: crate::Error::InvalidConfiguration
    /// [`Error::Overflow`]: crate::Error::Overflow
    /// [`Error::Memory`]: crate::Error::Memory
    pub fn new(
        shape: &[usize],
        neighborhood: &Neighborhood,
        reduction: Reduction,
        cval: T,
    ) -> Result<Self, Error> {
        let origin = neighborhood.resolved_origin();
        let modes = neighborhood.resolved_modes();
        let rectangular = neighborhood.footprint.is_none();
        let kernel = match reduction {
            Reduction::Minimum(policy) | Reduction::Maximum(policy) if rectangular => {
                validate_rectangular(shape, &neighborhood.size, &origin, &modes, policy)?;
                Kernel::Extrema {
                    size: neighborhood.size.clone(),
                    origin,
                    modes,
                    maximum: matches!(reduction, Reduction::Maximum(_)),
                    policy,
                }
            }
            Reduction::RollingMean(policy) => {
                if !rectangular {
                    return Err(Error::InvalidConfiguration(
                        "rolling means require rectangular size, not footprint",
                    ));
                }
                validate_rectangular(shape, &neighborhood.size, &origin, &modes, policy)?;
                Kernel::Rolling {
                    size: neighborhood.size.clone(),
                    origin,
                    modes,
                    policy,
                }
            }
            Reduction::HistogramMedian(policy) | Reduction::HistogramRank(policy, _) => {
                if !rectangular {
                    return Err(Error::InvalidConfiguration(
                        "histogram reductions require rectangular size, not footprint",
                    ));
                }
                validate_rectangular(shape, &neighborhood.size, &origin, &modes, policy)?;
                crate::histogram::validate_support(shape, &neighborhood.size)?;
                Kernel::Histogram {
                    size: neighborhood.size.clone(),
                    origin,
                    modes,
                    policy,
                    statistic: match reduction {
                        Reduction::HistogramRank(_, rank) => Statistic::Rank(rank),
                        _ => Statistic::Median,
                    },
                }
            }
            _ => Kernel::Direct(FilterConfig::from_window(
                neighborhood.window(shape)?,
                cval,
                reduction,
            )?),
        };
        Ok(Self {
            shape: shape.to_vec(),
            cval,
            kernel,
        })
    }

    /// The input and output shape of this plan.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of elements in each input and output buffer.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// `true` when the plan's shape has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Filter `input` into a new buffer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] when `input.len()` differs from
    /// [`WindowFilter::len`], and [`Error::Memory`] when allocation fails.
    ///
    /// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
    /// [`Error::Memory`]: crate::Error::Memory
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
    /// Returns [`Error::LengthMismatch`] when either length differs from
    /// [`WindowFilter::len`], and [`Error::Memory`] when workspace
    /// allocation fails.
    ///
    /// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
    /// [`Error::Memory`]: crate::Error::Memory
    pub fn apply_into(&self, input: &[T], output: &mut [T]) -> Result<(), Error> {
        let len = self.len();
        if input.len() != len || output.len() != len {
            return Err(Error::LengthMismatch);
        }
        match &self.kernel {
            Kernel::Direct(config) => crate::parallel::filter_nd_into(input, config, output),
            Kernel::Extrema {
                size,
                origin,
                modes,
                maximum,
                policy,
            } => crate::extrema::extrema_into(
                input,
                &self.shape,
                size,
                origin,
                modes,
                self.cval,
                *maximum,
                *policy,
                output,
            ),
            Kernel::Rolling {
                size,
                origin,
                modes,
                policy,
            } => crate::boxcar::mean_into(
                input,
                &self.shape,
                size,
                origin,
                modes,
                self.cval,
                *policy,
                output,
            ),
            Kernel::Histogram {
                size,
                origin,
                modes,
                policy,
                statistic,
            } => crate::histogram::order_statistic_into(
                input,
                &self.shape,
                size,
                origin,
                modes,
                self.cval,
                *policy,
                *statistic,
                output,
            ),
        }
    }
}

/// Apply one window filter to a C-ordered `input` of shape `shape`.
///
/// This builds a [`WindowFilter`] and applies it once. Build the plan
/// directly to filter many inputs with the same shape.
///
/// # Errors
///
/// Returns the errors of [`WindowFilter::new`] and [`WindowFilter::apply`].
///
/// # Examples
///
/// ```
/// use ndfiltrs::{window_filter, InvalidPolicy, Neighborhood, Reduction};
///
/// let row = [3.0_f32, 1.0, 2.0, 9.0];
/// let out = window_filter(
///     &row,
///     &[4],
///     &Neighborhood::rectangular([3]),
///     Reduction::Maximum(InvalidPolicy::Propagate),
///     0.0,
/// )?;
/// assert_eq!(out, [3.0, 3.0, 9.0, 9.0]);
/// # Ok::<(), ndfiltrs::Error>(())
/// ```
pub fn window_filter<T: FilterFloat>(
    input: &[T],
    shape: &[usize],
    neighborhood: &Neighborhood,
    reduction: Reduction,
    cval: T,
) -> Result<Vec<T>, Error> {
    WindowFilter::new(shape, neighborhood, reduction, cval)?.apply(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SKIP: InvalidPolicy = InvalidPolicy::Omit {
        ignore_inf: false,
        min_count: 1,
    };

    #[test]
    fn separable_paths_match_direct_traversal() {
        let shape = [4, 5];
        let input: Vec<f64> = (0..20)
            .map(|i| {
                if i == 7 {
                    f64::NAN
                } else {
                    (i * 7 % 11) as f64
                }
            })
            .collect();
        let rect = Neighborhood::rectangular([3, 2]).mode(BoundaryMode::Mirror);
        let mask = Neighborhood::footprint([3, 2], [true; 6]).mode(BoundaryMode::Mirror);
        for reduction in [Reduction::Maximum(SKIP), Reduction::Minimum(SKIP)] {
            let fast = window_filter(&input, &shape, &rect, reduction, 0.0).unwrap();
            let direct = window_filter(&input, &shape, &mask, reduction, 0.0).unwrap();
            assert_eq!(fast, direct);
        }
        let rolling =
            window_filter(&input, &shape, &rect, Reduction::RollingMean(SKIP), 0.0).unwrap();
        let direct = window_filter(&input, &shape, &rect, Reduction::Mean(SKIP), 0.0).unwrap();
        for (a, b) in rolling.iter().zip(&direct) {
            assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0));
        }
    }

    #[test]
    fn plans_reject_invalid_input() {
        let rect = Neighborhood::rectangular([3]);
        let plan = WindowFilter::<f32>::new(&[4], &rect, Reduction::Median(SKIP), 0.0).unwrap();
        assert_eq!(plan.apply(&[1.0; 3]), Err(Error::LengthMismatch));
        let footprint = Neighborhood::footprint([3], [true, false, true]);
        for reduction in [
            Reduction::RollingMean(SKIP),
            Reduction::HistogramMedian(SKIP),
            Reduction::HistogramRank(SKIP, 0),
        ] {
            assert!(matches!(
                WindowFilter::<f32>::new(&[4], &footprint, reduction, 0.0),
                Err(Error::InvalidConfiguration(_))
            ));
        }
        let too_many = InvalidPolicy::Omit {
            ignore_inf: false,
            min_count: 4,
        };
        for reduction in [
            Reduction::Maximum(too_many),
            Reduction::RollingMean(too_many),
            Reduction::HistogramMedian(too_many),
        ] {
            assert!(matches!(
                WindowFilter::<f64>::new(&[4], &rect, reduction, 0.0),
                Err(Error::InvalidConfiguration(_))
            ));
        }
        assert!(matches!(
            WindowFilter::<f64>::new(&[4, 4], &rect, Reduction::Median(SKIP), 0.0),
            Err(Error::InvalidConfiguration(_))
        ));
    }

    fn all_paths() -> [Reduction; 4] {
        [
            Reduction::Median(SKIP),
            Reduction::Maximum(SKIP),
            Reduction::RollingMean(SKIP),
            Reduction::HistogramMedian(SKIP),
        ]
    }

    #[test]
    fn invalid_geometry_is_an_error_on_every_path() {
        let config = |shape: &[usize], hood: &Neighborhood, reduction| {
            matches!(
                WindowFilter::<f64>::new(shape, hood, reduction, 0.0),
                Err(Error::InvalidConfiguration(_))
            )
        };
        for reduction in all_paths() {
            assert!(config(
                &[],
                &Neighborhood::rectangular([0usize; 0]),
                reduction
            ));
            assert!(config(&[4], &Neighborhood::rectangular([0]), reduction));
            assert!(config(
                &[4],
                &Neighborhood::rectangular([3]).origin([2]),
                reduction
            ));
            assert!(config(
                &[4],
                &Neighborhood::rectangular([3]).modes([BoundaryMode::Wrap; 2]),
                reduction
            ));
        }
        for reduction in [Reduction::Median(SKIP), Reduction::Maximum(SKIP)] {
            let short_mask = Neighborhood::footprint([3], [true, true]);
            assert_eq!(
                WindowFilter::<f64>::new(&[4], &short_mask, reduction, 0.0).err(),
                Some(Error::LengthMismatch)
            );
            assert!(config(
                &[4],
                &Neighborhood::footprint([3], [false; 3]),
                reduction
            ));
        }
    }

    #[test]
    fn zero_length_axes_produce_empty_output() {
        for reduction in all_paths() {
            let plan = WindowFilter::<f32>::new(
                &[0, 3],
                &Neighborhood::rectangular([3, 3]),
                reduction,
                0.0,
            )
            .unwrap();
            assert!(plan.is_empty());
            assert_eq!(plan.apply(&[]).unwrap(), Vec::<f32>::new());
        }
    }

    #[test]
    fn rank_and_ddof_beyond_the_window_give_nan() {
        let input = [1.0_f64, 2.0, 3.0, 4.0];
        let hood = Neighborhood::rectangular([3]);
        for reduction in [
            Reduction::Rank(InvalidPolicy::Propagate, 3),
            Reduction::HistogramRank(InvalidPolicy::Propagate, 3),
            Reduction::Variance(InvalidPolicy::Propagate, 3),
            Reduction::Std(InvalidPolicy::Propagate, 5),
        ] {
            let out = window_filter(&input, &[4], &hood, reduction, 0.0).unwrap();
            assert!(
                out.iter().all(|value| value.is_nan()),
                "{reduction:?}: {out:?}"
            );
        }
    }

    #[test]
    fn network_median_matches_direct_median_on_a_large_window() {
        let shape = [9, 11];
        let input: Vec<f32> = (0..99)
            .map(|i| {
                if i % 13 == 0 {
                    f32::NAN
                } else {
                    ((i * 37) % 23) as f32
                }
            })
            .collect();
        let hood = Neighborhood::rectangular([7, 9]).mode(BoundaryMode::Mirror);
        let network =
            window_filter(&input, &shape, &hood, Reduction::NetworkMedian(SKIP), 0.0).unwrap();
        let direct = window_filter(&input, &shape, &hood, Reduction::Median(SKIP), 0.0).unwrap();
        assert_eq!(network, direct);
    }
}
