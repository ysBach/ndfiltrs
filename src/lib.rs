//! NaN-aware N-dimensional filters on C-ordered buffers.
//!
//! `ndfiltrs` is the Rust core of the Python package of the same name.
//! Arrays are flat `f32` or `f64` slices in C (row-major) order with an
//! explicit shape. Geometry and boundary modes follow `scipy.ndimage`.
//!
//! - Window statistics: [`WindowFilter`], [`window_filter`].
//! - Robust statistics (percentile, MAD, mode, biweight, sigma clipping):
//!   [`RobustFilter`].
//! - Linear filters: [`Convolution`], [`separable_smooth`].
//! - Mask-directed repair: [`fixpix_linear`], [`fixpix_median`],
//!   [`fixpix_mean`].
//!
//! # Examples
//!
//! ```
//! use ndfiltrs::{BoundaryMode, InvalidPolicy, Neighborhood, Reduction, WindowFilter};
//!
//! // 3x3 NaN-skipping median of a 4x4 image, reusable for many frames.
//! let skip_nan = InvalidPolicy::Omit { ignore_inf: false, min_count: 1 };
//! let plan = WindowFilter::new(
//!     &[4, 4],
//!     &Neighborhood::rectangular([3, 3]).mode(BoundaryMode::Nearest),
//!     Reduction::Median(skip_nan),
//!     0.0_f64,
//! )?;
//! let mut frame = vec![1.0; 16];
//! frame[5] = f64::NAN;
//! frame[10] = 100.0;
//! let filtered = plan.apply(&frame)?;
//! assert_eq!(filtered[5], 1.0);
//! assert_eq!(filtered[10], 1.0);
//! # Ok::<(), ndfiltrs::Error>(())
//! ```
//!
//! # Features
//!
//! - `python`, `extension-module`: build the PyO3 extension used by the
//!   Python package. Rust users do not need them.
//! - `benchmark`, `test-hooks`: internal, unstable, and excluded from the
//!   semver contract.

mod boundary;
mod boxcar;
mod convolution;
mod dispersion;
mod error;
mod exact_sum;
mod extrema;
mod filter_nd;
mod fixpix;
mod histogram;
mod linear_plan;
mod location;
mod mean;
mod median_network;
mod parallel;
mod plan;
mod reduction;
mod repair;
mod robust;
mod robust_plan;
mod separable;
mod window;

pub use boundary::BoundaryMode;
pub use convolution::WeightPolicy;
pub use error::Error;
pub use linear_plan::{separable_smooth, Convolution, WeightKernel};
pub use location::ModeEstimator;
pub use plan::{window_filter, FilterFloat, Neighborhood, WindowFilter};
pub use reduction::{InvalidPolicy, Reduction};
pub use repair::{fixpix_linear, fixpix_mean, fixpix_median};
pub use robust::{ClipCenter, ClipConfig, ClipSpread, ClipStatistic, RobustReduction};
pub use robust_plan::RobustFilter;

#[cfg(feature = "benchmark")]
#[doc(hidden)]
pub mod benchmark;

#[cfg(feature = "python")]
mod pyapi;
