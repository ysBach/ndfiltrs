//! Rust kernels and optional Python extension for `ndfiltrs`.

#[cfg(any(test, feature = "python", feature = "benchmark"))]
mod boundary;
#[cfg(any(test, feature = "python"))]
mod boxcar;
#[cfg(any(test, feature = "python"))]
mod convolution;
#[cfg(any(test, feature = "python", feature = "benchmark"))]
mod dispersion;
#[cfg(any(test, feature = "python", feature = "benchmark"))]
mod error;
#[cfg(any(test, feature = "python"))]
mod exact_sum;
#[cfg(any(test, feature = "python"))]
mod extrema;
#[cfg(any(test, feature = "python", feature = "benchmark"))]
mod filter_nd;
#[cfg(any(test, feature = "python"))]
mod fixpix;
#[cfg(any(test, feature = "python"))]
mod location;
#[cfg(any(test, feature = "python", feature = "benchmark"))]
mod mean;
#[cfg(any(test, feature = "python", feature = "benchmark"))]
mod median_network;
#[cfg(any(test, feature = "python"))]
mod parallel;
#[cfg(any(test, feature = "python", feature = "benchmark"))]
mod reduction;
#[cfg(any(test, feature = "python"))]
mod robust;
#[cfg(feature = "python")]
mod separable;
#[cfg(any(test, feature = "python", feature = "benchmark"))]
mod window;

#[cfg(feature = "benchmark")]
pub mod benchmark;

#[cfg(feature = "python")]
mod pyapi;
