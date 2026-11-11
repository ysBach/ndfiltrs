//! Public mask-directed pixel repair (`fixpix`).
//!
//! Each function repairs `data` in place. `mask` has one entry per element of
//! `data`, in C order. `true` marks a bad pixel. Unmasked pixels are never
//! written. NaN and infinite values are not masked automatically.

use crate::convolution::{repair_masked, WeightPolicy};
use crate::error::Error;
use crate::filter_nd::{repair_median, FilterConfig};
use crate::linear_plan::WeightKernel;
use crate::plan::{FilterFloat, Neighborhood};
use crate::reduction::{InvalidPolicy, Reduction};

fn positions(mask: &[bool], len: usize) -> Result<Vec<usize>, Error> {
    if mask.len() != len {
        return Err(Error::LengthMismatch);
    }
    let count = mask.iter().filter(|&&bad| bad).count();
    let mut positions = Vec::new();
    positions
        .try_reserve_exact(count)
        .map_err(|_| Error::Memory)?;
    positions.extend(
        mask.iter()
            .enumerate()
            .filter_map(|(index, &bad)| bad.then_some(index)),
    );
    Ok(positions)
}

/// Repair masked pixels by linear interpolation along the shortest bad run.
///
/// This follows astroimred's `fixpix`. For each masked pixel, the shortest
/// contiguous masked run through it (along any axis) is interpolated between
/// the adjacent unmasked pixels. At an edge, the single adjacent pixel is
/// copied. A run that spans its whole axis is not written. `priority` lists
/// every axis once, highest priority first, to break ties. `None` uses
/// reversed axis order (last axis first).
///
/// # Errors
///
/// Returns [`Error::LengthMismatch`] when `data` or `mask` does not match
/// `shape`, and [`Error::InvalidConfiguration`] for an invalid `priority`.
///
/// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
/// [`Error::InvalidConfiguration`]: crate::Error::InvalidConfiguration
///
/// # Examples
///
/// ```
/// let mut row = [1.0_f64, 0.0, 0.0, 4.0];
/// ndfiltrs::fixpix_linear(&mut row, &[4], &[false, true, true, false], None)?;
/// assert_eq!(row, [1.0, 2.0, 3.0, 4.0]);
/// # Ok::<(), ndfiltrs::Error>(())
/// ```
pub fn fixpix_linear<T: FilterFloat>(
    data: &mut [T],
    shape: &[usize],
    mask: &[bool],
    priority: Option<&[usize]>,
) -> Result<(), Error> {
    let len = shape.iter().try_fold(1usize, |product, &value| {
        product.checked_mul(value).ok_or(Error::Overflow)
    })?;
    if data.len() != len {
        return Err(Error::LengthMismatch);
    }
    let positions = positions(mask, len)?;
    let default: Vec<usize>;
    let priority = match priority {
        Some(priority) => priority,
        None => {
            default = (0..shape.len()).rev().collect();
            &default
        }
    };
    crate::fixpix::fixpix(data, shape, &positions, priority)
}

/// Replace masked pixels by the median of the unmasked pixels in their
/// neighborhood.
///
/// Only masked windows are evaluated. Originally masked samples never enter
/// a median. NaN samples are dropped, and infinite samples too when
/// `ignore_inf`. A window with fewer than `min_count` retained samples gives
/// NaN.
///
/// # Errors
///
/// Returns the errors of [`WindowFilter::new`](crate::WindowFilter::new)
/// for the neighborhood, and [`Error::LengthMismatch`] when `data` or `mask`
/// does not match `shape`.
///
/// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
///
/// # Examples
///
/// ```
/// use ndfiltrs::Neighborhood;
///
/// let mut row = [1.0_f64, 9.0, 99.0, 3.0, 5.0];
/// let mask = [false, false, true, false, false];
/// ndfiltrs::fixpix_median(&mut row, &[5], &mask, &Neighborhood::rectangular([3]), 0.0, false, 1)?;
/// assert_eq!(row[2], 6.0);
/// # Ok::<(), ndfiltrs::Error>(())
/// ```
pub fn fixpix_median<T: FilterFloat>(
    data: &mut [T],
    shape: &[usize],
    mask: &[bool],
    neighborhood: &Neighborhood,
    cval: T,
    ignore_inf: bool,
    min_count: usize,
) -> Result<(), Error> {
    let config = FilterConfig::from_window(
        neighborhood.window(shape)?,
        cval,
        Reduction::Median(InvalidPolicy::Omit {
            ignore_inf,
            min_count,
        }),
    )?;
    let positions = positions(mask, config.output_len())?;
    repair_median(data, &config, &positions)
}

/// Replace masked pixels by the weighted mean of the unmasked pixels in
/// their neighborhood, like Astropy's `interpolate_replace_nans`.
///
/// `kernel` weights must be finite and nonnegative with a positive entry.
/// Each repaired value equals a NaN-skipping normalized convolution of
/// `data` with every masked pixel set to NaN. Only masked pixels are
/// evaluated.
///
/// # Errors
///
/// Returns the errors of [`Convolution::new`](crate::Convolution::new) for
/// the kernel, and [`Error::LengthMismatch`] when `data` or `mask` does not
/// match `shape`.
///
/// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
///
/// # Examples
///
/// ```
/// use ndfiltrs::WeightKernel;
///
/// let mut row = [1.0_f64, 2.0, 99.0, 6.0, 5.0];
/// let mask = [false, false, true, false, false];
/// let kernel = WeightKernel::new([3], [1.0, 1.0, 1.0]);
/// ndfiltrs::fixpix_mean(&mut row, &[5], &mask, &kernel, 0.0, false, 1)?;
/// assert_eq!(row[2], 4.0);
/// # Ok::<(), ndfiltrs::Error>(())
/// ```
pub fn fixpix_mean<T: FilterFloat>(
    data: &mut [T],
    shape: &[usize],
    mask: &[bool],
    kernel: &WeightKernel,
    cval: T,
    ignore_inf: bool,
    min_count: usize,
) -> Result<(), Error> {
    let config = kernel.config(
        shape,
        WeightPolicy::Normalize {
            omit_nan: true,
            ignore_inf,
            min_count,
        },
        cval,
    )?;
    let len = shape.iter().try_fold(1usize, |product, &value| {
        product.checked_mul(value).ok_or(Error::Overflow)
    })?;
    let positions = positions(mask, len)?;
    repair_masked(data, &config, &positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_and_priority_errors() {
        let mut data = [1.0_f64; 6];
        assert_eq!(
            fixpix_linear(&mut data, &[2, 3], &[false; 5], None),
            Err(Error::LengthMismatch)
        );
        assert_eq!(
            fixpix_linear(&mut data, &[2, 2], &[false; 4], None),
            Err(Error::LengthMismatch)
        );
        assert!(matches!(
            fixpix_linear(&mut data, &[2, 3], &[false; 6], Some(&[0, 0])),
            Err(Error::InvalidConfiguration(_))
        ));
        let hood = Neighborhood::rectangular([3, 3]);
        assert_eq!(
            fixpix_median(&mut data, &[2, 3], &[false; 5], &hood, 0.0, false, 1),
            Err(Error::LengthMismatch)
        );
        let kernel = WeightKernel::new([3, 3], [1.0; 9]);
        assert_eq!(
            fixpix_mean(&mut data, &[2, 3], &[false; 5], &kernel, 0.0, false, 1),
            Err(Error::LengthMismatch)
        );
    }

    #[test]
    fn unmasked_pixels_are_untouched() {
        let original = [1.0_f32, f32::NAN, 3.0, 4.0];
        let mut data = original;
        fixpix_mean(
            &mut data,
            &[4],
            &[false, false, true, false],
            &WeightKernel::new([3], [1.0; 3]),
            0.0,
            false,
            1,
        )
        .unwrap();
        assert!(data[1].is_nan());
        assert_eq!([data[0], data[3]], [original[0], original[3]]);
        assert_eq!(data[2], 4.0);
    }
}

#[cfg(test)]
mod edge_tests {
    use super::*;

    #[test]
    fn empty_and_full_masks() {
        let hood = Neighborhood::rectangular([3, 3]);
        let kernel = WeightKernel::new([3, 3], [1.0; 9]);
        let original: Vec<f64> = (0..12).map(f64::from).collect();
        for full in [false, true] {
            let mask = [full; 12];
            let mut linear = original.clone();
            fixpix_linear(&mut linear, &[3, 4], &mask, None).unwrap();
            assert_eq!(linear, original, "linear, full={full}");
            let mut median = original.clone();
            fixpix_median(&mut median, &[3, 4], &mask, &hood, 0.0, false, 1).unwrap();
            let mut mean = original.clone();
            fixpix_mean(&mut mean, &[3, 4], &mask, &kernel, 0.0, false, 1).unwrap();
            if full {
                // No unmasked neighbor exists, so every repaired value is NaN.
                assert!(median.iter().chain(&mean).all(|value| value.is_nan()));
            } else {
                assert_eq!(median, original);
                assert_eq!(mean, original);
            }
        }
    }

    #[test]
    fn zero_length_axes_are_no_ops() {
        let mut data: [f32; 0] = [];
        fixpix_linear(&mut data, &[0, 3], &[], None).unwrap();
        fixpix_median(
            &mut data,
            &[0, 3],
            &[],
            &Neighborhood::rectangular([3, 3]),
            0.0,
            false,
            1,
        )
        .unwrap();
        fixpix_mean(
            &mut data,
            &[0, 3],
            &[],
            &WeightKernel::new([3, 3], [1.0; 9]),
            0.0,
            false,
            1,
        )
        .unwrap();
    }
}
