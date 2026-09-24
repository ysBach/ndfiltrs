//! Robust location and scale estimators on retained finite float64 samples.
//!
//! Callers remove NaN/infinite values and validate tuning constants. Every
//! function may reorder `values`; none keeps a reference to it.

use crate::mean::Sum;
use reducers::reducers_1d::median_valid_in_place;

/// Rule used to estimate a mode from the mean, median, and spread.
///
/// Every rule uses Pearson's relation `mode = pearson * median - (pearson - 1)
/// * mean`; the caller supplies `pearson` (SExtractor's `BACK_PEARSON`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ModeEstimator {
    /// SExtractor mode: `2.5 median - 1.5 mean`, or the median when
    /// `|mean - median| / std >= 0.3`, with population `std` (`ddof=0`);
    /// the mean when `std == 0`. This is `backguess` in SExtractor's
    /// `src/back.c` with the default `BACK_PEARSON = 2.5`.
    Sex,
    /// DAOPHOT MMM-style mode: `3 median - 2 mean`.
    Mmm,
    /// IRAF-style mode: the mean when it is below the median, otherwise MMM.
    Iraf,
}

/// SExtractor skewness limit, in units of the population standard deviation,
/// at or above which the median replaces the mode formula.
const SEXTRACTOR_SKEW_LIMIT: f64 = 0.3;

/// Estimate the mode of finite samples from their mean, median, and spread.
///
/// `Mmm` and `Iraf` match astroimred's `_mode_from_stats`; `Sex` follows
/// SExtractor's `backguess` (see [`ModeEstimator`]). Differences are formed as
/// `median - mean` before scaling so equal bright summaries do not overflow.
/// Returns NaN for no samples. `values` is reordered.
pub(crate) fn mode_estimate(values: &mut [f64], estimator: ModeEstimator, pearson: f64) -> f64 {
    debug_assert!(values.iter().all(|value| value.is_finite()));
    if values.is_empty() {
        return f64::NAN;
    }
    // Mean and standard deviation accumulate in input order; take them before
    // the median selection reorders the buffer.
    let mean = crate::mean::mean_valid(values);
    let std = match estimator {
        ModeEstimator::Sex => crate::dispersion::dispersion_valid(values, 0, true),
        ModeEstimator::Mmm | ModeEstimator::Iraf => f64::NAN,
    };
    let median = median_valid_in_place(values);
    let difference = median - mean;
    let pearson_mode = median + (pearson - 1.0) * difference;
    match estimator {
        ModeEstimator::Mmm => pearson_mode,
        ModeEstimator::Iraf => {
            if mean < median {
                mean
            } else {
                pearson_mode
            }
        }
        ModeEstimator::Sex => {
            if std == 0.0 {
                mean
            } else if difference.abs() / std >= SEXTRACTOR_SKEW_LIMIT {
                median
            } else {
                pearson_mode
            }
        }
    }
}

/// Median and biweight distance unit of a sample.
///
/// After [`standardize`] returns `Spread`, `scratch[..n]` holds
/// `u = (x - M) / (c * MAD)` in the order of the (reordered) `values`.
enum Standardized {
    Empty,
    ZeroMad {
        median: f64,
    },
    Spread {
        median: f64,
        /// `c * MAD / factor`.
        width: f64,
        /// Power of two dividing deviations so they and `width` stay finite.
        factor: f64,
    },
}

fn standardize(values: &mut [f64], scratch: &mut Vec<f64>, c: f64) -> Standardized {
    debug_assert!(values.iter().all(|value| value.is_finite()));
    debug_assert!(c.is_finite() && c > 0.0);
    let count = values.len();
    if count == 0 {
        return Standardized::Empty;
    }
    let median = median_valid_in_place(values);
    scratch.clear();
    scratch.extend(values.iter().map(|&value| value - median));
    let mut factor = 1.0;
    if scratch.iter().any(|deviation| !deviation.is_finite()) {
        // Opposite finite extremes overflow the subtraction. Halving both
        // terms is exact at those magnitudes and bounds the difference.
        factor = 2.0;
        scratch.clear();
        scratch.extend(values.iter().map(|&value| value / 2.0 - median / 2.0));
    }
    // The MAD selection reorders its buffer, so work on a copy after the
    // signed deviations.
    scratch.extend_from_within(..count);
    for deviation in &mut scratch[count..] {
        *deviation = deviation.abs();
    }
    let mut mad = median_valid_in_place(&mut scratch[count..]);
    scratch.truncate(count);
    if mad == 0.0 {
        return Standardized::ZeroMad { median };
    }
    // Rescale by powers of two only if `c * MAD` overflows. Because MAD is
    // at most f64::MAX here, this takes at most ceil(log2(c)) steps.
    let mut extra = 1.0;
    while !(c * mad).is_finite() {
        mad /= 2.0;
        extra *= 2.0;
    }
    let width = c * mad;
    for deviation in scratch.iter_mut() {
        *deviation = *deviation / extra / width;
    }
    Standardized::Spread {
        median,
        width,
        factor: factor * extra,
    }
}

/// Compute the Tukey biweight location of finite samples about their median.
///
/// Matches `astropy.stats.biweight_location(values, c=c)`: with
/// `u = (x - M) / (c MAD)`, samples with `|u| < 1` receive weight
/// `(1 - u^2)^2` and the result is `M + sum(w (x - M)) / sum(w)`.
/// Returns the median when MAD is zero and NaN for no samples. `values` is
/// reordered; `scratch` is overwritten.
///
/// Notes
/// -----
/// The weighted mean is accumulated in units of `c MAD`, so it cannot
/// overflow for finite samples. Results may differ from astropy by a few
/// ulp. Deviations are rescaled by powers of two when `x - M` or `c MAD`
/// overflows; the rescaling factor stays finite when `c <= 2^1022`.
pub(crate) fn biweight_location(values: &mut [f64], scratch: &mut Vec<f64>, c: f64) -> f64 {
    let (median, width, factor) = match standardize(values, scratch, c) {
        Standardized::Empty => return f64::NAN,
        Standardized::ZeroMad { median } => return median,
        Standardized::Spread {
            median,
            width,
            factor,
        } => (median, width, factor),
    };
    let mut weighted = Sum::default();
    let mut weights = Sum::default();
    for &u in scratch.iter().filter(|u| u.abs() < 1.0) {
        let weight = (1.0 - u * u) * (1.0 - u * u);
        weighted.add(u * weight);
        weights.add(weight);
    }
    // |ratio| < 1, so the correction is bounded by `width`. No contributing
    // sample (only possible for c <= 1) gives 0/0 = NaN, as in astropy.
    let correction = width * (weighted.total() / weights.total());
    let unscaled = correction * factor;
    if unscaled.is_finite() {
        median + unscaled
    } else {
        // The result lies between finite samples, so only the intermediate
        // correction overflowed.
        (median / factor + correction) * factor
    }
}

/// Compute the biweight scale (square root of the biweight midvariance) of
/// finite samples about their median.
///
/// Matches `astropy.stats.biweight_scale(values, c=c,
/// modify_sample_size=modify_sample_size)`:
/// `sqrt(n sum((x - M)^2 (1 - u^2)^4)) / |sum((1 - u^2)(1 - 5 u^2))|`
/// over samples with `|u| < 1`, where `u = (x - M) / (c MAD)`. `n` counts
/// all samples, or only those with `|u| < 1` when `modify_sample_size` is
/// true. Returns 0 when MAD is zero and NaN for no samples. `values` is
/// reordered; `scratch` is overwritten.
///
/// Notes
/// -----
/// Squares are formed from `u`, not `x - M`, and the variance is never
/// formed, so the result overflows only if the scale itself exceeds
/// `f64::MAX` (astropy returns inf once `(x - M)^2` overflows). Results
/// may differ from astropy by a few ulp. Rescaling limits are as for
/// [`biweight_location`].
pub(crate) fn biweight_scale(
    values: &mut [f64],
    scratch: &mut Vec<f64>,
    c: f64,
    modify_sample_size: bool,
) -> f64 {
    let total = values.len();
    let (width, factor) = match standardize(values, scratch, c) {
        Standardized::Empty => return f64::NAN,
        Standardized::ZeroMad { .. } => return 0.0,
        Standardized::Spread { width, factor, .. } => (width, factor),
    };
    let mut numerator = Sum::default();
    let mut denominator = Sum::default();
    let mut kept = 0_usize;
    for &u in scratch.iter().filter(|u| u.abs() < 1.0) {
        let u2 = u * u;
        let taper = (1.0 - u2) * (1.0 - u2);
        numerator.add(u2 * taper * taper);
        denominator.add((1.0 - u2) * (1.0 - 5.0 * u2));
        kept += 1;
    }
    let count = if modify_sample_size { kept } else { total };
    let ratio = (count as f64 * numerator.total()).sqrt() / denominator.total().abs();
    width * ratio * factor
}

#[cfg(test)]
mod tests {
    use super::{biweight_location, biweight_scale, mode_estimate, ModeEstimator};

    const ESTIMATORS: [ModeEstimator; 3] =
        [ModeEstimator::Mmm, ModeEstimator::Iraf, ModeEstimator::Sex];

    // Reference values (astropy 8.0.1; astroimred for MMM/IRAF; the
    // SExtractor src/back.c backguess rule for SEx):
    //
    //   import numpy as np
    //   from astroimred._core.statistics import _mode_from_stats
    //   from astropy.stats import biweight_location as bl, biweight_scale as bs
    //   def sex(x):
    //       mean, med, std = np.mean(x), np.median(x), np.std(x)
    //       if std == 0: return mean
    //       return med if abs(mean - med) / std >= 0.3 else 2.5 * med - 1.5 * mean
    //   for x in (A, B, SKEWED, LOW_TAIL):
    //       mean, med = np.mean(x), np.median(x)
    //       print([_mode_from_stats(mean, med, method=m) for m in ("mmm", "iraf")],
    //             sex(x))
    //   for x in (A, B):
    //       print(bl(x), bl(x, c=4.0), bs(x), bs(x, modify_sample_size=True),
    //             bs(x, c=3.0, modify_sample_size=True))
    //   print(bl(A * 1e300), bs(A * 1e300))  # 1.9602591329467533e+300 inf
    const A: [f64; 11] = [3.0, -1.5, 2.25, 8.0, 0.5, 1.75, 40.0, -6.0, 2.0, 1.0, 4.5];
    const B: [f64; 8] = [10.0, 11.0, 12.0, 13.0, 50.0, 9.5, 10.5, 11.5];
    const SKEWED: [f64; 6] = [0.0, 0.0, 0.0, 0.0, 1.0, 100.0];
    const LOW_TAIL: [f64; 5] = [-100.0, 1.0, 2.0, 3.0, 4.0];

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = 4.0 * f64::EPSILON * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual:e} != {expected:e}"
        );
    }

    fn modes(values: &[f64]) -> [f64; 3] {
        // Default Pearson factors: 3 for MMM and IRAF, 2.5 for SExtractor.
        ESTIMATORS.map(|estimator| {
            let pearson = if estimator == ModeEstimator::Sex {
                2.5
            } else {
                3.0
            };
            mode_estimate(&mut values.to_vec(), estimator, pearson)
        })
    }

    fn location(values: &[f64], c: f64) -> f64 {
        biweight_location(&mut values.to_vec(), &mut Vec::new(), c)
    }

    fn scale(values: &[f64], c: f64, modify: bool) -> f64 {
        biweight_scale(&mut values.to_vec(), &mut Vec::new(), c, modify)
    }

    #[test]
    fn modes_match_references() {
        let cases: [(&[f64], [f64; 3]); 4] = [
            (
                &A,
                [-4.090909090909092, -4.090909090909092, -2.5681818181818183],
            ),
            (&B, [1.875, 1.875, 11.25]),
            (&SKEWED, [-33.666666666666664, -33.666666666666664, 0.0]),
            (&LOW_TAIL, [42.0, -18.0, 2.0]),
        ];
        for (values, expected) in cases {
            for (actual, expected) in modes(values).into_iter().zip(expected) {
                assert_close(actual, expected);
            }
        }
    }

    #[test]
    fn mode_literal_empty_and_single_sample() {
        // mean 2, median 1, population std sqrt(14 / 3): skew 0.46 >= 0.3
        // selects the median for SEx; MMM and IRAF give 1 + 2 (1 - 2).
        assert_eq!(modes(&[0.0, 1.0, 5.0]), [-1.0, -1.0, 1.0]);
        // A low tail also selects the median (the rule uses |mean - median|).
        assert_eq!(modes(&[-2.0, 0.0, 5.0])[2], 0.0);
        // Skew 0.13 < 0.3: 2.5 * 2.5 - 1.5 * 2.375.
        assert_eq!(modes(&[1.0, 2.0, 3.0, 3.5])[2], 2.6875);
        // A tight core with a low outlier stays inside the sample range.
        let core = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, -50.0];
        assert_eq!(modes(&core)[2], 10.0);
        for mode in modes(&[]) {
            assert!(mode.is_nan());
        }
        assert_eq!(modes(&[7.5]), [7.5; 3]);
        assert_eq!(modes(&[f64::MAX; 4]), [f64::MAX; 3]);
    }

    #[test]
    fn biweight_matches_astropy() {
        assert_close(location(&A, 6.0), 1.9602591329467531);
        assert_close(location(&A, 4.0), 1.8351324381177168);
        assert_close(scale(&A, 9.0, false), 3.281179906799965);
        assert_close(scale(&A, 9.0, true), 3.128482289718106);
        assert_close(scale(&A, 3.0, true), 1.7300121556223182);
        assert_close(location(&B, 6.0), 11.074571936132376);
        assert_close(location(&B, 4.0), 11.079463959276655);
        assert_close(scale(&B, 9.0, false), 1.252142592975011);
        assert_close(scale(&B, 9.0, true), 1.1712721455748067);
        assert_close(scale(&B, 3.0, true), 1.8786667369052221);
    }

    #[test]
    fn biweight_empty_single_and_zero_mad() {
        assert!(location(&[], 6.0).is_nan());
        assert!(scale(&[], 9.0, false).is_nan());
        assert_eq!(location(&[-3.5], 6.0), -3.5);
        assert_eq!(scale(&[-3.5], 9.0, false), 0.0);
        // More than half the samples equal the median, so MAD is zero.
        let values = [2.0, 2.0, 2.0, 9.0, -40.0];
        assert_eq!(location(&values, 6.0), 2.0);
        assert_eq!(scale(&values, 9.0, true), 0.0);
    }

    #[test]
    fn symmetric_samples_keep_the_median() {
        let values = [5.0, 1.0, 9.0, 4.0, 6.0, 2.0, 8.0];
        assert!((location(&values, 6.0) - 5.0).abs() <= 4.0 * f64::EPSILON * 5.0);
        // M = 0, MAD = 1, c = 1 keeps only u = 0, so the location is M.
        assert_eq!(location(&[-1.0, 0.0, 1.0], 1.0), 0.0);
        // c = 2 gives u = (-0.5, 0, 0.5): numerator 2 * 0.25 * 0.75^4,
        // denominator 1 + 2 * 0.75 * (-0.25), width c MAD = 2.
        let expected = (3.0_f64 * 2.0 * 0.25 * 0.75_f64.powi(4)).sqrt() / 0.625 * 2.0;
        assert_close(scale(&[-1.0, 0.0, 1.0], 2.0, false), expected);
    }

    #[test]
    fn biweight_survives_large_magnitudes() {
        let big = A.map(|value| value * 1e300);
        assert_close(location(&big, 6.0), 1.9602591329467533e300);
        // astropy overflows to inf here; the exact scale is finite.
        assert_close(scale(&big, 9.0, false), 3.281179906799965e300);
        // Symmetric extremes overflow c * MAD; partial sums are exact
        // multiples of one weight, so the location is exactly the median.
        let extremes = [-f64::MAX, -f64::MAX, 0.0, f64::MAX, f64::MAX];
        assert_eq!(location(&extremes, 6.0), 0.0);
        // Power-of-two rescaling is exact, so overflow paths agree bit for
        // bit with the same samples scaled into the ordinary range.
        let down = 2.0_f64.powi(-600);
        let up = 2.0_f64.powi(600);
        let opposite = [
            -f64::MAX,
            0.5 * f64::MAX,
            0.75 * f64::MAX,
            f64::MAX,
            f64::MAX,
        ];
        for values in [extremes, opposite] {
            let small = values.map(|value| value * down);
            assert_eq!(location(&values, 6.0), location(&small, 6.0) * up);
            for modify in [false, true] {
                let spread = scale(&values, 9.0, modify);
                assert!(spread.is_finite() && spread > 0.0);
                assert_eq!(spread, scale(&small, 9.0, modify) * up);
            }
        }
    }
}
