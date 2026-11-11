//! One-pass convolution of data together with independent per-pixel noise.
//!
//! Each output combines retained samples `x_j` with coefficients `a_j`.
//! Kernel weighting uses `a_j = w_j`. Inverse-variance weighting uses
//! `a_j = w_j u_j`, where `u_j = 1 / v_j`. Three sums share one traversal:
//!
//! - numerator `N = sum_j a_j x_j`;
//! - noise `Q = Var(N) = sum_p (sum_{j -> p} a_j)^2 v_p`, grouped by source
//!   pixel `p`, because boundary modes can read one pixel through several taps;
//! - denominator `S` (`1`, `sum_j w_j`, or `sum_j w_j a_j`).
//!
//! The estimate is `N / S`, its variance is `Q / S^2`, and the signal-to-noise
//! ratio is `N / sqrt(Q)`. Interior outputs read distinct pixels, so they
//! stream the sums without storing taps. Boundary outputs, and outputs whose
//! fast arithmetic overflows or underflows, store their taps first.

use rayon::prelude::*;
use reducers::Float;

use crate::convolution::{
    finite_weighted, locate, scale_binary, split, tap_source, ConvolutionConfig, WeightedSum,
};
use crate::error::Error;
use crate::mean::Sum;

/// How the noise array that accompanies the data is expressed.
///
/// Both forms describe independent Gaussian-like noise with per-pixel
/// variance `v`. A value that is NaN or negative is invalid. A variance of
/// `+inf` (inverse variance `0`) marks a sample that carries no information.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NoiseMap {
    /// Per-pixel variance `v`.
    Variance,
    /// Per-pixel inverse variance `u = 1 / v`, as an inverse-variance weight
    /// map.
    InverseVariance,
}

/// Coefficients of a [`MatchedFilter`].
///
/// [`MatchedFilter`]: crate::MatchedFilter
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MatchedWeighting {
    /// Use the template weights `w_j` unchanged.
    Kernel,
    /// Use `w_j / v_j`. This is the minimum-variance estimate of the template
    /// amplitude for independent Gaussian noise.
    InverseVariance,
}

/// Denominator of the estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Estimator {
    /// `S = 1`: the raw weighted sum.
    Sum,
    /// `S = sum_j w_j`: the weighted mean of retained samples.
    Mean,
    /// `S = sum_j w_j a_j`: the amplitude of the template.
    Amplitude,
}

/// Validated rules for one noise-aware convolution.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NoiseSpec {
    pub(crate) noise: NoiseMap,
    pub(crate) inverse_weighting: bool,
    pub(crate) estimator: Estimator,
    pub(crate) omit_nan: bool,
    pub(crate) ignore_inf: bool,
    pub(crate) min_count: usize,
}

impl NoiseSpec {
    /// Check the rules against the kernel taps of `config`.
    pub(crate) fn validate<T>(&self, config: &ConvolutionConfig<T>) -> Result<(), Error> {
        let count = config.weights.len();
        if self.inverse_weighting && self.estimator != Estimator::Amplitude {
            return Err(Error::InvalidConfiguration(
                "inverse-variance weighting requires the amplitude estimate",
            ));
        }
        if self.estimator == Estimator::Mean && config.weights.iter().any(|&w| w < 0.0) {
            return Err(Error::InvalidConfiguration(
                "normalized weights must be nonnegative with a positive entry",
            ));
        }
        if self.estimator != Estimator::Sum && count == 0 {
            return Err(Error::InvalidConfiguration(
                "weights must have a nonzero entry",
            ));
        }
        if !self.omit_nan && (self.ignore_inf || self.min_count != 1) {
            return Err(Error::InvalidConfiguration(
                "ignore_inf and min_count require NaN omission",
            ));
        }
        if self.omit_nan && (self.min_count == 0 || self.min_count > count) {
            return Err(Error::InvalidConfiguration(
                "min_count must be between one and the number of nonzero weights",
            ));
        }
        Ok(())
    }
}

/// Finite value `mantissa * 2^exponent` with `mantissa` zero or in `[1, 2)`.
#[derive(Clone, Copy, Debug)]
struct Scaled {
    mantissa: f64,
    exponent: i32,
}

impl Scaled {
    const ZERO: Self = Self {
        mantissa: 0.0,
        exponent: 0,
    };

    fn new(value: f64, exponent: i32) -> Self {
        if value == 0.0 {
            return Self::ZERO;
        }
        let (mantissa, value_exponent) = split(value);
        Self {
            mantissa,
            exponent: value_exponent + exponent,
        }
    }

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.mantissa * other.mantissa,
            self.exponent + other.exponent,
        )
    }

    /// IEEE division of the mantissas, so a zero divisor gives an infinity or NaN.
    fn div(self, other: Self) -> f64 {
        let quotient = self.mantissa / other.mantissa;
        if quotient.is_finite() && quotient != 0.0 {
            scale_binary(quotient, self.exponent - other.exponent)
        } else {
            quotient
        }
    }

    fn sqrt(self) -> Self {
        let odd = self.exponent.rem_euclid(2);
        Self::new(
            (self.mantissa * f64::from(1 << odd)).sqrt(),
            (self.exponent - odd) / 2,
        )
    }

    fn to_f64(self) -> f64 {
        scale_binary(self.mantissa, self.exponent)
    }
}

/// Compensated sum of scaled terms without intermediate overflow.
fn scaled_sum<I: Iterator<Item = Scaled>>(terms: impl Fn() -> I) -> Scaled {
    let Some(maximum) = terms()
        .filter(|term| term.mantissa != 0.0)
        .map(|term| term.exponent)
        .max()
    else {
        return Scaled::ZERO;
    };
    let count = terms().count();
    // Each scaled term is below 2^-count_bits, so the absolute sum stays below one.
    let count_bits = usize::BITS - (count - 1).leading_zeros();
    let exponent = maximum + count_bits as i32 + 1;
    let mut sum = Sum::default();
    for term in terms() {
        sum.add(scale_binary(term.mantissa, term.exponent - exponent));
    }
    Scaled::new(sum.total(), exponent)
}

/// Sum of products of finite factors.
///
/// The fast path is a compensated `f64` sum. Overflow or underflow of a
/// product or of the total replays the terms with binary scaling.
fn product_sum<I: Iterator<Item = [f64; 3]>>(terms: impl Fn() -> I) -> Scaled {
    let mut sum = Sum::default();
    let mut exact = true;
    for [a, b, c] in terms() {
        let product = a * b * c;
        if !product.is_finite()
            || (product.abs() < f64::MIN_POSITIVE && a != 0.0 && b != 0.0 && c != 0.0)
        {
            exact = false;
            break;
        }
        sum.add(product);
    }
    let total = sum.total();
    if exact && total.is_finite() {
        return Scaled::new(total, 0);
    }
    scaled_sum(|| {
        terms().map(|[a, b, c]| {
            Scaled::new(a, 0)
                .mul(Scaled::new(b, 0))
                .mul(Scaled::new(c, 0))
        })
    })
}

/// One retained sample of a buffered output.
#[derive(Clone, Copy, Debug)]
struct Tap {
    /// C-order source pixel, or `usize::MAX` for a constant boundary sample.
    source: usize,
    weight: f64,
    value: f64,
    /// `v` for kernel weighting, `u` for inverse-variance weighting. NaN or
    /// infinite when the sample has no finite noise term.
    noise: f64,
}

/// `Q = sum_p (sum_{j -> p} w_j)^2 s_p` after grouping taps by source pixel.
fn merged_noise(taps: &mut [Tap]) -> Scaled {
    taps.sort_unstable_by_key(|tap| tap.source);
    let groups = || {
        taps.chunk_by(|left, right| left.source == right.source)
            .filter(|group| group[0].source != usize::MAX && group[0].noise.is_finite())
    };
    let mut sum = Sum::default();
    let mut exact = true;
    for group in groups() {
        let coefficient: f64 = group.iter().map(|tap| tap.weight).sum();
        let noise = group[0].noise;
        let term = coefficient * coefficient * noise;
        if !term.is_finite() || (term < f64::MIN_POSITIVE && coefficient != 0.0 && noise != 0.0) {
            exact = false;
            break;
        }
        sum.add(term);
    }
    let total = sum.total();
    if exact && total.is_finite() {
        return Scaled::new(total, 0);
    }
    scaled_sum(|| {
        groups().map(|group| {
            let coefficient = scaled_sum(|| group.iter().map(|tap| Scaled::new(tap.weight, 0)));
            coefficient
                .mul(coefficient)
                .mul(Scaled::new(group[0].noise, 0))
        })
    })
}

/// Classified noise of one sample.
enum Noise {
    /// `v` for kernel weighting or `u` for inverse-variance weighting.
    Valid(f64),
    /// NaN or negative, or zero variance under inverse-variance weighting.
    Invalid,
    /// Infinite variance: the sample carries no information.
    Uninformative,
}

fn classify(raw: f64, map: NoiseMap, inverse_weighting: bool) -> Noise {
    if raw.is_nan() || raw < 0.0 {
        return Noise::Invalid;
    }
    // Adding positive zero turns -0.0 into +0.0 before any reciprocal.
    let raw = raw + 0.0;
    let wanted = match (map, inverse_weighting) {
        (NoiseMap::Variance, false) | (NoiseMap::InverseVariance, true) => raw,
        (NoiseMap::Variance, true) | (NoiseMap::InverseVariance, false) => 1.0 / raw,
    };
    match (inverse_weighting, wanted) {
        (false, v) if v == f64::INFINITY => Noise::Uninformative,
        (true, u) if u == 0.0 => Noise::Uninformative,
        (true, u) if u == f64::INFINITY => Noise::Invalid,
        (_, value) => Noise::Valid(value),
    }
}

/// Estimate, its variance, and the signal-to-noise ratio at one position.
#[derive(Clone, Copy, Debug)]
struct Estimate {
    value: f64,
    variance: f64,
    snr: f64,
}

const NAN_ESTIMATE: Estimate = Estimate {
    value: f64::NAN,
    variance: f64::NAN,
    snr: f64::NAN,
};

/// Per-thread buffers reused across output evaluations.
struct Scratch {
    coordinates: Vec<usize>,
    pairs: Vec<(f64, f64)>,
    taps: Vec<Tap>,
}

/// Combine the sums. `value` is the finished kernel-weighted sum or mean, or
/// `None` when the estimate is `numerator / denominator`.
#[allow(clippy::too_many_arguments)]
fn finish(
    spec: &NoiseSpec,
    value: Option<f64>,
    numerator: Scaled,
    denominator: Scaled,
    noise: Scaled,
    infinity: f64,
    noise_nan: bool,
    infinite_variance: bool,
) -> Estimate {
    let infinite = infinity != 0.0;
    let value = if infinite {
        infinity
    } else {
        value.unwrap_or_else(|| numerator.div(denominator))
    };
    let (variance, snr) = if noise_nan {
        (f64::NAN, f64::NAN)
    } else if infinite_variance {
        (f64::INFINITY, value / f64::INFINITY)
    } else {
        let variance = match spec.estimator {
            Estimator::Sum => noise.to_f64(),
            _ => noise.div(denominator.mul(denominator)),
        };
        let snr = if infinite {
            value / variance.sqrt()
        } else {
            let numerator = match (spec.estimator, value) {
                (Estimator::Sum, value) => Scaled::new(value, 0),
                (Estimator::Mean, value) => Scaled::new(value, 0).mul(denominator),
                (Estimator::Amplitude, _) => numerator,
            };
            numerator.div(noise.sqrt())
        };
        (variance, snr)
    };
    Estimate {
        value,
        variance,
        snr,
    }
}

/// Evaluate one output at C-order flat index `flat`.
///
/// Interior outputs stream their sums. Boundary outputs and exceptional
/// arithmetic use `BUFFERED`, which stores taps, groups repeated source pixels,
/// and replays overflowing sums with binary scaling.
fn estimate_at<T: Float, const BUFFERED: bool>(
    data: &[T],
    noise: &[T],
    config: &ConvolutionConfig<T>,
    spec: &NoiseSpec,
    flat: usize,
    scratch: &mut Scratch,
) -> Result<Estimate, Error> {
    let interior = locate(config, flat, &mut scratch.coordinates);
    if !BUFFERED && !interior {
        return estimate_at::<T, true>(data, noise, config, spec, flat, scratch);
    }
    let Scratch {
        coordinates,
        pairs,
        taps,
    } = scratch;
    if BUFFERED {
        pairs.clear();
        taps.clear();
        let count = config.weights.len();
        pairs.try_reserve(count).map_err(|_| Error::Memory)?;
        taps.try_reserve(count).map_err(|_| Error::Memory)?;
    }
    let kernel = !spec.inverse_weighting;
    let mean = spec.estimator == Estimator::Mean;
    let amplitude = spec.estimator == Estimator::Amplitude;
    let mut data_sum = WeightedSum::new(mean);
    let mut weight_sum = Sum::default();
    let mut denominator = Sum::default();
    let mut noise_sum = Sum::default();
    let mut exceptional = false;
    let mut retained = 0usize;
    let mut positive_inf = false;
    let mut negative_inf = false;
    let mut noise_nan = false;
    let mut infinite_variance = false;
    for (tap, &weight) in config.weights.iter().enumerate() {
        let source = tap_source(config, flat, coordinates, interior, tap)?;
        let value = match source {
            Some(source) => data[source],
            None => config.cval,
        }
        .to_f64();
        if value.is_nan() {
            if spec.omit_nan {
                continue;
            }
            return Ok(NAN_ESTIMATE);
        }
        if value.is_infinite() && spec.ignore_inf {
            continue;
        }
        let class = match source {
            Some(source) => classify(noise[source].to_f64(), spec.noise, spec.inverse_weighting),
            // A constant boundary sample is exact: zero variance.
            None if kernel => Noise::Valid(0.0),
            None => Noise::Invalid,
        };
        let noise_value = match class {
            Noise::Valid(value) => value,
            _ if spec.omit_nan => continue,
            Noise::Invalid if kernel => {
                noise_nan = true;
                f64::NAN
            }
            Noise::Invalid => return Ok(NAN_ESTIMATE),
            // Zero inverse-variance weight: the sample contributes nothing.
            Noise::Uninformative if !kernel => continue,
            Noise::Uninformative => {
                infinite_variance = true;
                f64::INFINITY
            }
        };
        retained += 1;
        let coefficient = if kernel { weight } else { weight * noise_value };
        if value.is_infinite() {
            if value.is_sign_negative() == coefficient.is_sign_negative() {
                positive_inf = true;
            } else {
                negative_inf = true;
            }
        }
        if BUFFERED {
            if kernel && value.is_finite() {
                pairs.push((value, weight));
            }
            taps.push(Tap {
                source: source.unwrap_or(usize::MAX),
                weight,
                value,
                noise: noise_value,
            });
            continue;
        }
        if value.is_finite() {
            data_sum.add(value, coefficient);
        }
        if mean {
            weight_sum.add(weight);
        }
        if noise_value.is_finite() {
            let term = weight * coefficient * if kernel { noise_value } else { 1.0 };
            // Weights and noise are nonzero here unless the noise is zero.
            exceptional |= !term.is_finite() || (term < f64::MIN_POSITIVE && noise_value != 0.0);
            if amplitude && !kernel {
                // Distinct interior pixels make Q equal to sum_j w_j^2 u_j.
                denominator.add(term);
            } else {
                noise_sum.add(term);
            }
        }
        if amplitude && kernel {
            let term = weight * weight;
            exceptional |= !term.is_finite() || term < f64::MIN_POSITIVE;
            denominator.add(term);
        }
    }
    if (spec.omit_nan && retained < spec.min_count)
        || (spec.estimator != Estimator::Sum && retained == 0)
        || (positive_inf && negative_inf)
    {
        return Ok(NAN_ESTIMATE);
    }
    let infinity = if positive_inf {
        f64::INFINITY
    } else if negative_inf {
        f64::NEG_INFINITY
    } else {
        0.0
    };
    if BUFFERED {
        let finite = || taps.iter().filter(|tap| tap.value.is_finite());
        let value = match spec.estimator {
            Estimator::Amplitude => None,
            _ if infinity != 0.0 => Some(infinity),
            _ => Some(finite_weighted(pairs, mean)),
        };
        let numerator = if amplitude {
            product_sum(|| {
                finite().map(|tap| [tap.value, tap.weight, if kernel { 1.0 } else { tap.noise }])
            })
        } else {
            Scaled::ZERO
        };
        let denominator = match spec.estimator {
            Estimator::Sum => Scaled::new(1.0, 0),
            Estimator::Mean => product_sum(|| taps.iter().map(|tap| [tap.weight, 1.0, 1.0])),
            Estimator::Amplitude => product_sum(|| {
                taps.iter()
                    .map(|tap| [tap.weight, tap.weight, if kernel { 1.0 } else { tap.noise }])
            }),
        };
        let noise = merged_noise(taps);
        return Ok(finish(
            spec,
            value,
            numerator,
            denominator,
            noise,
            infinity,
            noise_nan,
            infinite_variance,
        ));
    }
    let value = if amplitude || infinity != 0.0 {
        None
    } else {
        data_sum.finish()
    };
    let numerator = if amplitude {
        data_sum.finish()
    } else {
        Some(0.0)
    };
    let denominator = match spec.estimator {
        Estimator::Sum => 1.0,
        Estimator::Mean => weight_sum.total(),
        Estimator::Amplitude => denominator.total(),
    };
    let noise_total = if amplitude && !kernel {
        denominator
    } else {
        noise_sum.total()
    };
    let replay = exceptional
        || numerator.is_none()
        || (!amplitude && infinity == 0.0 && value.is_none())
        || !denominator.is_finite()
        || !noise_total.is_finite();
    if replay {
        return estimate_at::<T, true>(data, noise, config, spec, flat, scratch);
    }
    Ok(finish(
        spec,
        value,
        Scaled::new(numerator.unwrap_or(0.0), 0),
        Scaled::new(denominator, 0),
        Scaled::new(noise_total, 0),
        infinity,
        noise_nan,
        infinite_variance,
    ))
}

/// Output buffers of one noise-aware convolution.
pub(crate) struct NoiseOutputs<'a, T> {
    pub(crate) value: &'a mut [T],
    pub(crate) variance: &'a mut [T],
    pub(crate) snr: Option<&'a mut [T]>,
}

/// Convolve `data` and propagate the independent noise in `noise`.
///
/// Every output depends only on its position, so the thread partition does
/// not change results.
pub(crate) fn convolve_noise_into<T: Float>(
    data: &[T],
    noise: &[T],
    config: &ConvolutionConfig<T>,
    spec: &NoiseSpec,
    outputs: NoiseOutputs<'_, T>,
) -> Result<(), Error> {
    let len = config.output_len;
    let NoiseOutputs {
        value,
        variance,
        snr,
    } = outputs;
    if data.len() != len
        || noise.len() != len
        || value.len() != len
        || variance.len() != len
        || snr.as_ref().is_some_and(|snr| snr.len() != len)
    {
        return Err(Error::LengthMismatch);
    }
    spec.validate(config)?;
    if len == 0 {
        return Ok(());
    }
    let chunk_len = len.div_ceil(rayon::current_num_threads().max(1));
    let rank = config.shape.len();
    let evaluate = |chunk_index: usize,
                    values: &mut [T],
                    variances: &mut [T],
                    mut snrs: Option<&mut [T]>|
     -> Result<(), Error> {
        let start = chunk_index.checked_mul(chunk_len).ok_or(Error::Overflow)?;
        let mut coordinates = Vec::new();
        coordinates
            .try_reserve_exact(rank)
            .map_err(|_| Error::Memory)?;
        coordinates.resize(rank, 0usize);
        let mut scratch = Scratch {
            coordinates,
            pairs: Vec::new(),
            taps: Vec::new(),
        };
        for local in 0..values.len() {
            let flat = start.checked_add(local).ok_or(Error::Overflow)?;
            let estimate = estimate_at::<T, false>(data, noise, config, spec, flat, &mut scratch)?;
            values[local] = T::from_f64(estimate.value);
            variances[local] = T::from_f64(estimate.variance);
            if let Some(snrs) = snrs.as_deref_mut() {
                snrs[local] = T::from_f64(estimate.snr);
            }
        }
        Ok(())
    };
    match snr {
        Some(snr) => value
            .par_chunks_mut(chunk_len)
            .zip(variance.par_chunks_mut(chunk_len))
            .zip(snr.par_chunks_mut(chunk_len))
            .enumerate()
            .try_for_each(|(index, ((values, variances), snrs))| {
                evaluate(index, values, variances, Some(snrs))
            }),
        None => value
            .par_chunks_mut(chunk_len)
            .zip(variance.par_chunks_mut(chunk_len))
            .enumerate()
            .try_for_each(|(index, (values, variances))| evaluate(index, values, variances, None)),
    }
}

#[cfg(test)]
mod tests {
    use super::{convolve_noise_into, Estimator, NoiseMap, NoiseOutputs, NoiseSpec, Scaled};
    use crate::boundary::BoundaryMode;
    use crate::convolution::{convolve_into, ConvolutionConfig, WeightPolicy};

    fn spec(estimator: Estimator, omit_nan: bool, inverse_weighting: bool) -> NoiseSpec {
        NoiseSpec {
            noise: NoiseMap::Variance,
            inverse_weighting,
            estimator,
            omit_nan,
            ignore_inf: false,
            min_count: 1,
        }
    }

    fn config(
        shape: &[usize],
        kernel_shape: &[usize],
        weights: &[f64],
        mode: BoundaryMode,
        cval: f64,
        policy: WeightPolicy,
    ) -> ConvolutionConfig<f64> {
        ConvolutionConfig::new(
            shape.to_vec(),
            kernel_shape.to_vec(),
            weights.to_vec(),
            vec![mode; shape.len()],
            cval,
            vec![0; shape.len()],
            true,
            policy,
        )
        .unwrap()
    }

    fn run(
        data: &[f64],
        noise: &[f64],
        config: &ConvolutionConfig<f64>,
        spec: &NoiseSpec,
    ) -> [Vec<f64>; 3] {
        let mut outputs = [
            vec![0.0; data.len()],
            vec![0.0; data.len()],
            vec![0.0; data.len()],
        ];
        let [value, variance, snr] = &mut outputs;
        convolve_noise_into(
            data,
            noise,
            config,
            spec,
            NoiseOutputs {
                value,
                variance,
                snr: Some(snr),
            },
        )
        .unwrap();
        outputs
    }

    const NORMALIZE: WeightPolicy = WeightPolicy::Normalize {
        omit_nan: true,
        ignore_inf: false,
        min_count: 1,
    };

    #[test]
    fn boundary_repeats_add_coefficients_before_squaring() {
        // Reflect maps index -1 to 0, so output 0 reads pixel 0 twice.
        let config = config(
            &[3],
            &[3],
            &[1.0, 2.0, 3.0],
            BoundaryMode::Reflect,
            0.0,
            WeightPolicy::Sum,
        );
        let [value, variance, snr] = run(
            &[1.0, 2.0, 4.0],
            &[1.0, 10.0, 100.0],
            &config,
            &spec(Estimator::Sum, false, false),
        );
        assert_eq!(value, [9.0, 17.0, 22.0]);
        // (1 + 2)^2 * 1 + 3^2 * 10, not 1 + 4 + 90.
        assert_eq!(variance[0], 99.0);
        assert_eq!(variance[1], 1.0 + 40.0 + 900.0);
        // (2 + 3)^2 * 100 + 1 * 10.
        assert_eq!(variance[2], 2510.0);
        assert_eq!(snr[0], 9.0 / 99.0_f64.sqrt());
    }

    #[test]
    fn normalized_variance_divides_by_retained_weight_squared() {
        let config = config(
            &[3],
            &[3],
            &[1.0, 2.0, 1.0],
            BoundaryMode::Constant,
            f64::NAN,
            NORMALIZE,
        );
        let [value, variance, _] = run(
            &[1.0, f64::NAN, 5.0],
            &[2.0, 7.0, 6.0],
            &config,
            &spec(Estimator::Mean, true, false),
        );
        assert_eq!(value[1], 3.0);
        assert_eq!(variance[1], (2.0 + 6.0) / 4.0);
        // Output 0 keeps pixel 0 (weight 2) only, after omitting NaN padding and data.
        assert_eq!((value[0], variance[0]), (1.0, 2.0));
    }

    #[test]
    fn data_output_matches_convolution_bitwise() {
        let shape = [5, 7];
        let len = 35;
        let mut data: Vec<f64> = (0..len)
            .map(|index| ((index * 37 % 23) as f64 - 11.0) / 3.0)
            .collect();
        data[4] = f64::NAN;
        data[17] = f64::INFINITY;
        let noise: Vec<f64> = (0..len).map(|index| (index % 5) as f64 + 0.5).collect();
        let weights = [0.5, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0, 0.25, 0.0, 1.0, 1.5, 0.5];
        for mode in [
            BoundaryMode::Reflect,
            BoundaryMode::Constant,
            BoundaryMode::Nearest,
            BoundaryMode::Mirror,
            BoundaryMode::Wrap,
        ] {
            for (policy, estimator, omit) in [
                (WeightPolicy::Sum, Estimator::Sum, false),
                (NORMALIZE, Estimator::Mean, true),
            ] {
                let config = config(&shape, &[3, 4], &weights, mode, 1.5, policy);
                let mut expected = vec![0.0; len];
                convolve_into(&data, &config, &mut expected).unwrap();
                let [value, _, _] = run(&data, &noise, &config, &spec(estimator, omit, false));
                for (actual, expected) in value.iter().zip(&expected) {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
            }
        }
    }

    #[test]
    fn noise_rules_follow_the_policy() {
        let sum = config(
            &[3],
            &[3],
            &[1.0; 3],
            BoundaryMode::Nearest,
            0.0,
            WeightPolicy::Sum,
        );
        let data = [1.0, 2.0, 3.0];
        // Propagation keeps the data and marks only the variance.
        let [value, variance, snr] = run(
            &data,
            &[1.0, -1.0, 1.0],
            &sum,
            &spec(Estimator::Sum, false, false),
        );
        assert_eq!(value[1], 6.0);
        assert!(variance[1].is_nan() && snr[1].is_nan());
        let [value, variance, snr] = run(
            &data,
            &[1.0, f64::INFINITY, 1.0],
            &sum,
            &spec(Estimator::Sum, false, false),
        );
        assert_eq!((value[1], variance[1], snr[1]), (6.0, f64::INFINITY, 0.0));
        // Omission drops the sample from both outputs.
        let mean = config(&[3], &[3], &[1.0; 3], BoundaryMode::Nearest, 0.0, NORMALIZE);
        for bad in [f64::NAN, -1.0, f64::INFINITY] {
            let [value, variance, _] = run(
                &data,
                &[1.0, bad, 1.0],
                &mean,
                &spec(Estimator::Mean, true, false),
            );
            assert_eq!((value[1], variance[1]), (2.0, 0.5));
        }
        // NaN data propagate to every output.
        let [value, variance, snr] = run(
            &[1.0, f64::NAN, 3.0],
            &[1.0; 3],
            &sum,
            &spec(Estimator::Sum, false, false),
        );
        assert!(value[1].is_nan() && variance[1].is_nan() && snr[1].is_nan());
    }

    #[test]
    fn inverse_variance_input_matches_variance_input() {
        let config = config(
            &[4],
            &[3],
            &[1.0, 2.0, 1.0],
            BoundaryMode::Mirror,
            0.0,
            NORMALIZE,
        );
        let data = [1.0, 2.0, 4.0, 8.0];
        let variance = [0.5, 0.25, f64::INFINITY, 2.0];
        let inverse = [2.0, 4.0, 0.0, 0.5];
        for estimator in [Estimator::Mean, Estimator::Amplitude] {
            for inverse_weighting in [false, true] {
                if inverse_weighting && estimator != Estimator::Amplitude {
                    continue;
                }
                let from_variance = run(
                    &data,
                    &variance,
                    &config,
                    &spec(estimator, true, inverse_weighting),
                );
                let mut ivar = spec(estimator, true, inverse_weighting);
                ivar.noise = NoiseMap::InverseVariance;
                assert_eq!(run(&data, &inverse, &config, &ivar), from_variance);
            }
        }
    }

    #[test]
    fn matched_filter_interior_variance_is_reciprocal_denominator() {
        let config = config(
            &[5],
            &[3],
            &[1.0, 2.0, 1.0],
            BoundaryMode::Reflect,
            0.0,
            WeightPolicy::Sum,
        );
        let data = [0.0, 1.0, 3.0, 1.0, 0.0];
        let noise = [1.0, 2.0, 4.0, 2.0, 1.0];
        let [amplitude, variance, snr] = run(
            &data,
            &noise,
            &config,
            &spec(Estimator::Amplitude, true, true),
        );
        let denominator = 1.0 / 2.0 + 4.0 / 4.0 + 1.0 / 2.0;
        let numerator = 1.0 / 2.0 + 2.0 * 3.0 / 4.0 + 1.0 / 2.0;
        assert_eq!(amplitude[2], numerator / denominator);
        assert_eq!(variance[2], 1.0 / denominator);
        assert_eq!(snr[2], numerator / denominator.sqrt());
        // Kernel weighting: A = sum w x / sum w^2.
        let [amplitude, variance, snr] = run(
            &data,
            &noise,
            &config,
            &spec(Estimator::Amplitude, true, false),
        );
        assert_eq!(amplitude[2], 8.0 / 6.0);
        assert_eq!(variance[2], (2.0 + 16.0 + 2.0) / 36.0);
        assert_eq!(snr[2], 8.0 / 20.0_f64.sqrt());
    }

    #[test]
    fn scaling_recovers_overflowing_terms() {
        let config = config(
            &[3],
            &[3],
            &[1.0; 3],
            BoundaryMode::Wrap,
            0.0,
            WeightPolicy::Sum,
        );
        let [value, variance, snr] = run(
            &[1.0; 3],
            &[f64::MAX; 3],
            &config,
            &spec(Estimator::Sum, false, false),
        );
        assert_eq!(value, [3.0; 3]);
        assert_eq!(variance, [f64::INFINITY; 3]);
        // 3 / sqrt(3 MAX), although 3 MAX overflows.
        let expected = 3.0_f64.sqrt() / f64::MAX.sqrt();
        assert!((snr[0] - expected).abs() <= 4.0 * f64::EPSILON * expected);
        let mean = config_mean_large();
        let [value, variance, _] = run(
            &[2.0; 3],
            &[f64::MAX / 4.0; 3],
            &mean,
            &spec(Estimator::Mean, true, false),
        );
        assert_eq!(value, [2.0; 3]);
        // 3 w^2 v / (3 w)^2 = v / 3, although w^2 overflows.
        let expected = f64::MAX / 12.0;
        assert!((variance[1] - expected).abs() <= 4.0 * f64::EPSILON * expected);
    }

    fn config_mean_large() -> ConvolutionConfig<f64> {
        config(&[3], &[3], &[1e200; 3], BoundaryMode::Wrap, 0.0, NORMALIZE)
    }

    #[test]
    fn scaled_square_root_handles_odd_exponents() {
        for value in [2.0, 8.0, 0.5, 1e-300, 3e300] {
            let root = Scaled::new(value, 0).sqrt().to_f64();
            assert_eq!(root, value.sqrt());
        }
    }

    #[test]
    fn thread_count_does_not_change_results() {
        let config = config(
            &[6, 5],
            &[3, 3],
            &[1.0; 9],
            BoundaryMode::Reflect,
            0.0,
            NORMALIZE,
        );
        let data: Vec<f64> = (0..30).map(|index| (index * 7 % 11) as f64).collect();
        let noise: Vec<f64> = (0..30).map(|index| 1.0 + (index % 3) as f64).collect();
        let mut results = Vec::new();
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            results.push(pool.install(|| {
                run(
                    &data,
                    &noise,
                    &config,
                    &spec(Estimator::Amplitude, true, true),
                )
            }));
        }
        assert_eq!(results[0], results[1]);
    }

    #[test]
    fn invalid_rules_and_lengths_are_rejected() {
        let config = config(
            &[3],
            &[3],
            &[1.0, -1.0, 1.0],
            BoundaryMode::Reflect,
            0.0,
            WeightPolicy::Sum,
        );
        let mut out = [vec![0.0; 3], vec![0.0; 3]];
        let [value, variance] = &mut out;
        for bad in [
            spec(Estimator::Mean, true, false),
            spec(Estimator::Sum, true, true),
            NoiseSpec {
                min_count: 4,
                ..spec(Estimator::Amplitude, true, false)
            },
            NoiseSpec {
                ignore_inf: true,
                ..spec(Estimator::Amplitude, false, false)
            },
        ] {
            assert!(convolve_noise_into(
                &[1.0; 3],
                &[1.0; 3],
                &config,
                &bad,
                NoiseOutputs {
                    value,
                    variance,
                    snr: None
                }
            )
            .is_err());
        }
        assert!(convolve_noise_into(
            &[1.0; 3],
            &[1.0; 2],
            &config,
            &spec(Estimator::Sum, false, false),
            NoiseOutputs {
                value,
                variance,
                snr: None
            }
        )
        .is_err());
    }
}
