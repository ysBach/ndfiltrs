use reducers::Float;

/// Variance or standard deviation of retained values, in float64 arithmetic.
/// Shift before normalizing to retain small variations on a large background.
pub(crate) fn dispersion_valid<T: Float>(values: &[T], ddof: usize, std: bool) -> T {
    if values.len() <= ddof || values.iter().any(|value| !value.is_finite()) {
        return T::nan();
    }
    let reference = values[0].to_f64();
    // Opposite finite extremes can overflow subtraction. Halving both terms
    // bounds their difference; this branch depends only on arithmetic overflow.
    let halve = values
        .iter()
        .any(|value| !(value.to_f64() - reference).is_finite());
    let delta = |value: T| {
        if halve {
            value.to_f64() / 2.0 - reference / 2.0
        } else {
            value.to_f64() - reference
        }
    };
    let scale = values
        .iter()
        .map(|&value| delta(value).abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return T::from_f64(0.0);
    }
    // Normalized offsets are in [-1,1]. Both sums stay bounded by a constant
    // times the number of samples, rather than their physical magnitudes.
    let mut sum = crate::mean::Sum::default();
    for &value in values {
        sum.add(delta(value) / scale);
    }
    let center = sum.total() / values.len() as f64;
    let mut squares = crate::mean::Sum::default();
    for &value in values {
        let residual = delta(value) / scale - center;
        squares.add(residual * residual);
    }
    let normalized = squares.total() / (values.len() - ddof) as f64;
    let result = if std {
        // Do not form variance: its square can overflow/underflow even when
        // the standard deviation is representable in the output dtype.
        let result = scale * normalized.sqrt();
        if halve {
            result * 2.0
        } else {
            result
        }
    } else {
        // Multiply in this order to avoid prematurely squaring the scale.
        let result = scale * (scale * normalized);
        if halve {
            result * 4.0
        } else {
            result
        }
    };
    T::from_f64(result)
}

#[cfg(test)]
mod tests {
    use super::dispersion_valid;

    #[test]
    fn centered_scatter_survives_large_offsets() {
        assert_eq!(
            dispersion_valid(&[1e16, 1e16 + 2.0, 1e16 + 4.0], 0, false),
            8.0 / 3.0
        );
        assert_eq!(dispersion_valid(&[1e16, 1e16 + 2.0], 0, false), 1.0);
        assert_eq!(dispersion_valid(&[f64::MAX; 3], 0, false), 0.0);
    }

    #[test]
    fn standard_deviation_survives_variance_overflow_and_underflow() {
        assert_eq!(dispersion_valid(&[-f64::MAX, f64::MAX], 0, true), f64::MAX);
        assert!(dispersion_valid(&[-f64::MAX, f64::MAX], 0, false).is_infinite());
        assert_eq!(dispersion_valid(&[-1e-200, 1e-200], 0, true), 1e-200);
        assert_eq!(dispersion_valid(&[-1e-200, 1e-200], 0, false), 0.0);
    }

    #[test]
    fn count_correction_and_nonfinite_contract() {
        assert_eq!(dispersion_valid(&[1.0, 2.0, 3.0], 1, false), 1.0);
        for values in [&[][..], &[1.0][..], &[f64::INFINITY, 1.0][..]] {
            assert_eq!(
                dispersion_valid(values, 1, true).to_bits(),
                f64::NAN.to_bits()
            );
        }
    }
}
