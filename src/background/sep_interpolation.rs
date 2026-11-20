//! Interpolate already-estimated SEP background nodes in row-major order.
//! The black-box measurements are described in `docs/background.md`.

use crate::FilterFloat;
use rayon::prelude::*;

/// A uniform-grid cubic has zero curvature at each endpoint. These are the
/// reciprocal pivots for eliminating its interior tridiagonal equations.
fn inverse_pivots(nodes: usize) -> Vec<f64> {
    let mut pivots = vec![0.0; nodes];
    for i in 1..nodes - 1 {
        pivots[i] = 1.0 / (2.0 - 0.25 * pivots[i - 1]);
    }
    pivots
}

/// Solve all parallel lines together. Adjacent lanes are adjacent mesh columns
/// in the vertical pass. The horizontal pass has one lane. Endpoint entries
/// in `coeff` start at zero and are never overwritten.
fn coefficients(values: &[f64], pivots: &[f64], lanes: usize, coeff: &mut [f64]) {
    for (i, &inv) in pivots.iter().enumerate().take(pivots.len() - 1).skip(1) {
        let start = i * lanes;
        let (past, remaining) = coeff.split_at_mut(start);
        let previous = &past[start - lanes..];
        let left = &values[start - lanes..start];
        let middle = &values[start..start + lanes];
        let right = &values[start + lanes..start + 2 * lanes];
        for j in 0..lanes {
            let difference = (right[j] - middle[j]) - (middle[j] - left[j]);
            remaining[j] = 3.0_f64.mul_add(difference, -0.5 * previous[j]) * inv;
        }
    }
    for i in (1..pivots.len() - 1).rev() {
        let start = i * lanes;
        let (current, next) = coeff[start..].split_at_mut(lanes);
        let multiplier = -0.5 * pivots[i];
        for (value, &right) in current.iter_mut().zip(next.iter()) {
            // The right-hand coefficient has already been divided by six.
            *value = multiplier.mul_add(right, *value) / 6.0;
        }
    }
}

// Pixel phase is observable in a unit ramp. Quantize only the legacy
// coordinate, independently of the level values and cubic arithmetic.
fn legacy_phase(value: f64) -> f64 {
    f64::from(value as f32)
}

struct XWeight {
    fraction: f64,
    left_curve: f64,
    right_curve: f64,
}

/// The first polynomial extends to the left edge. Later polynomials all
/// start at the same fractional pixel offset, so they share a weight table.
struct Horizontal {
    lengths: Vec<usize>,
    first: Vec<XWeight>,
    later: Vec<XWeight>,
}

impl Horizontal {
    fn new(nx: usize, width: usize, columns: usize) -> Self {
        let step = legacy_phase(1.0 / legacy_phase(width as f64));
        let weights = |mut b: f64, length| {
            (0..length)
                .map(|_| {
                    let a = 1.0 - b;
                    let weight = XWeight {
                        fraction: b,
                        left_curve: a * a.mul_add(a, -1.0),
                        right_curve: b * b.mul_add(b, -1.0),
                    };
                    b = legacy_phase(b + step);
                    weight
                })
                .collect()
        };

        // With width one, measurements follow the first polynomial throughout.
        let intervals = if width == 1 { 1 } else { columns - 1 };
        let mut lengths = Vec::with_capacity(intervals);
        let mut start = 0;
        for i in 1..=intervals {
            let end = if i == intervals {
                nx
            } else {
                i * width + width / 2
            };
            lengths.push(end - start);
            start = end;
        }
        let first = weights(legacy_phase(0.5 * step - 0.5), lengths[0]);
        let restart = if width % 2 == 0 { 0.5 * step } else { 0.0 };
        let later = weights(restart, lengths[1..].iter().copied().max().unwrap_or(0));
        Self {
            lengths,
            first,
            later,
        }
    }

    fn row<T>(
        &self,
        values: &[f64],
        coeff: &[f64],
        mut output: &mut [T],
        convert: impl Fn(f64) -> T,
    ) {
        for (i, &length) in self.lengths.iter().enumerate() {
            let (segment, rest) = output.split_at_mut(length);
            output = rest;
            let weights = if i == 0 { &self.first } else { &self.later };
            let (left, right) = (values[i], values[i + 1]);
            let (cl, cr) = (coeff[i], coeff[i + 1]);
            let slope = right - left;
            for (pixel, w) in segment.iter_mut().zip(weights) {
                let linear = w.fraction.mul_add(slope, left);
                *pixel = convert(w.right_curve.mul_add(cr, w.left_curve.mul_add(cl, linear)));
            }
        }
    }
}

/// Expand a mesh of already-estimated levels to an image. Both dimension pairs
/// are ordered [height, width]. The caller supplies a nonempty valid output.
pub(super) fn interpolate<T: FilterFloat + Send>(
    mesh: &[f64],
    shape: [usize; 2],
    box_size: [usize; 2],
    output: &mut [T],
) {
    let [ny, nx] = shape;
    let [height, width] = box_size;
    let rows = ny.div_ceil(height);
    let columns = nx.div_ceil(width);

    let rows_per_band = ny.div_ceil(rayon::current_num_threads());
    let band_len = rows_per_band * nx;
    if let &[value] = mesh {
        output.par_chunks_mut(band_len).for_each(|band| {
            band.fill_with(|| T::from_f64(value));
        });
        return;
    }

    // Factor out the binary exponent so differences and the forward solve
    // stay in range even near the limits of f64.
    let peak = mesh.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let exponent_mask = f64::INFINITY.to_bits();
    let scale = f64::from_bits(peak.to_bits() & exponent_mask).max(f64::MIN_POSITIVE);
    let inverse_scale = scale.recip();
    let values: Vec<_> = mesh.iter().map(|v| v * inverse_scale).collect();
    let convert = |v: f64| T::from_f64(v * scale);
    let x = (columns > 1).then(|| Horizontal::new(nx, width, columns));

    if let (1, Some(x)) = (rows, &x) {
        // A single mesh row has a piecewise linear horizontal profile.
        let mut profile = vec![0.0; nx];
        x.row(&values, &vec![0.0; columns], &mut profile, |v| v * scale);
        output.par_chunks_mut(band_len).for_each(|band| {
            for row in band.chunks_exact_mut(nx) {
                for (dst, &v) in row.iter_mut().zip(&profile) {
                    *dst = T::from_f64(v);
                }
            }
        });
        return;
    }

    let mut y_coeff = vec![0.0; mesh.len()];
    coefficients(&values, &inverse_pivots(rows), columns, &mut y_coeff);
    let x_pivots = inverse_pivots(columns);
    let phase_height = legacy_phase(height as f64);
    output
        .par_chunks_mut(band_len)
        .enumerate()
        .for_each(|(band_index, band)| {
            let mut line = vec![0.0; columns];
            let mut x_coeff = vec![0.0; columns];
            for (offset, row) in band.chunks_exact_mut(nx).enumerate() {
                let y = band_index * rows_per_band + offset;
                let box_coordinate = legacy_phase(legacy_phase(y as f64) / phase_height);
                let z = legacy_phase(box_coordinate - 0.5);
                let i = (z as usize).min(rows - 2);
                let b = z - i as f64;
                let a = 1.0 - b;
                let ca = (a * a).mul_add(a, -a);
                let cb = (b * b).mul_add(b, -b);
                let start = i * columns;
                let top = &values[start..start + columns];
                let bottom = &values[start + columns..start + 2 * columns];
                let ct = &y_coeff[start..start + columns];
                let cbottom = &y_coeff[start + columns..start + 2 * columns];
                for j in 0..columns {
                    let linear = b.mul_add(bottom[j] - top[j], top[j]);
                    line[j] = cb.mul_add(cbottom[j], ca.mul_add(ct[j], linear));
                }
                if let Some(x) = &x {
                    coefficients(&line, &x_pivots, 1, &mut x_coeff);
                    x.row(&line, &x_coeff, row, convert);
                } else {
                    row.fill_with(|| convert(line[0]));
                }
            }
        });
}
