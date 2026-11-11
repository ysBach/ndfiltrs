//! Savitzky-Golay filters: least-squares polynomial fits in sliding windows.

use rayon::prelude::*;
use reducers::Float;

use crate::boundary::{map_index, BoundaryMode};
use crate::convolution::finite_weighted;
use crate::error::Error;
use crate::plan::FilterFloat;
use crate::reduction::InvalidPolicy;

/// Polynomial space fitted in each Savitzky-Golay window.
///
/// A monomial is `x_0^k_0 * x_1^k_1 * ...` in window offsets. Axes with
/// window length one have exponent zero.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PolynomialBasis {
    /// Monomials with `k_0 + k_1 + ... <= p`. The degree must be less than
    /// every window length above one.
    TotalDegree(usize),
    /// Monomials with `k_a <= p[a]` on each axis `a`, one entry per axis.
    /// Each degree must be less than the window length on its axis. On a
    /// complete rectangular window, the fit equals one-dimensional
    /// Savitzky-Golay filters applied axis by axis.
    PerAxisDegree(Vec<usize>),
}

/// How a Savitzky-Golay window treats positions near an array edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FitBoundary {
    /// Extend the array with a [`BoundaryMode`] and fit the centered window.
    /// The padded samples take part in the fit, as in
    /// `scipy.signal.savgol_filter` with modes other than `"interp"`.
    Pad(BoundaryMode),
    /// Keep the window inside the array. Near an edge, the window moves
    /// inward and the fit is evaluated at the output position, as in
    /// `scipy.signal.savgol_filter(mode="interp")`. The window length must
    /// not exceed the axis length.
    Interp,
}

#[derive(Clone, Debug, PartialEq)]
enum Boundaries {
    All(FitBoundary),
    PerAxis(Vec<FitBoundary>),
}

/// Window, polynomial, derivative, weights, and edge handling of a
/// Savitzky-Golay filter.
///
/// The window has one odd length per array axis. A length of one leaves
/// that axis unfiltered. The defaults are no derivative, unit sample
/// spacing, uniform weights, and [`FitBoundary::Interp`] on every axis.
///
/// # Examples
///
/// ```
/// use ndfiltrs::{FitBoundary, PolynomialBasis, SavgolKernel};
///
/// // Quadratic surface fit in 5x5 windows; derivative along the last axis.
/// let kernel = SavgolKernel::new([5, 5], PolynomialBasis::TotalDegree(2))
///     .derivative([0, 1])
///     .delta([1.0, 0.5])
///     .boundary(FitBoundary::Interp);
/// # let _ = kernel;
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SavgolKernel {
    window: Vec<usize>,
    basis: PolynomialBasis,
    derivative: Option<Vec<usize>>,
    delta: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
    boundaries: Boundaries,
}

impl SavgolKernel {
    /// Fit `basis` in windows with odd lengths `window`, one per axis.
    pub fn new(window: impl Into<Vec<usize>>, basis: PolynomialBasis) -> Self {
        Self {
            window: window.into(),
            basis,
            derivative: None,
            delta: None,
            weights: None,
            boundaries: Boundaries::All(FitBoundary::Interp),
        }
    }

    /// Set the derivative order per axis. All zeros (the default) returns
    /// the fitted value. The monomial with these exponents must be in the
    /// basis.
    pub fn derivative(mut self, orders: impl Into<Vec<usize>>) -> Self {
        self.derivative = Some(orders.into());
        self
    }

    /// Set the sample spacing per axis. Derivatives are divided by
    /// `delta[a]^k[a]`. The default is one on every axis.
    pub fn delta(mut self, delta: impl Into<Vec<f64>>) -> Self {
        self.delta = Some(delta.into());
        self
    }

    /// Use finite, nonnegative least-squares weights in C order with the
    /// window shape. A zero weight removes that window position from every
    /// fit. The default weights are all one.
    pub fn weights(mut self, weights: impl Into<Vec<f64>>) -> Self {
        self.weights = Some(weights.into());
        self
    }

    /// Use one edge treatment on every axis.
    pub fn boundary(mut self, boundary: FitBoundary) -> Self {
        self.boundaries = Boundaries::All(boundary);
        self
    }

    /// Use one edge treatment per axis.
    pub fn boundaries(mut self, boundaries: impl Into<Vec<FitBoundary>>) -> Self {
        self.boundaries = Boundaries::PerAxis(boundaries.into());
        self
    }
}

/// Dense least-squares workspace: Householder QR, then a one-sided Jacobi
/// SVD of the triangular factor.
///
/// For a design `B` with `n >= r` rows, `B = Q R` and `R = U S V^T`. The
/// minimum-norm coefficients of a linear functional `g` of the fitted
/// polynomial are `Q U S^-1 V^T g`.
#[derive(Clone, Debug, Default)]
struct Solver {
    rows: usize,
    terms: usize,
    /// Column-major `rows x terms` design, then Householder vectors.
    matrix: Vec<f64>,
    tau: Vec<f64>,
    diagonal: Vec<f64>,
    /// Column-major `terms x terms`: `R`, then `R V = U S`.
    left: Vec<f64>,
    /// Column-major `terms x terms` right singular vectors.
    right: Vec<f64>,
    sigma: Vec<f64>,
    projected: Vec<f64>,
}

/// LAPACK `dgesvj` also stops one-sided Jacobi after 30 sweeps.
const MAX_SWEEPS: usize = 30;

impl Solver {
    fn reserve(&mut self, rows: usize, terms: usize) -> Result<(), Error> {
        let cells = rows.checked_mul(terms).ok_or(Error::Overflow)?;
        let square = terms.checked_mul(terms).ok_or(Error::Overflow)?;
        for (buffer, length) in [
            (&mut self.matrix, cells),
            (&mut self.tau, terms),
            (&mut self.diagonal, terms),
            (&mut self.left, square),
            (&mut self.right, square),
            (&mut self.sigma, terms),
            (&mut self.projected, rows),
        ] {
            buffer.clear();
            buffer
                .try_reserve_exact(length)
                .map_err(|_| Error::Memory)?;
            buffer.resize(length, 0.0);
        }
        self.rows = rows;
        self.terms = terms;
        Ok(())
    }

    /// Factor the filled design. Return `false` when it is numerically
    /// rank deficient: `sigma_min <= sigma_max * max(rows, terms) * eps`,
    /// the default tolerance of `numpy.linalg.matrix_rank`.
    fn factor(&mut self) -> bool {
        let (n, r) = (self.rows, self.terms);
        if n < r {
            return false;
        }
        for j in 0..r {
            let (before, rest) = self.matrix.split_at_mut((j + 1) * n);
            let column = &mut before[j * n + j..];
            let norm = column.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm == 0.0 {
                self.tau[j] = 0.0;
                self.diagonal[j] = 0.0;
                continue;
            }
            let alpha = if column[0] > 0.0 { -norm } else { norm };
            column[0] -= alpha;
            let beta = column.iter().map(|x| x * x).sum::<f64>();
            let tau = 2.0 / beta;
            self.tau[j] = tau;
            self.diagonal[j] = alpha;
            for k in 0..r - j - 1 {
                let target = &mut rest[k * n + j..(k + 1) * n];
                let dot = column
                    .iter()
                    .zip(target.iter())
                    .map(|(v, a)| v * a)
                    .sum::<f64>();
                let scale = tau * dot;
                for (a, v) in target.iter_mut().zip(column.iter()) {
                    *a -= scale * v;
                }
            }
        }
        for k in 0..r {
            for i in 0..r {
                self.left[k * r + i] = match i.cmp(&k) {
                    std::cmp::Ordering::Less => self.matrix[k * n + i],
                    std::cmp::Ordering::Equal => self.diagonal[k],
                    std::cmp::Ordering::Greater => 0.0,
                };
                self.right[k * r + i] = if i == k { 1.0 } else { 0.0 };
            }
        }
        self.jacobi();
        let largest = self.sigma.iter().copied().fold(0.0, f64::max);
        let tolerance = largest * n.max(r) as f64 * f64::EPSILON;
        largest > 0.0 && self.sigma.iter().all(|&value| value > tolerance)
    }

    fn jacobi(&mut self) {
        let r = self.terms;
        let orthogonal = r as f64 * f64::EPSILON;
        for _ in 0..MAX_SWEEPS {
            let mut rotated = false;
            for p in 0..r {
                for q in p + 1..r {
                    let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                    for i in 0..r {
                        let (x, y) = (self.left[p * r + i], self.left[q * r + i]);
                        alpha += x * x;
                        beta += y * y;
                        gamma += x * y;
                    }
                    // A computed dot product of length r can differ from the exact
                    // one by r * eps * |x| |y| (Higham, gamma_r), so columns
                    // within that bound are orthogonal to working precision.
                    if gamma == 0.0 || gamma.abs() <= orthogonal * (alpha * beta).sqrt() {
                        continue;
                    }
                    rotated = true;
                    let zeta = (beta - alpha) / (2.0 * gamma);
                    let t = zeta.signum() / (zeta.abs() + zeta.hypot(1.0));
                    let c = 1.0 / t.hypot(1.0);
                    let s = c * t;
                    for matrix in [&mut self.left, &mut self.right] {
                        for i in 0..r {
                            let (x, y) = (matrix[p * r + i], matrix[q * r + i]);
                            matrix[p * r + i] = c * x - s * y;
                            matrix[q * r + i] = s * x + c * y;
                        }
                    }
                }
            }
            if !rotated {
                break;
            }
        }
        for j in 0..r {
            self.sigma[j] = self.left[j * r..(j + 1) * r]
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
        }
    }

    /// Write `Q U S^-1 V^T functional` into `self.projected`. Requires a
    /// successful `factor`.
    fn project(&mut self, functional: &[f64]) {
        let (n, r) = (self.rows, self.terms);
        let projected = &mut self.projected;
        projected.fill(0.0);
        for j in 0..r {
            let along = (0..r)
                .map(|i| self.right[j * r + i] * functional[i])
                .sum::<f64>();
            // U S^-1 = (R V) S^-2 column by column.
            let scale = along / (self.sigma[j] * self.sigma[j]);
            for (value, left) in projected.iter_mut().zip(&self.left[j * r..(j + 1) * r]) {
                *value += left * scale;
            }
        }
        for j in (0..r).rev() {
            let vector = &self.matrix[j * n + j..(j + 1) * n];
            let target = &mut projected[j..];
            let dot = vector
                .iter()
                .zip(target.iter())
                .map(|(v, y)| v * y)
                .sum::<f64>();
            let scale = self.tau[j] * dot;
            for (y, v) in target.iter_mut().zip(vector) {
                *y -= scale * v;
            }
        }
    }
}

/// Validated geometry, basis, and precomputed full-window coefficients.
#[derive(Clone, Debug)]
pub(crate) struct SavgolConfig<T> {
    shape: Vec<usize>,
    strides: Vec<usize>,
    half: Vec<usize>,
    boundaries: Vec<FitBoundary>,
    cval: T,
    policy: InvalidPolicy,
    /// Per positive-weight tap: signed offsets, one per axis.
    offsets: Vec<i128>,
    /// Per tap flat displacement; empty when one is not representable.
    flat_offsets: Vec<isize>,
    /// Square roots of the weights divided by the largest weight.
    root_weights: Vec<f64>,
    /// Row-major `taps x terms` weighted design on offsets scaled to [-1, 1].
    design: Vec<f64>,
    /// Row-major `terms x rank` exponents.
    exponents: Vec<u32>,
    terms: usize,
    derivative: Vec<u32>,
    /// `1 / half` per axis (one on unfiltered axes).
    scale: Vec<f64>,
    /// Chain-rule factor `prod (scale / delta)^k`.
    factor: f64,
    /// Full centered-window coefficients per tap.
    center: Vec<f64>,
    /// Full window: row-major `taps x terms` map from functional to coefficients.
    projector: Vec<f64>,
    /// Centered coefficients in window C order, zero for zero-weight positions.
    dense: Vec<f64>,
    output_len: usize,
}

fn checked_product(values: &[usize]) -> Result<usize, Error> {
    values.iter().try_fold(1usize, |product, &value| {
        product.checked_mul(value).ok_or(Error::Overflow)
    })
}

fn filled(len: usize, value: f64) -> Result<Vec<f64>, Error> {
    let mut result = Vec::new();
    result.try_reserve_exact(len).map_err(|_| Error::Memory)?;
    result.resize(len, value);
    Ok(result)
}

fn strides(shape: &[usize]) -> Result<Vec<usize>, Error> {
    let mut result = vec![1usize; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        result[axis] = result[axis + 1]
            .checked_mul(shape[axis + 1])
            .ok_or(Error::Overflow)?;
    }
    Ok(result)
}

/// Enumerate basis exponents, stopping once more than `limit` terms exist.
fn enumerate_exponents(
    bounds: &[usize],
    total: Option<usize>,
    limit: usize,
) -> Result<Vec<u32>, Error> {
    fn visit(
        axis: usize,
        bounds: &[usize],
        remaining: Option<usize>,
        current: &mut Vec<u32>,
        output: &mut Vec<u32>,
        count: &mut usize,
        limit: usize,
    ) -> Result<(), Error> {
        if axis == bounds.len() {
            *count += 1;
            if *count > limit {
                return Err(Error::InvalidConfiguration(
                    "the polynomial has more terms than positive-weight window samples",
                ));
            }
            output
                .try_reserve(current.len())
                .map_err(|_| Error::Memory)?;
            output.extend_from_slice(current);
            return Ok(());
        }
        let top = remaining.map_or(bounds[axis], |left| left.min(bounds[axis]));
        for exponent in 0..=top {
            current[axis] = u32::try_from(exponent).map_err(|_| Error::Overflow)?;
            visit(
                axis + 1,
                bounds,
                remaining.map(|left| left - exponent),
                current,
                output,
                count,
                limit,
            )?;
        }
        current[axis] = 0;
        Ok(())
    }
    let mut current = vec![0u32; bounds.len()];
    let mut output = Vec::new();
    let mut count = 0usize;
    visit(
        0,
        bounds,
        total,
        &mut current,
        &mut output,
        &mut count,
        limit,
    )?;
    Ok(output)
}

/// `d^k t^n / dt^k`.
fn monomial_derivative(exponent: u32, order: u32, t: f64) -> f64 {
    if exponent < order {
        return 0.0;
    }
    let falling =
        (exponent - order + 1..=exponent).fold(1.0, |product, value| product * value as f64);
    falling * t.powi((exponent - order) as i32)
}

impl<T: Copy> SavgolConfig<T> {
    pub(crate) fn new(
        shape: &[usize],
        kernel: &SavgolKernel,
        policy: InvalidPolicy,
        cval: T,
    ) -> Result<Self, Error> {
        let rank = shape.len();
        if rank == 0 {
            return Err(Error::InvalidConfiguration(
                "Savitzky-Golay filtering requires at least one dimension",
            ));
        }
        let window = &kernel.window;
        let boundaries = match &kernel.boundaries {
            Boundaries::All(boundary) => vec![*boundary; rank],
            Boundaries::PerAxis(boundaries) => boundaries.clone(),
        };
        let derivative = kernel.derivative.clone().unwrap_or_else(|| vec![0; rank]);
        let delta = kernel.delta.clone().unwrap_or_else(|| vec![1.0; rank]);
        if window.len() != rank
            || boundaries.len() != rank
            || derivative.len() != rank
            || delta.len() != rank
        {
            return Err(Error::InvalidConfiguration(
                "window, derivative, delta, and boundaries need one entry per axis",
            ));
        }
        if window.iter().any(|&length| length % 2 == 0) {
            return Err(Error::InvalidConfiguration(
                "Savitzky-Golay window lengths must be odd",
            ));
        }
        if delta.iter().any(|&step| !(step.is_finite() && step > 0.0)) {
            return Err(Error::InvalidConfiguration(
                "delta must be finite and positive",
            ));
        }
        let output_len = checked_product(shape)?;
        if output_len > 0 {
            for axis in 0..rank {
                if boundaries[axis] == FitBoundary::Interp && window[axis] > shape[axis] {
                    return Err(Error::InvalidConfiguration(
                        "interp requires window lengths no longer than their axes",
                    ));
                }
            }
        }
        let volume = checked_product(window)?;
        let weights = match &kernel.weights {
            Some(weights) => {
                if weights.len() != volume {
                    return Err(Error::LengthMismatch);
                }
                if weights.iter().any(|&w| !(w.is_finite() && w >= 0.0)) {
                    return Err(Error::InvalidConfiguration(
                        "Savitzky-Golay weights must be finite and nonnegative",
                    ));
                }
                let mut copy = filled(weights.len(), 0.0)?;
                copy.copy_from_slice(weights);
                copy
            }
            None => filled(volume, 1.0)?,
        };
        let largest = weights.iter().copied().fold(0.0, f64::max);
        if largest == 0.0 {
            return Err(Error::InvalidConfiguration(
                "Savitzky-Golay weights need a positive entry",
            ));
        }
        let taps = weights.iter().filter(|&&w| w > 0.0).count();
        let (min_count, omit) = match policy {
            InvalidPolicy::Propagate => (1, false),
            InvalidPolicy::Omit {
                ignore_inf: _,
                min_count,
            } => (min_count, true),
        };

        let filtered: Vec<bool> = window.iter().map(|&length| length > 1).collect();
        let (bounds, total) = match &kernel.basis {
            PolynomialBasis::TotalDegree(degree) => {
                if window.iter().any(|&length| length > 1 && *degree >= length)
                    || (*degree > 0 && !filtered.contains(&true))
                {
                    return Err(Error::InvalidConfiguration(
                        "polynomial degree must be less than every window length above one",
                    ));
                }
                let bounds: Vec<usize> = filtered
                    .iter()
                    .map(|&used| if used { *degree } else { 0 })
                    .collect();
                if derivative.iter().sum::<usize>() > *degree {
                    return Err(Error::InvalidConfiguration(
                        "derivative orders exceed the polynomial degree",
                    ));
                }
                (bounds, Some(*degree))
            }
            PolynomialBasis::PerAxisDegree(degrees) => {
                if degrees.len() != rank {
                    return Err(Error::InvalidConfiguration(
                        "per-axis degrees need one entry per axis",
                    ));
                }
                if degrees.iter().zip(window).any(|(&p, &m)| p >= m) {
                    return Err(Error::InvalidConfiguration(
                        "each polynomial degree must be less than its window length",
                    ));
                }
                if derivative.iter().zip(degrees).any(|(&k, &p)| k > p) {
                    return Err(Error::InvalidConfiguration(
                        "derivative orders exceed the polynomial degree",
                    ));
                }
                (degrees.clone(), None)
            }
        };
        if derivative
            .iter()
            .zip(&filtered)
            .any(|(&order, &used)| order > 0 && !used)
        {
            return Err(Error::InvalidConfiguration(
                "derivatives require a window length above one",
            ));
        }
        let exponents = enumerate_exponents(&bounds, total, taps)?;
        let terms = exponents.len() / rank;
        if omit && (min_count == 0 || min_count > taps) {
            return Err(Error::InvalidConfiguration(
                "min_count must be between one and the number of positive weights",
            ));
        }

        let half: Vec<usize> = window.iter().map(|&length| length / 2).collect();
        let scale: Vec<f64> = half
            .iter()
            .map(|&h| if h > 0 { 1.0 / h as f64 } else { 1.0 })
            .collect();
        let derivative = derivative
            .iter()
            .map(|&order| u32::try_from(order).map_err(|_| Error::Overflow))
            .collect::<Result<Vec<u32>, Error>>()?;
        let mut factor = 1.0f64;
        for axis in 0..rank {
            let order = i32::try_from(derivative[axis]).map_err(|_| Error::Overflow)?;
            factor *= (scale[axis] / delta[axis]).powi(order);
        }
        if !(factor.is_finite() && factor > 0.0) {
            return Err(Error::InvalidConfiguration(
                "derivative scale is not representable in float64",
            ));
        }

        let window_strides = strides(window)?;
        let array_strides = strides(shape)?;
        let mut offsets = Vec::new();
        let mut flat_offsets = Vec::new();
        let mut root_weights = Vec::new();
        let mut design = Vec::new();
        offsets
            .try_reserve_exact(taps.checked_mul(rank).ok_or(Error::Overflow)?)
            .map_err(|_| Error::Memory)?;
        flat_offsets
            .try_reserve_exact(taps)
            .map_err(|_| Error::Memory)?;
        root_weights
            .try_reserve_exact(taps)
            .map_err(|_| Error::Memory)?;
        design
            .try_reserve_exact(taps.checked_mul(terms).ok_or(Error::Overflow)?)
            .map_err(|_| Error::Memory)?;
        let mut flat_ok = true;
        let mut scaled = vec![0.0; rank];
        for (index, &weight) in weights.iter().enumerate() {
            if weight == 0.0 {
                continue;
            }
            let root = (weight / largest).sqrt();
            root_weights.push(root);
            let mut remainder = index;
            let mut flat = 0i128;
            for axis in 0..rank {
                let coordinate = remainder / window_strides[axis];
                remainder %= window_strides[axis];
                let offset = coordinate as i128 - half[axis] as i128;
                offsets.push(offset);
                scaled[axis] = offset as f64 * scale[axis];
                flat = offset
                    .checked_mul(array_strides[axis] as i128)
                    .and_then(|step| flat.checked_add(step))
                    .ok_or(Error::Overflow)?;
            }
            match isize::try_from(flat) {
                Ok(value) => flat_offsets.push(value),
                Err(_) => flat_ok = false,
            }
            for term in exponents.chunks_exact(rank) {
                let value = term
                    .iter()
                    .zip(&scaled)
                    .fold(1.0, |product, (&power, &t)| product * t.powi(power as i32));
                design.push(root * value);
            }
        }
        if !flat_ok {
            flat_offsets.clear();
        }

        let mut solver = Solver::default();
        solver.reserve(taps, terms)?;
        for tap in 0..taps {
            for term in 0..terms {
                solver.matrix[term * taps + tap] = design[tap * terms + term];
            }
        }
        if !solver.factor() {
            return Err(Error::InvalidConfiguration(
                "the window weights leave the polynomial undetermined",
            ));
        }
        let interp = boundaries.contains(&FitBoundary::Interp);
        let mut projector = Vec::new();
        let mut unit = filled(terms, 0.0)?;
        if interp {
            projector
                .try_reserve_exact(taps.checked_mul(terms).ok_or(Error::Overflow)?)
                .map_err(|_| Error::Memory)?;
            projector.resize(taps * terms, 0.0);
            for term in 0..terms {
                unit.fill(0.0);
                unit[term] = 1.0;
                solver.project(&unit);
                for tap in 0..taps {
                    projector[tap * terms + term] = solver.projected[tap] * root_weights[tap];
                }
            }
        }
        let mut config = Self {
            shape: shape.to_vec(),
            strides: array_strides,
            half,
            boundaries,
            cval,
            policy,
            offsets,
            flat_offsets,
            root_weights,
            design,
            exponents,
            terms,
            derivative,
            scale,
            factor,
            center: Vec::new(),
            projector,
            dense: Vec::new(),
            output_len,
        };
        let zero = vec![0isize; rank];
        config.functional(&zero, &mut unit);
        solver.project(&unit);
        let mut center = filled(taps, 0.0)?;
        for ((value, projected), root) in center
            .iter_mut()
            .zip(&solver.projected)
            .zip(&config.root_weights)
        {
            *value = projected * root;
        }
        let mut dense = filled(volume, 0.0)?;
        let mut tap = 0;
        for (index, &weight) in weights.iter().enumerate() {
            if weight > 0.0 {
                dense[index] = center[tap];
                tap += 1;
            }
        }
        config.center = center;
        config.dense = dense;
        Ok(config)
    }

    /// Fill `output` with the derivative functional at signed offsets `at`.
    fn functional(&self, at: &[isize], output: &mut [f64]) {
        let rank = self.shape.len();
        for (term, value) in self.exponents.chunks_exact(rank).zip(output.iter_mut()) {
            let mut product = self.factor;
            for axis in 0..rank {
                product *= monomial_derivative(
                    term[axis],
                    self.derivative[axis],
                    at[axis] as f64 * self.scale[axis],
                );
            }
            *value = product;
        }
    }
}

/// Per-thread buffers reused across output positions.
struct Scratch {
    coordinates: Vec<usize>,
    centers: Vec<usize>,
    shift: Vec<isize>,
    values: Vec<f64>,
    coefficients: Vec<f64>,
    retained: Vec<usize>,
    functional: Vec<f64>,
    pairs: Vec<(f64, f64)>,
    solver: Solver,
}

impl Scratch {
    fn new<T>(config: &SavgolConfig<T>) -> Result<Self, Error> {
        let rank = config.shape.len();
        let taps = config.root_weights.len();
        let buffer = |length: usize| -> Result<Vec<f64>, Error> {
            let mut values = Vec::new();
            values
                .try_reserve_exact(length)
                .map_err(|_| Error::Memory)?;
            values.resize(length, 0.0);
            Ok(values)
        };
        let values = buffer(taps)?;
        let coefficients = buffer(taps)?;
        let functional = buffer(config.terms)?;
        let mut retained = Vec::new();
        retained
            .try_reserve_exact(taps)
            .map_err(|_| Error::Memory)?;
        let mut pairs = Vec::new();
        pairs.try_reserve_exact(taps).map_err(|_| Error::Memory)?;
        Ok(Self {
            coordinates: vec![0; rank],
            centers: vec![0; rank],
            shift: vec![0; rank],
            values,
            coefficients,
            retained,
            functional,
            pairs,
            solver: Solver::default(),
        })
    }
}

/// Sum `values * coefficients`. An infinite value gives an infinity with the
/// sign of value times coefficient; both signs, or a zero coefficient, give NaN.
fn weighted_sum(pairs: &mut Vec<(f64, f64)>, terms: impl Iterator<Item = (f64, f64)>) -> f64 {
    pairs.clear();
    let (mut positive, mut negative) = (false, false);
    for (value, coefficient) in terms {
        if value.is_infinite() {
            if coefficient == 0.0 {
                return f64::NAN;
            }
            if value.is_sign_negative() == coefficient.is_sign_negative() {
                positive = true;
            } else {
                negative = true;
            }
        } else if coefficient != 0.0 {
            // A finite sample times a zero coefficient adds nothing.
            pairs.push((value, coefficient));
        }
    }
    match (positive, negative) {
        (true, true) => f64::NAN,
        (true, false) => f64::INFINITY,
        (false, true) => f64::NEG_INFINITY,
        (false, false) => finite_weighted(pairs, false),
    }
}

fn evaluate<T: Float>(
    input: &[T],
    config: &SavgolConfig<T>,
    flat: usize,
    scratch: &mut Scratch,
) -> Result<T, Error> {
    let rank = config.shape.len();
    let taps = config.root_weights.len();
    let terms = config.terms;
    let mut remainder = flat;
    let mut interior = !config.flat_offsets.is_empty();
    let mut shifted = false;
    let mut center_flat = 0usize;
    for axis in 0..rank {
        let coordinate = remainder / config.strides[axis];
        remainder %= config.strides[axis];
        let (half, length) = (config.half[axis], config.shape[axis]);
        let center = match config.boundaries[axis] {
            // Validation guarantees 2 * half < length on interp axes.
            FitBoundary::Interp => coordinate.clamp(half, length - 1 - half),
            FitBoundary::Pad(_) => coordinate,
        };
        scratch.coordinates[axis] = coordinate;
        scratch.centers[axis] = center;
        scratch.shift[axis] = coordinate as isize - center as isize;
        shifted |= coordinate != center;
        interior &= center >= half && center + half < length;
        center_flat += center * config.strides[axis];
    }
    for tap in 0..taps {
        scratch.values[tap] = if interior {
            // Every axis keeps the window inside the array.
            input[center_flat.wrapping_add_signed(config.flat_offsets[tap])]
        } else {
            let mut source = 0usize;
            let mut constant = false;
            for axis in 0..rank {
                let coordinate = (scratch.centers[axis] as i128)
                    .checked_add(config.offsets[tap * rank + axis])
                    .ok_or(Error::Overflow)?;
                let mode = match config.boundaries[axis] {
                    FitBoundary::Pad(mode) => mode,
                    FitBoundary::Interp => BoundaryMode::Nearest,
                };
                match map_index(coordinate, config.shape[axis], mode)? {
                    Some(mapped) => source += mapped * config.strides[axis],
                    None => {
                        constant = true;
                        break;
                    }
                }
            }
            if constant {
                config.cval
            } else {
                input[source]
            }
        }
        .to_f64();
    }
    let (omit, ignore_inf, min_count) = match config.policy {
        InvalidPolicy::Propagate => (false, false, 1),
        InvalidPolicy::Omit {
            ignore_inf,
            min_count,
        } => (true, ignore_inf, min_count),
    };
    let missing = |value: f64| value.is_nan() || (ignore_inf && value.is_infinite());
    scratch.retained.clear();
    for (tap, &value) in scratch.values.iter().enumerate() {
        if !missing(value) {
            scratch.retained.push(tap);
        }
    }
    let Scratch {
        values,
        coefficients,
        retained,
        functional,
        pairs,
        solver,
        shift,
        ..
    } = scratch;
    let result = if retained.len() == taps {
        if shifted {
            config.functional(shift, functional);
            for (tap, coefficient) in coefficients.iter_mut().enumerate() {
                let row = &config.projector[tap * terms..(tap + 1) * terms];
                *coefficient = row.iter().zip(functional.iter()).map(|(m, g)| m * g).sum();
            }
            weighted_sum(
                pairs,
                values.iter().copied().zip(coefficients.iter().copied()),
            )
        } else {
            weighted_sum(
                pairs,
                values.iter().copied().zip(config.center.iter().copied()),
            )
        }
    } else if !omit || retained.len() < min_count || retained.len() < terms {
        f64::NAN
    } else {
        let rows = retained.len();
        solver.reserve(rows, terms)?;
        for (row, &tap) in retained.iter().enumerate() {
            for term in 0..terms {
                solver.matrix[term * rows + row] = config.design[tap * terms + term];
            }
        }
        if solver.factor() {
            config.functional(shift, functional);
            solver.project(functional);
            let projected = &solver.projected;
            weighted_sum(
                pairs,
                retained
                    .iter()
                    .enumerate()
                    .map(|(row, &tap)| (values[tap], projected[row] * config.root_weights[tap])),
            )
        } else {
            f64::NAN
        }
    };
    Ok(T::from_f64(result))
}

fn filter_range<T: Float>(
    input: &[T],
    config: &SavgolConfig<T>,
    start: usize,
    output: &mut [T],
) -> Result<(), Error> {
    let mut scratch = Scratch::new(config)?;
    for (local, destination) in output.iter_mut().enumerate() {
        let flat = start.checked_add(local).ok_or(Error::Overflow)?;
        *destination = evaluate(input, config, flat, &mut scratch)?;
    }
    Ok(())
}

pub(crate) fn savgol_into<T: Float + Send + Sync>(
    input: &[T],
    config: &SavgolConfig<T>,
    output: &mut [T],
) -> Result<(), Error> {
    if input.len() != config.output_len || output.len() != config.output_len {
        return Err(Error::LengthMismatch);
    }
    if output.is_empty() {
        return Ok(());
    }
    let threads = rayon::current_num_threads().max(1);
    let chunk_len = output.len().div_ceil(threads);
    output
        .par_chunks_mut(chunk_len)
        .enumerate()
        .try_for_each(|(chunk_index, chunk)| {
            let start = chunk_index.checked_mul(chunk_len).ok_or(Error::Overflow)?;
            filter_range(input, config, start, chunk)
        })
}

/// A validated Savitzky-Golay filter for one array shape.
///
/// Each output is a least-squares polynomial fit to the samples in its
/// window, evaluated (or differentiated) at the output position. With
/// [`InvalidPolicy::Propagate`], a NaN sample in a window gives NaN, and the
/// result is a fixed linear filter, as `scipy.signal.savgol_filter`. With
/// [`InvalidPolicy::Omit`], the fit uses only the retained samples. A window
/// with fewer than `min_count` retained samples, or whose retained samples
/// do not determine the polynomial, gives NaN.
///
/// The polynomial is undetermined when the singular values of the weighted
/// design matrix satisfy `sigma_min <= sigma_max * max(n, r) * eps`, with
/// `n` retained samples, `r` polynomial terms, and `eps = f64::EPSILON`.
/// This is the default tolerance of `numpy.linalg.matrix_rank`. The design
/// uses offsets divided by the window half-width on each axis.
///
/// # Examples
///
/// ```
/// use ndfiltrs::{InvalidPolicy, PolynomialBasis, SavgolKernel, SavitzkyGolay};
///
/// // A quadratic is reproduced exactly, even with a missing sample.
/// let line: Vec<f64> = (0..7).map(|i| (i * i) as f64).collect();
/// let mut damaged = line.clone();
/// damaged[3] = f64::NAN;
/// let skip_nan = InvalidPolicy::Omit { ignore_inf: false, min_count: 1 };
/// let plan = SavitzkyGolay::new(
///     &[7],
///     &SavgolKernel::new([5], PolynomialBasis::TotalDegree(2)),
///     skip_nan,
///     0.0,
/// )?;
/// let fitted = plan.apply(&damaged)?;
/// for (fit, exact) in fitted.iter().zip(&line) {
///     assert!((fit - exact).abs() < 1e-12);
/// }
/// # Ok::<(), ndfiltrs::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct SavitzkyGolay<T> {
    config: SavgolConfig<T>,
}

impl<T: FilterFloat> SavitzkyGolay<T> {
    /// Validate `kernel` and `policy` for inputs of shape `shape`.
    ///
    /// `cval` is the value outside the array for
    /// `FitBoundary::Pad(BoundaryMode::Constant)`. A NaN `cval` with
    /// [`InvalidPolicy::Omit`] leaves exterior positions out of the fit.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfiguration`] for a rank mismatch, an even
    /// window length, a degree not below its window length, a derivative
    /// outside the basis, a nonpositive or nonfinite `delta`, invalid
    /// weights, an interp window longer than its axis, a `min_count` outside
    /// one to the number of positive weights, or weights that leave the
    /// full-window fit undetermined. Returns [`Error::LengthMismatch`] when
    /// the weight count differs from the window volume.
    ///
    /// [`Error::InvalidConfiguration`]: crate::Error::InvalidConfiguration
    /// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
    pub fn new(
        shape: &[usize],
        kernel: &SavgolKernel,
        policy: InvalidPolicy,
        cval: T,
    ) -> Result<Self, Error> {
        Ok(Self {
            config: SavgolConfig::new(shape, kernel, policy, cval)?,
        })
    }

    /// Number of elements in each input and output buffer.
    pub fn len(&self) -> usize {
        self.config.output_len
    }

    /// `true` when the plan's shape has no elements.
    pub fn is_empty(&self) -> bool {
        self.config.output_len == 0
    }

    /// Full-window coefficients for the centered window, in C order with
    /// the window shape.
    ///
    /// Coefficient `k` multiplies the sample at offset `k - window / 2`, as
    /// correlation weights (`scipy.signal.savgol_coeffs(..., use="dot")` in
    /// one dimension). Positions with zero weight have coefficient zero.
    pub fn coefficients(&self) -> &[f64] {
        &self.config.dense
    }

    /// Filter `input` into a new buffer.
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

    /// Filter `input` into `output`. The buffers must not overlap.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] for a wrong buffer length.
    ///
    /// [`Error::LengthMismatch`]: crate::Error::LengthMismatch
    pub fn apply_into(&self, input: &[T], output: &mut [T]) -> Result<(), Error> {
        savgol_into(input, &self.config, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SKIP: InvalidPolicy = InvalidPolicy::Omit {
        ignore_inf: false,
        min_count: 1,
    };

    fn plan(shape: &[usize], kernel: SavgolKernel, policy: InvalidPolicy) -> SavitzkyGolay<f64> {
        SavitzkyGolay::new(shape, &kernel, policy, 0.0).unwrap()
    }

    fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() <= tolerance, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn classic_one_dimensional_coefficients() {
        let smooth = plan(
            &[9],
            SavgolKernel::new([5], PolynomialBasis::TotalDegree(2)),
            InvalidPolicy::Propagate,
        );
        let expected: Vec<f64> = [-3.0, 12.0, 17.0, 12.0, -3.0]
            .iter()
            .map(|v| v / 35.0)
            .collect();
        close(smooth.coefficients(), &expected, 1e-15);
        let slope = plan(
            &[9],
            SavgolKernel::new([5], PolynomialBasis::TotalDegree(2))
                .derivative([1])
                .delta([0.5]),
            InvalidPolicy::Propagate,
        );
        let expected: Vec<f64> = [-2.0, -1.0, 0.0, 1.0, 2.0]
            .iter()
            .map(|v| v / 10.0 / 0.5)
            .collect();
        close(slope.coefficients(), &expected, 1e-15);
    }

    #[test]
    fn masked_fits_reproduce_polynomials_with_interp_edges() {
        let shape = [6, 7];
        let truth = |y: f64, x: f64| 1.0 + 2.0 * x - 0.5 * y + 0.25 * x * y - 0.1 * y * y;
        let slope_x = |y: f64, _x: f64| 2.0 + 0.25 * y;
        let mut data = Vec::new();
        let mut value = Vec::new();
        let mut slope = Vec::new();
        for y in 0..6 {
            for x in 0..7 {
                let (y, x) = (y as f64, x as f64);
                data.push(truth(y, x));
                value.push(truth(y, x));
                slope.push(slope_x(y, x) / 2.0);
            }
        }
        for hole in [3, 10, 17, 24, 31, 38] {
            data[hole] = f64::NAN;
        }
        let kernel = SavgolKernel::new([5, 5], PolynomialBasis::TotalDegree(2));
        close(
            &plan(&shape, kernel.clone(), SKIP).apply(&data).unwrap(),
            &value,
            1e-10,
        );
        let kernel = kernel.derivative([0, 1]).delta([1.0, 2.0]);
        close(
            &plan(&shape, kernel, SKIP).apply(&data).unwrap(),
            &slope,
            1e-10,
        );
    }

    #[test]
    fn propagate_matches_omit_on_complete_data_and_propagates_nan() {
        let data: Vec<f64> = (0..11).map(|i| ((i * 7) % 5) as f64).collect();
        let kernel = SavgolKernel::new([5], PolynomialBasis::TotalDegree(2))
            .boundary(FitBoundary::Pad(BoundaryMode::Mirror));
        let dense = plan(&[11], kernel.clone(), InvalidPolicy::Propagate);
        let masked = plan(&[11], kernel, SKIP);
        assert_eq!(dense.apply(&data).unwrap(), masked.apply(&data).unwrap());
        let mut damaged = data;
        damaged[5] = f64::NAN;
        let result = dense.apply(&damaged).unwrap();
        for (index, value) in result.iter().enumerate() {
            assert_eq!(value.is_nan(), (3..=7).contains(&index), "{index}");
        }
    }

    #[test]
    fn undetermined_windows_and_min_count_give_nan() {
        // Only one row of a 3x3 window survives: x * y terms are undetermined.
        let mut data = vec![f64::NAN; 9];
        data[3..6].copy_from_slice(&[1.0, 2.0, 3.0]);
        let kernel = SavgolKernel::new([3, 3], PolynomialBasis::TotalDegree(1))
            .boundary(FitBoundary::Pad(BoundaryMode::Constant));
        let nan_cval = SavitzkyGolay::new(&[3, 3], &kernel, SKIP, f64::NAN).unwrap();
        assert!(nan_cval.apply(&data).unwrap()[4].is_nan());
        data[1] = 0.0;
        assert_eq!(nan_cval.apply(&data).unwrap()[4], 2.0);
        let strict = InvalidPolicy::Omit {
            ignore_inf: false,
            min_count: 5,
        };
        let strict = SavitzkyGolay::new(&[3, 3], &kernel, strict, f64::NAN).unwrap();
        assert!(strict.apply(&data).unwrap()[4].is_nan());
    }

    #[test]
    fn invalid_configurations_are_errors() {
        let new = |kernel: SavgolKernel, shape: &[usize]| {
            SavitzkyGolay::<f64>::new(shape, &kernel, SKIP, 0.0).err()
        };
        let basis = PolynomialBasis::TotalDegree(2);
        for (kernel, shape) in [
            (SavgolKernel::new([4], basis.clone()), &[9][..]),
            (
                SavgolKernel::new([3], PolynomialBasis::TotalDegree(3)),
                &[9],
            ),
            (SavgolKernel::new([5], basis.clone()), &[4]),
            (SavgolKernel::new([5], basis.clone()).derivative([3]), &[9]),
            (SavgolKernel::new([5], basis.clone()).delta([0.0]), &[9]),
            (
                SavgolKernel::new([5], basis.clone()).weights([1.0, -1.0, 1.0, 1.0, 1.0]),
                &[9],
            ),
            (
                SavgolKernel::new([5], basis.clone()).weights([0.0, 0.0, 1.0, 0.0, 1.0]),
                &[9],
            ),
            (
                SavgolKernel::new([5, 1], basis.clone()).derivative([0, 1]),
                &[9, 3],
            ),
            (
                SavgolKernel::new([5, 5], PolynomialBasis::PerAxisDegree(vec![2])),
                &[9, 9],
            ),
        ] {
            assert!(matches!(
                new(kernel, shape),
                Some(Error::InvalidConfiguration(_))
            ));
        }
        assert_eq!(
            new(
                SavgolKernel::new([5], basis.clone()).weights([1.0; 3]),
                &[9]
            ),
            Some(Error::LengthMismatch)
        );
        let tight = InvalidPolicy::Omit {
            ignore_inf: false,
            min_count: 6,
        };
        assert!(
            SavitzkyGolay::<f64>::new(&[9], &SavgolKernel::new([5], basis), tight, 0.0).is_err()
        );
    }

    #[test]
    fn empty_shapes_and_lengths() {
        let empty = plan(
            &[0, 4],
            SavgolKernel::new([3, 3], PolynomialBasis::TotalDegree(1)),
            SKIP,
        );
        assert!(empty.is_empty());
        assert_eq!(empty.apply(&[]).unwrap(), Vec::<f64>::new());
        let line = plan(
            &[5],
            SavgolKernel::new([3], PolynomialBasis::TotalDegree(1)),
            SKIP,
        );
        assert_eq!(line.apply(&[1.0; 4]).err(), Some(Error::LengthMismatch));
    }
}
