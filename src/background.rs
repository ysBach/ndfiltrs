//! Mesh (grid) background and RMS estimation.
//!
//! The array is divided into boxes ("meshes"). Each mesh gets a robust
//! background and RMS from its valid samples. Meshes with too few samples
//! are replaced from valid neighbors, the grid is median filtered, and a
//! spline maps it back to full resolution. See `docs/background.md`.

mod sep_interpolation;

use rayon::prelude::*;

use crate::error::Error;
use crate::location::{mode_from_moments, ModeEstimator};
use crate::mean::{mean_valid, Sum};
use crate::plan::FilterFloat;
use crate::robust::{
    validate_pearson, ClipCenter, ClipConfig, ClipSpread, ClipStatistic, NORMAL_UPPER_QUARTILE,
};

/// SExtractor clips once at the mean +- 2 standard deviations ...
const SEXTRACTOR_FIRST_SIGMA: f64 = 2.0;
/// ... then histograms the samples within +- 5 of those deviations ...
const SEXTRACTOR_SPAN_SIGMA: f64 = 5.0;
/// ... and clips at the median +- 3 deviations for at most 100 passes.
const SEXTRACTOR_CLIP_SIGMA: f64 = 3.0;
const SEXTRACTOR_MAXITERS: usize = 100;

/// Statistics computed from the valid samples of one mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum MeshEstimator {
    /// SExtractor's sequence (`src/back.c`) on exact samples instead of a
    /// histogram: keep samples within the mean +- 2 standard deviations;
    /// from all valid samples keep those within +- 5 deviations of that
    /// subset's mean; clip at the median +- 3 deviations until stable (at
    /// most 100 passes). The background is the [`ModeEstimator::Sex`] mode
    /// and the RMS is the population standard deviation of the survivors.
    SExtractor {
        /// Pearson factor (`>= 1`), SExtractor's `BACK_PEARSON` (`2.5`).
        pearson: f64,
    },
    /// One iterative clip as in [`ClipConfig`] (Astropy `SigmaClip`
    /// semantics), then the clip's statistic as background and `rms` as RMS.
    /// A mesh is invalid when fewer than `clip.min_count` samples survive.
    SigmaClip {
        /// Clipping and background statistic.
        clip: ClipConfig,
        /// RMS statistic of the survivors.
        rms: MeshRms,
    },
}

/// RMS statistic for [`MeshEstimator::SigmaClip`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MeshRms {
    /// Population standard deviation (`ddof = 0`).
    Std,
    /// Median absolute deviation divided by the standard normal 75th
    /// percentile (about `0.67449`), as `astropy.stats.mad_std`.
    MadStd,
}

/// Sample count compared with the minimum valid fraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MeshValidCount {
    /// Finite, unmasked samples before clipping (SExtractor).
    Finite,
    /// Samples that survive clipping (photutils).
    Retained,
}

/// Sample count that the minimum valid fraction multiplies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MeshValidBase {
    /// Pixels in the mesh, smaller for partial edge meshes (SExtractor).
    Mesh,
    /// Pixels in a full box, also for partial edge meshes (photutils).
    Box,
}

/// Replacement for invalid meshes, from valid meshes only.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum MeshFill {
    /// Mean of all valid meshes at the smallest Euclidean distance in mesh
    /// units (SExtractor).
    Nearest,
    /// Inverse-distance weights `1 / d^power` over the `neighbors` nearest
    /// valid meshes (photutils uses 10 and 1). All meshes at the same
    /// distance as the last selected one are also included.
    InverseDistance {
        /// Number of neighbors (`>= 1`).
        neighbors: usize,
        /// Finite, nonnegative distance exponent.
        power: f64,
    },
}

/// Window of the mesh median filter at the grid edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MeshFilterEdge {
    /// Reduce the half width on each axis to the distance from the nearer
    /// edge, so the window stays centered (SExtractor).
    Shrink,
    /// Drop window positions outside the grid (photutils).
    Truncate,
}

/// Meshes replaced by the median filter.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum MeshThreshold {
    /// Replace every mesh.
    All,
    /// Replace a mesh when `|median - background| >= value` (SExtractor
    /// `BACK_FILTTHRESH`, sep `fthresh`).
    Deviation(f64),
    /// Replace a mesh when its background is `> value` (photutils
    /// `filter_threshold`).
    Above(f64),
}

/// Map from the mesh grid to full resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MeshInterpolation {
    /// Natural cubic spline along each axis (tensor product). Nodes are at
    /// mesh centers `(k + 0.5) * box - 0.5`, also for partial edge meshes;
    /// outside the end nodes the end cubic continues.
    Spline,
    /// Two-dimensional interpolation compatible with sep background maps.
    SExtractor,
}

/// Rules for every stage of a [`MeshBackground`].
///
/// [`MeshConfig::sextractor`] and [`MeshConfig::photutils`] give the
/// reference rule sets; the setters change one stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshConfig {
    estimator: MeshEstimator,
    min_valid_fraction: f64,
    valid_count: MeshValidCount,
    valid_base: MeshValidBase,
    fill: MeshFill,
    filter_edge: MeshFilterEdge,
    threshold: MeshThreshold,
    interpolation: MeshInterpolation,
}

impl MeshConfig {
    /// SExtractor rules: [`MeshEstimator::SExtractor`] with Pearson `2.5`,
    /// valid when at least half of the mesh pixels are finite and unmasked,
    /// nearest fill, shrinking filter window, `Deviation(0.0)` (replace
    /// all), and the [`MeshInterpolation::Spline`] interpolation.
    pub fn sextractor() -> Self {
        Self {
            estimator: MeshEstimator::SExtractor { pearson: 2.5 },
            min_valid_fraction: 0.5,
            valid_count: MeshValidCount::Finite,
            valid_base: MeshValidBase::Mesh,
            fill: MeshFill::Nearest,
            filter_edge: MeshFilterEdge::Shrink,
            threshold: MeshThreshold::Deviation(0.0),
            interpolation: MeshInterpolation::Spline,
        }
    }

    /// photutils `Background2D` defaults: 3-sigma median/std clip with at
    /// most 10 passes, SExtractor mode and standard deviation, valid when
    /// at least 90% of a full box survives clipping, 10-neighbor inverse
    /// distance fill, truncated filter window, all meshes filtered, and the
    /// [`MeshInterpolation::Spline`] interpolation (photutils itself uses
    /// `scipy.ndimage.zoom`).
    pub fn photutils() -> Self {
        let clip = ClipConfig::new(3.0)
            .maxiters(Some(10))
            .center(ClipCenter::Median)
            .spread(ClipSpread::Std)
            .statistic(ClipStatistic::Mode(ModeEstimator::Sex, 2.5));
        Self {
            estimator: MeshEstimator::SigmaClip {
                clip,
                rms: MeshRms::Std,
            },
            min_valid_fraction: 0.9,
            valid_count: MeshValidCount::Retained,
            valid_base: MeshValidBase::Box,
            fill: MeshFill::InverseDistance {
                neighbors: 10,
                power: 1.0,
            },
            filter_edge: MeshFilterEdge::Truncate,
            threshold: MeshThreshold::All,
            interpolation: MeshInterpolation::Spline,
        }
    }

    /// Set the per-mesh estimator.
    pub fn estimator(mut self, estimator: MeshEstimator) -> Self {
        self.estimator = estimator;
        self
    }

    /// Set the validity rule: a mesh is valid when `count >= fraction *
    /// base` and `count > 0`.
    pub fn min_valid(mut self, fraction: f64, count: MeshValidCount, base: MeshValidBase) -> Self {
        self.min_valid_fraction = fraction;
        self.valid_count = count;
        self.valid_base = base;
        self
    }

    /// Set the replacement for invalid meshes.
    pub fn fill(mut self, fill: MeshFill) -> Self {
        self.fill = fill;
        self
    }

    /// Set the median filter's edge rule and replacement threshold.
    pub fn filter(mut self, edge: MeshFilterEdge, threshold: MeshThreshold) -> Self {
        self.filter_edge = edge;
        self.threshold = threshold;
        self
    }

    /// Set the full-resolution interpolation.
    pub fn interpolation(mut self, interpolation: MeshInterpolation) -> Self {
        self.interpolation = interpolation;
        self
    }

    fn validate(&self, ndim: usize) -> Result<(), Error> {
        match self.estimator {
            MeshEstimator::SExtractor { pearson } => validate_pearson(pearson)?,
            MeshEstimator::SigmaClip { clip, .. } => {
                if !clip.lower.is_finite()
                    || !clip.upper.is_finite()
                    || clip.lower < 0.0
                    || clip.upper < 0.0
                {
                    return Err(Error::InvalidConfiguration(
                        "clipping limits must be finite and nonnegative",
                    ));
                }
                if let ClipStatistic::Mode(_, pearson) = clip.statistic {
                    validate_pearson(pearson)?;
                }
                if clip.min_count == 0 {
                    return Err(Error::InvalidConfiguration(
                        "min_count must be at least one",
                    ));
                }
            }
        }
        if !(0.0..=1.0).contains(&self.min_valid_fraction) {
            return Err(Error::InvalidConfiguration(
                "minimum valid fraction must be between zero and one",
            ));
        }
        if let MeshFill::InverseDistance { neighbors, power } = self.fill {
            if neighbors == 0 || !power.is_finite() || power < 0.0 {
                return Err(Error::InvalidConfiguration(
                    "inverse-distance fill needs at least one neighbor and a finite nonnegative power",
                ));
            }
        }
        if let MeshThreshold::Deviation(value) | MeshThreshold::Above(value) = self.threshold {
            if !value.is_finite() {
                return Err(Error::InvalidConfiguration(
                    "filter threshold must be finite",
                ));
            }
        }
        if self.interpolation == MeshInterpolation::SExtractor && ndim != 2 {
            return Err(Error::InvalidConfiguration(
                "SExtractor interpolation needs a 2-D shape",
            ));
        }
        Ok(())
    }
}

/// Background and RMS grids of one array, one value per mesh.
///
/// Grids are `f64` in C order with shape [`MeshGrid::shape`].
#[derive(Clone, Debug, PartialEq)]
pub struct MeshGrid {
    shape: Vec<usize>,
    background: Vec<f64>,
    rms: Vec<f64>,
    valid: Vec<bool>,
    finite_count: Vec<usize>,
    retained_count: Vec<usize>,
    global_background: f64,
    global_rms: f64,
}

impl MeshGrid {
    /// Number of meshes along each axis.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Background per mesh after fill and median filter. All NaN when no
    /// mesh is valid.
    pub fn background(&self) -> &[f64] {
        &self.background
    }

    /// RMS per mesh after fill and median filter.
    pub fn rms(&self) -> &[f64] {
        &self.rms
    }

    /// `true` for meshes that passed the validity rule (before fill).
    pub fn valid(&self) -> &[bool] {
        &self.valid
    }

    /// Finite, unmasked samples per mesh.
    pub fn finite_count(&self) -> &[usize] {
        &self.finite_count
    }

    /// Samples used by the final statistics per mesh.
    pub fn retained_count(&self) -> &[usize] {
        &self.retained_count
    }

    /// Median of [`MeshGrid::background`].
    pub fn global_background(&self) -> f64 {
        self.global_background
    }

    /// Median of [`MeshGrid::rms`].
    pub fn global_rms(&self) -> f64 {
        self.global_rms
    }
}

/// Interpolation weights of one output index along one axis:
/// `value = a y[seg] + b y[seg + 1] + c m[seg] + d m[seg + 1]`.
#[derive(Clone, Copy, Debug)]
struct Tap {
    seg: usize,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
}

impl Tap {
    /// Evaluate a natural cubic spline at node coordinate `u`.
    fn new(u: f64, nodes: usize) -> Self {
        if nodes < 2 {
            return Self {
                seg: 0,
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
            };
        }
        let seg = (u.floor().max(0.0) as usize).min(nodes - 2);
        let t = u - seg as f64;
        let a = 1.0 - t;
        Self {
            seg,
            a,
            b: t,
            c: (a * a * a - a) / 6.0,
            d: (t * t * t - t) / 6.0,
        }
    }

    fn eval(&self, y: &[f64], m: &[f64]) -> f64 {
        if y.len() < 2 {
            return y[0];
        }
        let s = self.seg;
        self.a * y[s] + self.b * y[s + 1] + self.c * m[s] + self.d * m[s + 1]
    }
}

/// A validated mesh background estimator for one array shape.
///
/// Mesh `k` along an axis covers indices `k * box .. min((k + 1) * box,
/// n)`; the last mesh is partial when `box` does not divide `n`. NaN,
/// infinite, and masked samples are never used. Statistics are computed in
/// `f64`; full-resolution maps are cast once to `T`.
///
/// # Examples
///
/// ```
/// use ndfiltrs::{MeshBackground, MeshConfig};
///
/// // A sloped 64x96 image with one bright source; 16x16 meshes.
/// let (ny, nx) = (64, 96);
/// let mut image: Vec<f64> = (0..ny * nx)
///     .map(|i| 10.0 + 0.01 * (i / nx) as f64 + ((i * 7919) % 13) as f64 * 0.1)
///     .collect();
/// image[20 * nx + 30] = 1e4;
/// let plan = MeshBackground::new(&[ny, nx], &[16, 16], &[3, 3], MeshConfig::sextractor())?;
/// let grid = plan.estimate(&image, None)?;
/// assert_eq!(grid.shape(), &[4, 6]);
/// let mut background = vec![0.0_f64; ny * nx];
/// plan.interpolate_into(grid.background(), &mut background)?;
/// assert!((background[20 * nx + 30] - 10.8).abs() < 0.5);
/// # Ok::<(), ndfiltrs::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct MeshBackground {
    shape: Vec<usize>,
    box_size: Vec<usize>,
    filter_size: Vec<usize>,
    mesh_shape: Vec<usize>,
    config: MeshConfig,
    len: usize,
    mesh_len: usize,
    box_volume: usize,
    /// Per axis: one tap per output index.
    taps: Vec<Vec<Tap>>,
    /// Per axis: reciprocal pivots of the natural-spline tridiagonal solve.
    pivots: Vec<Vec<f64>>,
}

impl MeshBackground {
    /// Validate the geometry and rules for inputs of shape `shape`.
    ///
    /// `box_size` and `filter_size` have one entry per axis. Box sizes are
    /// positive and may exceed the axis length (one mesh). Filter sizes
    /// are odd and positive; `1` on every axis disables the filter.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfiguration`] for a rank-0 shape, wrong
    /// entry counts, a zero box size, an even or zero filter size, or an
    /// invalid rule parameter. Returns [`Error::Overflow`] when the element
    /// count, mesh count, or box volume overflows `usize`.
    pub fn new(
        shape: &[usize],
        box_size: &[usize],
        filter_size: &[usize],
        config: MeshConfig,
    ) -> Result<Self, Error> {
        let ndim = shape.len();
        if ndim == 0 {
            return Err(Error::InvalidConfiguration(
                "shape must have at least one axis",
            ));
        }
        if box_size.len() != ndim || filter_size.len() != ndim {
            return Err(Error::InvalidConfiguration(
                "box and filter sizes need one entry per axis",
            ));
        }
        if box_size.contains(&0) {
            return Err(Error::InvalidConfiguration("box sizes must be positive"));
        }
        if filter_size.iter().any(|&size| size % 2 == 0) {
            return Err(Error::InvalidConfiguration(
                "filter sizes must be odd and positive",
            ));
        }
        config.validate(ndim)?;
        let len = checked_product(shape)?;
        let mesh_shape: Vec<usize> = shape
            .iter()
            .zip(box_size)
            .map(|(&n, &b)| n.div_ceil(b))
            .collect();
        let mesh_len = checked_product(&mesh_shape)?;
        let box_volume = checked_product(box_size)?;
        let mut taps = Vec::with_capacity(ndim);
        let mut pivots = Vec::with_capacity(ndim);
        if config.interpolation == MeshInterpolation::Spline {
            for axis in 0..ndim {
                let (n, b, m) = (shape[axis], box_size[axis] as f64, mesh_shape[axis]);
                let mut axis_taps = Vec::new();
                axis_taps.try_reserve_exact(n).map_err(|_| Error::Memory)?;
                for p in 0..n {
                    let p = p as f64;
                    axis_taps.push(Tap::new((p + 0.5) / b - 0.5, m));
                }
                taps.push(axis_taps);
                pivots.push(natural_pivots(m));
            }
        }
        Ok(Self {
            shape: shape.to_vec(),
            box_size: box_size.to_vec(),
            filter_size: filter_size.to_vec(),
            mesh_shape,
            config,
            len,
            mesh_len,
            box_volume,
            taps,
            pivots,
        })
    }

    /// Input shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of meshes along each axis.
    pub fn mesh_shape(&self) -> &[usize] {
        &self.mesh_shape
    }

    /// Number of elements in each input and full-resolution output.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` when the input shape has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Estimate the mesh grids of `data`.
    ///
    /// `mask` entries that are `true` exclude the sample.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] when `data` or `mask` has the
    /// wrong length.
    pub fn estimate<T: FilterFloat>(
        &self,
        data: &[T],
        mask: Option<&[bool]>,
    ) -> Result<MeshGrid, Error> {
        if data.len() != self.len || mask.is_some_and(|mask| mask.len() != self.len) {
            return Err(Error::LengthMismatch);
        }
        let stats: Vec<MeshStats> = (0..self.mesh_len)
            .into_par_iter()
            .map_init(
                || (Vec::new(), Vec::new(), Vec::new()),
                |(values, scratch, keys), mesh| {
                    self.mesh_stats(data, mask, mesh, values, scratch, keys)
                },
            )
            .collect();
        let valid: Vec<bool> = stats.iter().map(|stat| stat.valid).collect();
        let mut raw_background: Vec<f64> = stats.iter().map(|stat| stat.background).collect();
        let mut raw_rms: Vec<f64> = stats.iter().map(|stat| stat.rms).collect();
        self.fill_invalid(&mut raw_background, &mut raw_rms, &valid);
        let (background, rms) = self.filter(raw_background, raw_rms);
        let global_background = median_of(&mut background.clone());
        let global_rms = median_of(&mut rms.clone());
        Ok(MeshGrid {
            shape: self.mesh_shape.clone(),
            background,
            rms,
            valid,
            finite_count: stats.iter().map(|stat| stat.finite).collect(),
            retained_count: stats.iter().map(|stat| stat.retained).collect(),
            global_background,
            global_rms,
        })
    }

    /// Interpolate a mesh-shaped `f64` grid (for example
    /// [`MeshGrid::background`] or [`MeshGrid::rms`]) to full resolution.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] when `mesh` or `output` has the
    /// wrong length and [`Error::Memory`] when a workspace allocation fails.
    pub fn interpolate_into<T: FilterFloat>(
        &self,
        mesh: &[f64],
        output: &mut [T],
    ) -> Result<(), Error> {
        if mesh.len() != self.mesh_len || output.len() != self.len {
            return Err(Error::LengthMismatch);
        }
        if self.len == 0 {
            return Ok(());
        }
        // Both interpolators preserve constants. Filling directly also avoids
        // overflow in intermediate cubic coefficients near the dtype limit.
        let first = mesh[0];
        if mesh.iter().all(|&value| value == first) {
            let constant = T::from_f64(first);
            output.par_iter_mut().for_each(|pixel| *pixel = constant);
            return Ok(());
        }
        match self.config.interpolation {
            MeshInterpolation::Spline => self.spline_into(mesh, output),
            MeshInterpolation::SExtractor => {
                sep_interpolation::interpolate(
                    mesh,
                    [self.shape[0], self.shape[1]],
                    [self.box_size[0], self.box_size[1]],
                    output,
                );
                Ok(())
            }
        }
    }

    fn mesh_stats<T: FilterFloat>(
        &self,
        data: &[T],
        mask: Option<&[bool]>,
        mesh: usize,
        values: &mut Vec<f64>,
        scratch: &mut Vec<f64>,
        keys: &mut Vec<u64>,
    ) -> MeshStats {
        let ndim = self.shape.len();
        // Mesh bounds along each axis.
        let mut start = vec![0usize; ndim];
        let mut stop = vec![0usize; ndim];
        let mut rest = mesh;
        for axis in (0..ndim).rev() {
            let k = rest % self.mesh_shape[axis];
            rest /= self.mesh_shape[axis];
            start[axis] = k * self.box_size[axis];
            stop[axis] = (start[axis] + self.box_size[axis]).min(self.shape[axis]);
        }
        let volume: usize = start.iter().zip(&stop).map(|(a, b)| b - a).product();
        values.clear();
        // Copy contiguous runs along the last axis.
        let mut index = start.clone();
        loop {
            let offset = index
                .iter()
                .zip(&self.shape)
                .fold(0, |offset, (&k, &n)| offset * n + k);
            let run = stop[ndim - 1] - start[ndim - 1];
            for i in offset..offset + run {
                let value = data[i].to_f64();
                if value.is_finite() && !mask.is_some_and(|mask| mask[i]) {
                    values.push(value);
                }
            }
            // Advance the odometer over all but the last axis.
            let mut axis = ndim - 1;
            loop {
                if axis == 0 {
                    return self.finish_stats(values, scratch, keys, volume);
                }
                axis -= 1;
                index[axis] += 1;
                if index[axis] < stop[axis] {
                    break;
                }
                index[axis] = start[axis];
            }
        }
    }

    fn finish_stats(
        &self,
        values: &mut [f64],
        scratch: &mut Vec<f64>,
        keys: &mut Vec<u64>,
        volume: usize,
    ) -> MeshStats {
        let finite = values.len();
        let mut stats = MeshStats {
            background: f64::NAN,
            rms: f64::NAN,
            finite,
            retained: 0,
            valid: false,
        };
        if finite == 0 {
            return stats;
        }
        sort_finite(values, keys);
        let (background, rms, retained) = match self.config.estimator {
            MeshEstimator::SExtractor { pearson } => sextractor_stats(values, pearson),
            MeshEstimator::SigmaClip { clip, rms } => clipped_stats(values, scratch, clip, rms),
        };
        stats.retained = retained;
        let count = match self.config.valid_count {
            MeshValidCount::Finite => finite,
            MeshValidCount::Retained => retained,
        };
        let base = match self.config.valid_base {
            MeshValidBase::Mesh => volume,
            MeshValidBase::Box => self.box_volume,
        };
        if count > 0
            && count as f64 >= self.config.min_valid_fraction * base as f64
            && background.is_finite()
            && rms.is_finite()
        {
            stats.background = background;
            stats.rms = rms;
            stats.valid = true;
        }
        stats
    }

    /// Multi-index of a mesh.
    fn mesh_index(&self, mut flat: usize, index: &mut [usize]) {
        for axis in (0..index.len()).rev() {
            index[axis] = flat % self.mesh_shape[axis];
            flat /= self.mesh_shape[axis];
        }
    }

    fn fill_invalid(&self, background: &mut [f64], rms: &mut [f64], valid: &[bool]) {
        if valid.iter().all(|&v| v) {
            return;
        }
        let good: Vec<usize> = (0..self.mesh_len).filter(|&i| valid[i]).collect();
        if good.is_empty() {
            // Invalid meshes already contain NaNs.
            return;
        }
        let ndim = self.mesh_shape.len();
        let mut good_index = vec![0usize; good.len() * ndim];
        for (slot, &flat) in good.iter().enumerate() {
            self.mesh_index(flat, &mut good_index[slot * ndim..(slot + 1) * ndim]);
        }
        let replacements: Vec<(usize, f64, f64)> = (0..self.mesh_len)
            .into_par_iter()
            .filter(|&i| !valid[i])
            .map_init(
                || (vec![0usize; ndim], Vec::new()),
                |(index, distances), flat| {
                    self.mesh_index(flat, index);
                    distances.clear();
                    for (slot, &source) in good.iter().enumerate() {
                        let other = &good_index[slot * ndim..(slot + 1) * ndim];
                        let d2: u64 = index
                            .iter()
                            .zip(other)
                            .map(|(&a, &b)| (a.abs_diff(b) as u64).pow(2))
                            .sum();
                        distances.push((d2, source));
                    }
                    let (value, spread) = self.replacement(distances, background, rms);
                    (flat, value, spread)
                },
            )
            .collect();
        for (flat, value, spread) in replacements {
            background[flat] = value;
            rms[flat] = spread;
        }
    }

    fn replacement(
        &self,
        distances: &mut [(u64, usize)],
        background: &[f64],
        rms: &[f64],
    ) -> (f64, f64) {
        match self.config.fill {
            MeshFill::Nearest => {
                let nearest = distances.iter().map(|&(d2, _)| d2).min().unwrap_or(0);
                let mut value = Sum::default();
                let mut spread = Sum::default();
                let mut count = 0usize;
                for &(_, source) in distances.iter().filter(|&&(d2, _)| d2 == nearest) {
                    value.add(background[source]);
                    spread.add(rms[source]);
                    count += 1;
                }
                (value.total() / count as f64, spread.total() / count as f64)
            }
            MeshFill::InverseDistance { neighbors, power } => {
                distances.sort_unstable();
                let last = distances[neighbors.min(distances.len()) - 1].0;
                let mut weighted = Sum::default();
                let mut weighted_rms = Sum::default();
                let mut total = Sum::default();
                for &(d2, source) in distances.iter().take_while(|&&(d2, _)| d2 <= last) {
                    let weight = 1.0 / (d2 as f64).sqrt().powf(power);
                    weighted.add(weight * background[source]);
                    weighted_rms.add(weight * rms[source]);
                    total.add(weight);
                }
                (
                    weighted.total() / total.total(),
                    weighted_rms.total() / total.total(),
                )
            }
        }
    }

    fn filter(&self, background: Vec<f64>, rms: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        if self.filter_size.iter().all(|&size| size == 1) || self.mesh_len == 0 {
            return (background, rms);
        }
        let mut out_background = background.clone();
        let mut out_rms = rms.clone();
        let ndim = self.mesh_shape.len();
        let mut index = vec![0usize; ndim];
        let mut low = vec![0usize; ndim];
        let mut high = vec![0usize; ndim];
        let mut window_background = Vec::new();
        let mut window_rms = Vec::new();
        for flat in 0..self.mesh_len {
            self.mesh_index(flat, &mut index);
            for axis in 0..ndim {
                let (k, n) = (index[axis], self.mesh_shape[axis]);
                let half = self.filter_size[axis] / 2;
                match self.config.filter_edge {
                    MeshFilterEdge::Shrink => {
                        let half = half.min(k).min(n - 1 - k);
                        low[axis] = k - half;
                        high[axis] = k + half;
                    }
                    MeshFilterEdge::Truncate => {
                        low[axis] = k.saturating_sub(half);
                        high[axis] = (k + half).min(n - 1);
                    }
                }
            }
            window_background.clear();
            window_rms.clear();
            let volume: usize = (0..ndim).map(|axis| high[axis] - low[axis] + 1).product();
            for position in 0..volume {
                let mut rest = position;
                let mut offset = 0;
                let mut stride = 1;
                for axis in (0..ndim).rev() {
                    let width = high[axis] - low[axis] + 1;
                    offset += (low[axis] + rest % width) * stride;
                    rest /= width;
                    stride *= self.mesh_shape[axis];
                }
                window_background.push(background[offset]);
                window_rms.push(rms[offset]);
            }
            let median = median_of(&mut window_background);
            let replace = match self.config.threshold {
                MeshThreshold::All => true,
                MeshThreshold::Deviation(value) => (median - background[flat]).abs() >= value,
                MeshThreshold::Above(value) => background[flat] > value,
            };
            if replace {
                out_background[flat] = median;
                out_rms[flat] = median_of(&mut window_rms);
            }
        }
        (out_background, out_rms)
    }

    /// Natural-spline tensor product, one axis per pass.
    fn spline_into<T: FilterFloat>(&self, mesh: &[f64], output: &mut [T]) -> Result<(), Error> {
        let ndim = self.shape.len();
        let mut current = mesh.to_vec();
        for axis in 0..ndim {
            let outer: usize = self.shape[..axis].iter().product();
            let nodes = self.mesh_shape[axis];
            let inner: usize = self.mesh_shape[axis + 1..].iter().product();
            let taps = &self.taps[axis];
            let pivots = &self.pivots[axis];
            if axis + 1 == ndim {
                output
                    .par_chunks_mut(self.shape[axis])
                    .enumerate()
                    .for_each_init(Vec::new, |second, (line, out)| {
                        let y = &current[line * nodes..(line + 1) * nodes];
                        natural_second_derivatives(y, pivots, second);
                        for (value, tap) in out.iter_mut().zip(taps) {
                            *value = T::from_f64(tap.eval(y, second));
                        }
                    });
                return Ok(());
            }
            let block = nodes * inner;
            let mut second = Vec::new();
            second
                .try_reserve_exact(current.len())
                .map_err(|_| Error::Memory)?;
            second.resize(current.len(), 0.0);
            for o in 0..outer {
                columns_second_derivatives(
                    &current[o * block..(o + 1) * block],
                    nodes,
                    inner,
                    pivots,
                    &mut second[o * block..(o + 1) * block],
                );
            }
            let length = self.shape[axis];
            let mut next = Vec::new();
            next.try_reserve_exact(outer * length * inner)
                .map_err(|_| Error::Memory)?;
            next.resize(outer * length * inner, 0.0);
            next.par_chunks_mut(inner)
                .enumerate()
                .for_each(|(row, out)| {
                    let (o, p) = (row / length, row % length);
                    let tap = taps[p];
                    let base = o * block;
                    if nodes < 2 {
                        out.copy_from_slice(&current[base..base + inner]);
                        return;
                    }
                    let lo = base + tap.seg * inner;
                    let hi = lo + inner;
                    for j in 0..inner {
                        out[j] = tap.a * current[lo + j]
                            + tap.b * current[hi + j]
                            + tap.c * second[lo + j]
                            + tap.d * second[hi + j];
                    }
                });
            current = next;
        }
        Ok(())
    }
}

struct MeshStats {
    background: f64,
    rms: f64,
    finite: usize,
    retained: usize,
    valid: bool,
}

fn checked_product(values: &[usize]) -> Result<usize, Error> {
    values
        .iter()
        .try_fold(1usize, |total, &value| total.checked_mul(value))
        .ok_or(Error::Overflow)
}

/// Median of finite values; NaN for none. Reorders `values`.
fn median_of(values: &mut [f64]) -> f64 {
    if values.is_empty() || values.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    values.sort_unstable_by(f64::total_cmp);
    median_sorted(values)
}

/// Median of sorted finite values: the mean of the two middle values for an
/// even count, as `numpy.median`.
fn median_sorted(values: &[f64]) -> f64 {
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        let (a, b) = (values[n / 2 - 1], values[n / 2]);
        let mid = (a + b) / 2.0;
        if mid.is_finite() {
            mid
        } else {
            a / 2.0 + b / 2.0
        }
    }
}

fn mean_of(values: &[f64]) -> f64 {
    let mut sum = Sum::default();
    for &value in values {
        sum.add(value);
    }
    let total = sum.total();
    if total.is_finite() {
        total / values.len() as f64
    } else {
        mean_valid(values)
    }
}

/// Sort finite values through order-preserving integer keys, which sort
/// faster than `f64::total_cmp` and give the same order.
fn sort_finite(values: &mut [f64], keys: &mut Vec<u64>) {
    const SIGN: u64 = 1 << 63;
    keys.clear();
    keys.extend(values.iter().map(|value| {
        let bits = value.to_bits();
        if bits & SIGN == 0 {
            bits | SIGN
        } else {
            !bits
        }
    }));
    keys.sort_unstable();
    for (value, &key) in values.iter_mut().zip(keys.iter()) {
        *value = f64::from_bits(if key & SIGN != 0 { key & !SIGN } else { !key });
    }
}

/// Population standard deviation of sorted values about `mean`. Deviations
/// are divided by the largest one (at an end of the sorted run) before
/// squaring, so finite inputs do not overflow.
fn std_about(values: &[f64], mean: f64) -> f64 {
    let (Some(&first), Some(&last)) = (values.first(), values.last()) else {
        return f64::NAN;
    };
    let scale = (first - mean).abs().max((last - mean).abs());
    if scale == 0.0 || !scale.is_finite() {
        return if scale == 0.0 { 0.0 } else { f64::NAN };
    }
    let mut squares = Sum::default();
    for &value in values {
        let deviation = (value - mean) / scale;
        squares.add(deviation * deviation);
    }
    scale * (squares.total() / values.len() as f64).sqrt()
}

/// `astropy.stats.mad_std` of sorted values.
fn mad_std(values: &[f64], median: f64, scratch: &mut Vec<f64>) -> f64 {
    scratch.clear();
    scratch.extend(values.iter().map(|&value| (value - median).abs()));
    median_of(scratch) / NORMAL_UPPER_QUARTILE
}

/// Indices `lo..hi` of sorted `values[range]` inside `low..=high`.
fn within(values: &[f64], range: (usize, usize), low: f64, high: f64) -> (usize, usize) {
    let slice = &values[range.0..range.1];
    let lo = range.0 + slice.partition_point(|&value| value < low);
    let hi = range.0 + slice.partition_point(|&value| value <= high);
    (lo, hi.max(lo))
}

/// Iterative clip of a sorted sample. Survivors of interval bounds are a
/// contiguous range, so each pass is two binary searches.
#[allow(clippy::too_many_arguments)]
fn clip_sorted(
    values: &[f64],
    mut range: (usize, usize),
    lower: f64,
    upper: f64,
    maxiters: Option<usize>,
    center: ClipCenter,
    spread: ClipSpread,
    min_count: usize,
    scratch: &mut Vec<f64>,
) -> (usize, usize) {
    let mut iteration = 0usize;
    while range.1 - range.0 >= min_count.max(1) && maxiters.is_none_or(|limit| iteration < limit) {
        let kept = &values[range.0..range.1];
        let middle = match center {
            ClipCenter::Mean => mean_of(kept),
            ClipCenter::Median => median_sorted(kept),
        };
        let width = match spread {
            ClipSpread::Std => std_about(kept, mean_of(kept)),
            ClipSpread::Mad => mad_std(kept, median_sorted(kept), scratch),
        };
        let next = within(
            values,
            range,
            middle - lower * width,
            middle + upper * width,
        );
        iteration += 1;
        if next == range {
            break;
        }
        range = next;
    }
    range
}

fn sextractor_stats(values: &[f64], pearson: f64) -> (f64, f64, usize) {
    let all = (0, values.len());
    let mean = mean_of(values);
    let std = std_about(values, mean);
    let first = within(
        values,
        all,
        mean - SEXTRACTOR_FIRST_SIGMA * std,
        mean + SEXTRACTOR_FIRST_SIGMA * std,
    );
    if first.0 == first.1 {
        return (f64::NAN, f64::NAN, 0);
    }
    let subset = &values[first.0..first.1];
    let mean = mean_of(subset);
    let std = std_about(subset, mean);
    let span = within(
        values,
        all,
        mean - SEXTRACTOR_SPAN_SIGMA * std,
        mean + SEXTRACTOR_SPAN_SIGMA * std,
    );
    let (lo, hi) = clip_sorted(
        values,
        span,
        SEXTRACTOR_CLIP_SIGMA,
        SEXTRACTOR_CLIP_SIGMA,
        Some(SEXTRACTOR_MAXITERS),
        ClipCenter::Median,
        ClipSpread::Std,
        1,
        &mut Vec::new(),
    );
    if lo == hi {
        return (f64::NAN, f64::NAN, 0);
    }
    let kept = &values[lo..hi];
    let mean = mean_of(kept);
    let std = std_about(kept, mean);
    let background = mode_from_moments(mean, median_sorted(kept), std, ModeEstimator::Sex, pearson);
    (background, std, hi - lo)
}

fn clipped_stats(
    values: &[f64],
    scratch: &mut Vec<f64>,
    clip: ClipConfig,
    rms: MeshRms,
) -> (f64, f64, usize) {
    let (lo, hi) = clip_sorted(
        values,
        (0, values.len()),
        clip.lower,
        clip.upper,
        clip.maxiters,
        clip.center,
        clip.spread,
        clip.min_count,
        scratch,
    );
    let count = hi - lo;
    if count == 0 || count < clip.min_count {
        return (f64::NAN, f64::NAN, count);
    }
    let kept = &values[lo..hi];
    let mean = mean_of(kept);
    let median = median_sorted(kept);
    let background = match clip.statistic {
        ClipStatistic::Mean => mean,
        ClipStatistic::Median => median,
        ClipStatistic::Mode(estimator, pearson) => {
            mode_from_moments(mean, median, std_about(kept, mean), estimator, pearson)
        }
    };
    let spread = match rms {
        MeshRms::Std => std_about(kept, mean),
        MeshRms::MadStd => mad_std(kept, median, scratch),
    };
    (background, spread, count)
}

/// Reciprocal pivots of `M[i-1] + 4 M[i] + M[i+1] = r[i]` for the interior
/// nodes `1..n-1` of a natural spline with unit spacing.
fn natural_pivots(nodes: usize) -> Vec<f64> {
    let mut pivots = vec![0.0; nodes];
    let mut previous = 0.0;
    for pivot in pivots.iter_mut().take(nodes.saturating_sub(1)).skip(1) {
        *pivot = 1.0 / (4.0 - previous);
        previous = *pivot;
    }
    pivots
}

/// Second derivatives of the natural cubic spline through `y` (unit node
/// spacing), by the Thomas algorithm. `second` is resized to `y.len()`.
fn natural_second_derivatives(y: &[f64], pivots: &[f64], second: &mut Vec<f64>) {
    let n = y.len();
    second.clear();
    second.resize(n, 0.0);
    if n < 3 {
        return;
    }
    // Forward sweep stores the modified right-hand side.
    let mut previous = 0.0;
    for i in 1..n - 1 {
        let rhs = 6.0 * (y[i - 1] - 2.0 * y[i] + y[i + 1]);
        previous = (rhs - previous) * pivots[i];
        second[i] = previous;
    }
    for i in (1..n - 2).rev() {
        second[i] -= pivots[i] * second[i + 1];
    }
}

/// [`natural_second_derivatives`] for the `inner` interleaved columns of a
/// `nodes x inner` block.
fn columns_second_derivatives(
    block: &[f64],
    nodes: usize,
    inner: usize,
    pivots: &[f64],
    second: &mut [f64],
) {
    second.fill(0.0);
    if nodes < 3 {
        return;
    }
    for i in 1..nodes - 1 {
        for j in 0..inner {
            let rhs = 6.0
                * (block[(i - 1) * inner + j] - 2.0 * block[i * inner + j]
                    + block[(i + 1) * inner + j]);
            let previous = second[(i - 1) * inner + j];
            second[i * inner + j] = (rhs - previous) * pivots[i];
        }
    }
    for i in (1..nodes - 2).rev() {
        for j in 0..inner {
            second[i * inner + j] -= pivots[i] * second[(i + 1) * inner + j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mean of all finite samples, no clipping: mesh values are exact means.
    fn mean_config() -> MeshConfig {
        let clip = ClipConfig::new(3.0)
            .maxiters(Some(0))
            .statistic(ClipStatistic::Mean);
        MeshConfig::sextractor().estimator(MeshEstimator::SigmaClip {
            clip,
            rms: MeshRms::Std,
        })
    }

    fn maps(plan: &MeshBackground, data: &[f64], mask: Option<&[bool]>) -> (MeshGrid, Vec<f64>) {
        let grid = plan.estimate(data, mask).unwrap();
        let mut background = vec![0.0; plan.len()];
        plan.interpolate_into(grid.background(), &mut background)
            .unwrap();
        (grid, background)
    }

    fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() <= tolerance, "{index}: {a} != {e}");
        }
    }

    #[test]
    fn spline_reproduces_planes_in_two_and_three_dimensions() {
        // Mesh means of a plane equal the plane at full-mesh centers, and a
        // natural spline (including its end extrapolation) is exact for
        // linear data.
        for (shape, boxes) in [
            (vec![64, 96], vec![16, 16]),
            (vec![12, 10, 8], vec![4, 5, 4]),
            (vec![30], vec![6]),
        ] {
            let len: usize = shape.iter().product();
            let plane = |flat: usize| {
                let mut rest = flat;
                let mut value = 3.0;
                for (axis, &n) in shape.iter().enumerate().rev() {
                    value += (0.5 - 0.25 * axis as f64) * (rest % n) as f64;
                    rest /= n;
                }
                value
            };
            let data: Vec<f64> = (0..len).map(plane).collect();
            let filter = vec![1; shape.len()];
            let plan = MeshBackground::new(&shape, &boxes, &filter, mean_config()).unwrap();
            let (grid, background) = maps(&plan, &data, None);
            assert!(grid.valid().iter().all(|&valid| valid));
            assert_close(&background, &data, 1e-11);
        }
    }

    #[test]
    fn spline_row_matches_reference_literals() {
        // tests/_background_reference.py: natural spline through
        // [0, 8, 2, 10, 4] with second derivatives [0, -30, 36, -30, 0];
        // box width 3 over 14 columns (last mesh partial).
        let row = [0.0, 8.0, 2.0, 10.0, 4.0];
        let data: Vec<f64> = (0..2 * 14).map(|i| row[(i % 14) / 3]).collect();
        let plan = MeshBackground::new(&[2, 14], &[2, 3], &[1, 1], mean_config()).unwrap();
        let (_, background) = maps(&plan, &data, None);
        let expected = [
            -4.148148148148149,
            0.0,
            4.148148148148149,
            7.185185185185186,
            8.0,
            6.074074074074074,
            3.2592592592592604,
            2.0,
            3.9259259259259274,
            7.407407407407406,
            10.0,
            9.851851851851851,
            7.481481481481479,
            4.0,
        ];
        assert_close(&background[..14], &expected, 1e-13);
        assert_close(&background[14..], &expected, 1e-13);
    }

    #[test]
    fn sextractor_interpolation_with_one_mesh_row_is_linear() {
        let row = [0.0, 8.0, 2.0, 10.0];
        let data: Vec<f64> = (0..3 * 16).map(|i| row[(i % 16) / 4]).collect();
        let config = mean_config().interpolation(MeshInterpolation::SExtractor);
        let plan = MeshBackground::new(&[3, 16], &[4, 4], &[1, 1], config).unwrap();
        let (_, background) = maps(&plan, &data, None);
        // Nodes at 1.5, 5.5, 9.5, 13.5; linear between and beyond them.
        for (p, value) in background[..16].iter().enumerate() {
            let u = (p as f64 + 0.5) / 4.0 - 0.5;
            let seg = (u.floor().max(0.0) as usize).min(2);
            let t = u - seg as f64;
            assert!((value - ((1.0 - t) * row[seg] + t * row[seg + 1])).abs() < 1e-13);
        }
        assert!(MeshBackground::new(&[3, 4, 5], &[2, 2, 2], &[1, 1, 1], config).is_err());
    }

    #[test]
    fn sextractor_estimator_clips_outliers() {
        // Reference: 2-sigma cut drops 1000; 1..9 survive; mean = median = 5.
        let mut values: Vec<f64> = (1..=9).map(f64::from).collect();
        values.push(1000.0);
        let plan = MeshBackground::new(&[10], &[10], &[1], MeshConfig::sextractor()).unwrap();
        let grid = plan.estimate(&values, None).unwrap();
        assert_eq!(grid.background(), &[5.0]);
        assert!((grid.rms()[0] - 2.581988897471611).abs() < 1e-14);
        assert_eq!(grid.retained_count(), &[9]);
        assert_eq!(grid.finite_count(), &[10]);
    }

    #[test]
    fn invalid_samples_and_meshes_are_excluded_and_filled() {
        // 3x3 meshes of one pixel; the center is masked and the corner NaN.
        let mut data: Vec<f64> = (0..9).map(|i| i as f64 * 10.0).collect();
        data[0] = f64::NAN;
        let mut mask = vec![false; 9];
        mask[4] = true;
        let plan = MeshBackground::new(&[3, 3], &[1, 1], &[1, 1], mean_config()).unwrap();
        let grid = plan.estimate(&data, Some(&mask)).unwrap();
        assert_eq!(
            grid.valid(),
            &[false, true, true, true, false, true, true, true, true]
        );
        // Center: four neighbors at distance 1 (10, 30, 50, 70). Corner:
        // two at distance 1 (10, 30).
        assert_eq!(grid.background()[4], 40.0);
        assert_eq!(grid.background()[0], 20.0);
        // Inverse distance with ties at the last neighbor keeps all ties.
        let idw = |neighbors| {
            let config = mean_config().fill(MeshFill::InverseDistance {
                neighbors,
                power: 1.0,
            });
            let plan = MeshBackground::new(&[3, 3], &[1, 1], &[1, 1], config).unwrap();
            plan.estimate(&data, Some(&mask)).unwrap().background()[4]
        };
        assert_eq!(idw(2), 40.0);
        // Distance 1: 10, 30, 50, 70; distance sqrt(2): 20, 60, 80.
        let w = 1.0 / 2.0_f64.sqrt();
        let expected = (160.0 + w * 160.0) / (4.0 + 3.0 * w);
        assert!((idw(5) - expected).abs() < 1e-12);
    }

    #[test]
    fn median_filter_edges_and_thresholds() {
        let data = [1.0, 100.0, 3.0, 4.0, 5.0];
        let run = |edge, threshold| {
            let config = mean_config().filter(edge, threshold);
            let plan = MeshBackground::new(&[5], &[1], &[3], config).unwrap();
            plan.estimate(&data, None).unwrap().background().to_vec()
        };
        use MeshFilterEdge::{Shrink, Truncate};
        assert_eq!(run(Shrink, MeshThreshold::All), [1.0, 3.0, 4.0, 4.0, 5.0]);
        assert_eq!(
            run(Truncate, MeshThreshold::All),
            [50.5, 3.0, 4.0, 4.0, 4.5]
        );
        assert_eq!(
            run(Shrink, MeshThreshold::Deviation(10.0)),
            [1.0, 3.0, 3.0, 4.0, 5.0]
        );
        assert_eq!(
            run(Truncate, MeshThreshold::Above(50.0)),
            [1.0, 3.0, 3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn validity_counts_and_bases() {
        // Two meshes along 7 samples with box 4: [0..4) and partial [4..7).
        let data = [1.0, 1.0, f64::NAN, f64::NAN, 2.0, 2.0, 2.0];
        let valid = |fraction, count, base| {
            let config = mean_config().min_valid(fraction, count, base);
            let plan = MeshBackground::new(&[7], &[4], &[1], config).unwrap();
            plan.estimate(&data, None).unwrap().valid().to_vec()
        };
        use MeshValidBase::{Box, Mesh};
        use MeshValidCount::{Finite, Retained};
        assert_eq!(valid(0.5, Finite, Mesh), [true, true]);
        assert_eq!(valid(0.51, Finite, Mesh), [false, true]);
        assert_eq!(valid(0.9, Retained, Box), [false, false]);
        assert_eq!(valid(0.75, Finite, Box), [false, true]);
    }

    #[test]
    fn no_valid_mesh_gives_nan_maps() {
        let plan = MeshBackground::new(&[4, 4], &[2, 2], &[3, 3], mean_config()).unwrap();
        let (grid, background) = maps(&plan, &[f64::NAN; 16], None);
        assert!(grid.global_background().is_nan());
        assert!(background.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn invalid_configurations_are_errors() {
        let ok = MeshConfig::sextractor();
        assert!(MeshBackground::new(&[], &[], &[], ok).is_err());
        assert!(MeshBackground::new(&[4], &[0], &[1], ok).is_err());
        assert!(MeshBackground::new(&[4], &[2], &[2], ok).is_err());
        assert!(MeshBackground::new(&[4, 4], &[2], &[1, 1], ok).is_err());
        let fill = ok.fill(MeshFill::InverseDistance {
            neighbors: 0,
            power: 1.0,
        });
        let fraction = ok.min_valid(1.5, MeshValidCount::Finite, MeshValidBase::Mesh);
        let pearson = ok.estimator(MeshEstimator::SExtractor { pearson: 0.5 });
        let threshold = ok.filter(MeshFilterEdge::Shrink, MeshThreshold::Above(f64::NAN));
        for config in [fill, fraction, pearson, threshold] {
            assert!(matches!(
                MeshBackground::new(&[4], &[2], &[1], config),
                Err(Error::InvalidConfiguration(_))
            ));
        }
        let plan = MeshBackground::new(&[4], &[2], &[1], ok).unwrap();
        assert_eq!(
            plan.estimate(&[0.0_f64; 3], None),
            Err(Error::LengthMismatch)
        );
        assert_eq!(
            plan.estimate(&[0.0_f64; 4], Some(&[false; 3])),
            Err(Error::LengthMismatch)
        );
    }

    #[test]
    fn box_volume_overflow_is_an_error() {
        let huge = usize::MAX / 2;
        assert_eq!(
            MeshBackground::new(&[4, 4], &[huge, huge], &[1, 1], mean_config()).err(),
            Some(Error::Overflow)
        );
    }

    #[test]
    fn results_do_not_depend_on_thread_count() {
        let shape = [97, 83];
        let data: Vec<f64> = (0..97 * 83)
            .map(|i| {
                100.0 + ((i * 7919) % 1009) as f64 * 0.01 + ((i % 211 == 0) as u8) as f64 * 500.0
            })
            .collect();
        let mut mask = vec![false; data.len()];
        mask[..2000].fill(true);
        for config in [
            MeshConfig::sextractor(),
            MeshConfig::photutils(),
            MeshConfig::sextractor().interpolation(MeshInterpolation::SExtractor),
        ] {
            let plan = MeshBackground::new(&shape, &[10, 12], &[3, 3], config).unwrap();
            let run = |threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| maps(&plan, &data, Some(&mask)))
            };
            let (grid, map) = run(1);
            let (grid_many, map_many) = run(4);
            assert_eq!(grid, grid_many);
            assert_eq!(
                map.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                map_many.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn empty_axis_is_a_no_op() {
        let plan = MeshBackground::new(&[0, 5], &[2, 2], &[1, 1], MeshConfig::photutils()).unwrap();
        assert!(plan.is_empty());
        assert_eq!(plan.mesh_shape(), &[0, 3]);
        let grid = plan.estimate::<f32>(&[], None).unwrap();
        plan.interpolate_into::<f32>(grid.background(), &mut [])
            .unwrap();
    }
}
