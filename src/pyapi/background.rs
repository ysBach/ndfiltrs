use numpy::{Element, PyArray1, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use super::filter::{allocate_output, python_error};
use crate::background::{
    MeshBackground, MeshConfig, MeshEstimator, MeshFill, MeshFilterEdge, MeshInterpolation,
    MeshRms, MeshThreshold, MeshValidBase, MeshValidCount,
};
use crate::location::ModeEstimator;
use crate::plan::FilterFloat;
use crate::robust::{ClipCenter, ClipConfig, ClipSpread, ClipStatistic};

/// Python names of the mesh rules, already validated by the Python layer.
pub(crate) struct MeshOptions<'a> {
    estimator: &'a str,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: Option<usize>,
    center: &'a str,
    spread: &'a str,
    statistic: &'a str,
    mode_estimator: &'a str,
    pearson: f64,
    rms: &'a str,
    min_valid_fraction: f64,
    valid_count: &'a str,
    valid_base: &'a str,
    fill: &'a str,
    idw_neighbors: usize,
    idw_power: f64,
    filter_edge: &'a str,
    filter_threshold: Option<f64>,
    threshold_kind: &'a str,
    interpolation: &'a str,
}

fn unknown(name: &str) -> PyErr {
    PyValueError::new_err(format!("unsupported {name}"))
}

impl MeshOptions<'_> {
    fn config(&self) -> PyResult<MeshConfig> {
        let estimator = match self.estimator {
            "sextractor" => MeshEstimator::SExtractor {
                pearson: self.pearson,
            },
            "sigma_clip" => {
                let mode = match self.mode_estimator {
                    "sex" => ModeEstimator::Sex,
                    "mmm" => ModeEstimator::Mmm,
                    "iraf" => ModeEstimator::Iraf,
                    _ => return Err(unknown("mode_estimator")),
                };
                let clip = ClipConfig::new(3.0)
                    .bounds(self.sigma_lower, self.sigma_upper)
                    .maxiters(self.maxiters)
                    .center(match self.center {
                        "median" => ClipCenter::Median,
                        "mean" => ClipCenter::Mean,
                        _ => return Err(unknown("center")),
                    })
                    .spread(match self.spread {
                        "std" => ClipSpread::Std,
                        "mad" => ClipSpread::Mad,
                        _ => return Err(unknown("spread")),
                    })
                    .statistic(match self.statistic {
                        "mean" => ClipStatistic::Mean,
                        "median" => ClipStatistic::Median,
                        "mode" => ClipStatistic::Mode(mode, self.pearson),
                        _ => return Err(unknown("statistic")),
                    });
                MeshEstimator::SigmaClip {
                    clip,
                    rms: match self.rms {
                        "std" => MeshRms::Std,
                        "mad" => MeshRms::MadStd,
                        _ => return Err(unknown("rms")),
                    },
                }
            }
            _ => return Err(unknown("estimator")),
        };
        let count = match self.valid_count {
            "finite" => MeshValidCount::Finite,
            "retained" => MeshValidCount::Retained,
            _ => return Err(unknown("valid_count")),
        };
        let base = match self.valid_base {
            "mesh" => MeshValidBase::Mesh,
            "box" => MeshValidBase::Box,
            _ => return Err(unknown("valid_base")),
        };
        let fill = match self.fill {
            "nearest" => MeshFill::Nearest,
            "idw" => MeshFill::InverseDistance {
                neighbors: self.idw_neighbors,
                power: self.idw_power,
            },
            _ => return Err(unknown("fill")),
        };
        let edge = match self.filter_edge {
            "shrink" => MeshFilterEdge::Shrink,
            "truncate" => MeshFilterEdge::Truncate,
            _ => return Err(unknown("filter_edge")),
        };
        let threshold = match (self.filter_threshold, self.threshold_kind) {
            (None, "deviation" | "above") => MeshThreshold::All,
            (Some(value), "deviation") => MeshThreshold::Deviation(value),
            (Some(value), "above") => MeshThreshold::Above(value),
            _ => return Err(unknown("threshold_kind")),
        };
        let interpolation = match self.interpolation {
            "spline" => MeshInterpolation::Spline,
            "sextractor" => MeshInterpolation::SExtractor,
            _ => return Err(unknown("interpolation")),
        };
        Ok(MeshConfig::sextractor()
            .estimator(estimator)
            .min_valid(self.min_valid_fraction, count, base)
            .fill(fill)
            .filter(edge, threshold)
            .interpolation(interpolation))
    }
}

type MeshOutput = (
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    f64,
    f64,
);

fn mesh_array<E: Element>(py: Python<'_>, values: Vec<E>, shape: &[usize]) -> PyResult<Py<PyAny>> {
    Ok(PyArray1::from_vec(py, values)
        .reshape(shape.to_vec())?
        .into_any()
        .unbind())
}

fn run<T: FilterFloat + Element>(
    py: Python<'_>,
    array: PyReadonlyArrayDyn<'_, T>,
    mask: Option<PyReadonlyArrayDyn<'_, bool>>,
    box_size: Vec<usize>,
    filter_size: Vec<usize>,
    config: MeshConfig,
) -> PyResult<MeshOutput> {
    let shape = array.shape().to_vec();
    let plan =
        MeshBackground::new(&shape, &box_size, &filter_size, config).map_err(python_error)?;
    // Snapshot the input so the GIL can be released without borrowing it.
    let mut input = Vec::new();
    {
        let source = array
            .as_slice()
            .map_err(|_| PyValueError::new_err("native input must be C-contiguous"))?;
        input
            .try_reserve_exact(source.len())
            .map_err(|_| PyMemoryError::new_err("unable to allocate native input snapshot"))?;
        input.extend_from_slice(source);
    }
    drop(array);
    let flags = match mask {
        Some(mask) => {
            if mask.shape() != shape {
                return Err(PyValueError::new_err(
                    "mask must have the same shape as data",
                ));
            }
            let source = mask
                .as_slice()
                .map_err(|_| PyValueError::new_err("native mask must be C-contiguous"))?;
            Some(source.to_vec())
        }
        None => None,
    };
    let background = allocate_output::<T>(py, &shape)?;
    let rms = allocate_output::<T>(py, &shape)?;
    let grid = {
        let mut background_write = background
            .try_readwrite()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let mut rms_write = rms
            .try_readwrite()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let background_out = background_write
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("native output must be C-contiguous"))?;
        let rms_out = rms_write
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("native output must be C-contiguous"))?;
        py.detach(|| {
            let grid = plan.estimate(&input, flags.as_deref())?;
            plan.interpolate_into(grid.background(), background_out)?;
            plan.interpolate_into(grid.rms(), rms_out)?;
            Ok(grid)
        })
        .map_err(python_error)?
    };
    let mesh_shape = grid.shape().to_vec();
    let to_i64 =
        |counts: &[usize]| -> Vec<i64> { counts.iter().map(|&count| count as i64).collect() };
    Ok((
        background.into_any().unbind(),
        rms.into_any().unbind(),
        mesh_array(py, grid.background().to_vec(), &mesh_shape)?,
        mesh_array(py, grid.rms().to_vec(), &mesh_shape)?,
        mesh_array(py, grid.valid().to_vec(), &mesh_shape)?,
        mesh_array(py, to_i64(grid.finite_count()), &mesh_shape)?,
        mesh_array(py, to_i64(grid.retained_count()), &mesh_shape)?,
        grid.global_background(),
        grid.global_rms(),
    ))
}

#[pyfunction(signature = (array, mask, box_size, filter_size, estimator, sigma_lower, sigma_upper, maxiters, center, spread, statistic, mode_estimator, pearson, rms, min_valid_fraction, valid_count, valid_base, fill, idw_neighbors, idw_power, filter_edge, filter_threshold, threshold_kind, interpolation))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn mesh_background<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArrayDyn<'py, bool>>,
    box_size: Vec<usize>,
    filter_size: Vec<usize>,
    estimator: &str,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: Option<usize>,
    center: &str,
    spread: &str,
    statistic: &str,
    mode_estimator: &str,
    pearson: f64,
    rms: &str,
    min_valid_fraction: f64,
    valid_count: &str,
    valid_base: &str,
    fill: &str,
    idw_neighbors: usize,
    idw_power: f64,
    filter_edge: &str,
    filter_threshold: Option<f64>,
    threshold_kind: &str,
    interpolation: &str,
) -> PyResult<MeshOutput> {
    let config = MeshOptions {
        estimator,
        sigma_lower,
        sigma_upper,
        maxiters,
        center,
        spread,
        statistic,
        mode_estimator,
        pearson,
        rms,
        min_valid_fraction,
        valid_count,
        valid_base,
        fill,
        idw_neighbors,
        idw_power,
        filter_edge,
        filter_threshold,
        threshold_kind,
        interpolation,
    }
    .config()?;
    if let Ok(typed) = array.extract::<PyReadonlyArrayDyn<'py, f32>>() {
        return run(py, typed, mask, box_size, filter_size, config);
    }
    let typed = array.extract::<PyReadonlyArrayDyn<'py, f64>>()?;
    run(py, typed, mask, box_size, filter_size, config)
}
