use numpy::{Element, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use reducers::Float;

use super::filter::{allocate_output, python_error, WindowSpec};
use crate::location::ModeEstimator;
use crate::robust::{
    robust_into, Center, ClipConfig, RobustConfig, RobustReduction, SamplePolicy, Spread, Statistic,
};

fn parse_center(name: &str) -> PyResult<Center> {
    match name {
        "mean" => Ok(Center::Mean),
        "median" => Ok(Center::Median),
        _ => Err(PyValueError::new_err("center must be 'mean' or 'median'")),
    }
}

fn parse_estimator(name: &str) -> PyResult<ModeEstimator> {
    match name {
        "sex" => Ok(ModeEstimator::Sex),
        "mmm" => Ok(ModeEstimator::Mmm),
        "iraf" => Ok(ModeEstimator::Iraf),
        _ => Err(PyValueError::new_err(
            "mode estimator must be 'sex', 'mmm', or 'iraf'",
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_filter<T: Float + Element + Send + Sync>(
    py: Python<'_>,
    array: PyReadonlyArrayDyn<'_, T>,
    window: WindowSpec,
    mode: Vec<String>,
    cval: f64,
    reduction: RobustReduction,
) -> PyResult<Py<PyArrayDyn<T>>> {
    if !array.is_c_contiguous() || !array.is_aligned() {
        return Err(PyValueError::new_err(
            "native input must be aligned and C-contiguous",
        ));
    }
    let shape = array.shape().to_vec();
    let config = RobustConfig::from_window(
        window.build(shape.clone(), &mode)?,
        T::from_f64(cval),
        reduction,
    )
    .map_err(python_error)?;
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
    let output = allocate_output(py, &shape)?;
    {
        let mut write = output
            .try_readwrite()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let destination = write
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("native output must be C-contiguous"))?;
        py.detach(|| robust_into(&input, &config, destination))
            .map_err(python_error)?;
    }
    Ok(output.unbind())
}

#[pyfunction(signature = (array, size, footprint, origin, mode, cval, kind, parameter, omit_nan, ignore_inf, min_count, sigma_lower, sigma_upper, maxiters, center, spread, estimator="sex", modify_sample_size=false, pearson=2.5))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn robust_filter<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    size: Vec<usize>,
    footprint: Option<Vec<bool>>,
    origin: Vec<isize>,
    mode: Vec<String>,
    cval: f64,
    kind: &str,
    parameter: f64,
    omit_nan: bool,
    ignore_inf: bool,
    min_count: usize,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: Option<usize>,
    center: &str,
    spread: &str,
    estimator: &str,
    modify_sample_size: bool,
    pearson: f64,
) -> PyResult<Py<PyAny>> {
    let policy = if omit_nan {
        SamplePolicy::Omit {
            ignore_inf,
            min_count,
        }
    } else {
        SamplePolicy::Propagate
    };
    let reduction = match kind {
        "percentile" | "mad" | "mode" | "biweight_location" | "biweight_scale" => {
            if !omit_nan && (ignore_inf || min_count != 1) {
                return Err(PyValueError::new_err(
                    "ignore_inf and min_count require NaN omission",
                ));
            }
            match kind {
                "percentile" => RobustReduction::Percentile {
                    q: parameter,
                    policy,
                },
                "mad" => RobustReduction::Mad {
                    scale: parameter,
                    policy,
                },
                "mode" => RobustReduction::Mode {
                    estimator: parse_estimator(estimator)?,
                    pearson,
                    policy,
                },
                "biweight_location" => RobustReduction::BiweightLocation {
                    c: parameter,
                    policy,
                },
                _ => RobustReduction::BiweightScale {
                    c: parameter,
                    modify_sample_size,
                    policy,
                },
            }
        }
        "sigma_mean" | "sigma_median" | "sigma_mode" => RobustReduction::SigmaClip(ClipConfig {
            lower: sigma_lower,
            upper: sigma_upper,
            maxiters,
            center: parse_center(center)?,
            spread: match spread {
                "std" => Spread::Std,
                "mad" => Spread::Mad,
                _ => return Err(PyValueError::new_err("spread must be 'std' or 'mad'")),
            },
            statistic: match kind {
                "sigma_mean" => Statistic::Mean,
                "sigma_median" => Statistic::Median,
                _ => Statistic::Mode(parse_estimator(estimator)?, pearson),
            },
            min_count,
        }),
        _ => return Err(PyValueError::new_err("unsupported robust filter kind")),
    };
    let window = WindowSpec {
        size,
        footprint,
        origin,
    };
    if let Ok(typed) = array.extract::<PyReadonlyArrayDyn<'py, f32>>() {
        return run_filter(py, typed, window, mode, cval, reduction).map(|value| value.into_any());
    }
    let typed = array.extract::<PyReadonlyArrayDyn<'py, f64>>()?;
    run_filter(py, typed, window, mode, cval, reduction).map(|value| value.into_any())
}
