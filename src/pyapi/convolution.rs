use numpy::{Element, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use reducers::Float;

#[cfg(feature = "test-hooks")]
use std::{sync::Arc, time::Duration};

use super::filter::{allocate_output, parse_modes, python_error};
use crate::convolution::{convolve_into, ConvolutionConfig, WeightPolicy};

struct ConvolutionHooks<A, D> {
    after_snapshot: A,
    during_detach: D,
}

#[allow(clippy::too_many_arguments)]
fn run_convolve<T, A, D>(
    py: Python<'_>,
    array: PyReadonlyArrayDyn<'_, T>,
    weights: PyReadonlyArrayDyn<'_, f64>,
    mode: Vec<String>,
    cval: f64,
    origin: Vec<isize>,
    correlate: bool,
    policy: WeightPolicy,
    hooks: ConvolutionHooks<A, D>,
) -> PyResult<Py<PyArrayDyn<T>>>
where
    T: Float + Element + Send + Sync,
    A: FnOnce() -> PyResult<()>,
    D: FnOnce() -> Result<(), &'static str> + Send,
{
    let ConvolutionHooks {
        after_snapshot,
        during_detach,
    } = hooks;
    if !array.is_c_contiguous() || !array.is_aligned() {
        return Err(PyValueError::new_err(
            "native input must be aligned and C-contiguous",
        ));
    }
    if !weights.is_c_contiguous() || !weights.is_aligned() {
        return Err(PyValueError::new_err(
            "native weights must be aligned and C-contiguous",
        ));
    }
    let shape = array.shape().to_vec();
    let kernel_shape = weights.shape().to_vec();
    let mut coefficients = Vec::new();
    {
        let borrowed = weights
            .as_slice()
            .map_err(|_| PyValueError::new_err("native weights must be C-contiguous"))?;
        coefficients
            .try_reserve_exact(borrowed.len())
            .map_err(|_| PyMemoryError::new_err("unable to allocate kernel snapshot"))?;
        coefficients.extend_from_slice(borrowed);
    }
    drop(weights);
    let config = ConvolutionConfig::new(
        shape.clone(),
        kernel_shape,
        coefficients,
        parse_modes(&mode)?,
        T::from_f64(cval),
        origin,
        correlate,
        policy,
    )
    .map_err(python_error)?;
    let mut input = Vec::new();
    {
        let borrowed = array
            .as_slice()
            .map_err(|_| PyValueError::new_err("native input must be C-contiguous"))?;
        input
            .try_reserve_exact(borrowed.len())
            .map_err(|_| PyMemoryError::new_err("unable to allocate native input snapshot"))?;
        input.extend_from_slice(borrowed);
    }
    drop(array);
    after_snapshot()?;
    let output = allocate_output(py, &shape)?;
    {
        let mut write = output
            .try_readwrite()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let result = write
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("native output is not C-contiguous"))?;
        // Neither input nor coefficients reference Python-owned storage here.
        let (probe_error, convolution_result) = py.detach(|| match during_detach() {
            Ok(()) => (None, convolve_into(&input, &config, result)),
            Err(message) => (Some(message), Ok(())),
        });
        if let Some(message) = probe_error {
            return Err(PyRuntimeError::new_err(message));
        }
        convolution_result.map_err(python_error)?;
    }
    Ok(output.unbind())
}

fn weight_policy(
    omit_nan: bool,
    ignore_inf: bool,
    min_count: usize,
    normalize: bool,
) -> PyResult<WeightPolicy> {
    if !omit_nan && (ignore_inf || min_count != 1) {
        return Err(PyValueError::new_err(
            "ignore_inf and min_count require NaN omission",
        ));
    }
    if omit_nan || normalize {
        Ok(WeightPolicy::Normalize {
            omit_nan,
            ignore_inf,
            min_count,
        })
    } else {
        Ok(WeightPolicy::Sum)
    }
}

#[pyfunction(signature = (array, weights, mode, cval, origin, correlate, omit_nan, ignore_inf, min_count, normalize=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn convolve<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    weights: PyReadonlyArrayDyn<'py, f64>,
    mode: Vec<String>,
    cval: f64,
    origin: Vec<isize>,
    correlate: bool,
    omit_nan: bool,
    ignore_inf: bool,
    min_count: usize,
    normalize: bool,
) -> PyResult<Py<PyAny>> {
    let policy = weight_policy(omit_nan, ignore_inf, min_count, normalize)?;
    if let Ok(typed) = array.extract::<PyReadonlyArrayDyn<'py, f32>>() {
        return run_convolve(
            py,
            typed,
            weights,
            mode,
            cval,
            origin,
            correlate,
            policy,
            ConvolutionHooks {
                after_snapshot: || Ok(()),
                during_detach: || Ok(()),
            },
        )
        .map(|value| value.into_any());
    }
    let typed = array.extract::<PyReadonlyArrayDyn<'py, f64>>()?;
    run_convolve(
        py,
        typed,
        weights,
        mode,
        cval,
        origin,
        correlate,
        policy,
        ConvolutionHooks {
            after_snapshot: || Ok(()),
            during_detach: || Ok(()),
        },
    )
    .map(|value| value.into_any())
}

#[cfg(feature = "test-hooks")]
#[pyfunction(name = "_test_convolve_after_snapshot", signature = (array, weights, mode, cval, origin, callback, *, correlate=false, omit_nan=false, ignore_inf=false, min_count=1))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn test_convolve_after_snapshot<'py>(
    py: Python<'py>,
    array: PyReadonlyArrayDyn<'py, f64>,
    weights: PyReadonlyArrayDyn<'py, f64>,
    mode: Vec<String>,
    cval: f64,
    origin: Vec<isize>,
    callback: &Bound<'py, PyAny>,
    correlate: bool,
    omit_nan: bool,
    ignore_inf: bool,
    min_count: usize,
) -> PyResult<Py<PyAny>> {
    run_convolve(
        py,
        array,
        weights,
        mode,
        cval,
        origin,
        correlate,
        weight_policy(omit_nan, ignore_inf, min_count, false)?,
        ConvolutionHooks {
            after_snapshot: || callback.call0().map(|_| ()),
            during_detach: || Ok(()),
        },
    )
    .map(|value| value.into_any())
}

#[cfg(feature = "test-hooks")]
#[pyfunction(name = "_test_convolve_during_detach", signature = (array, weights, mode, cval, origin, probe, *, correlate=false, omit_nan=false, ignore_inf=false, min_count=1))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn test_convolve_during_detach<'py>(
    py: Python<'py>,
    array: PyReadonlyArrayDyn<'py, f64>,
    weights: PyReadonlyArrayDyn<'py, f64>,
    mode: Vec<String>,
    cval: f64,
    origin: Vec<isize>,
    probe: PyRef<'py, super::filter::DetachProbe>,
    correlate: bool,
    omit_nan: bool,
    ignore_inf: bool,
    min_count: usize,
) -> PyResult<Py<PyAny>> {
    let inner = Arc::clone(&probe.inner);
    run_convolve(
        py,
        array,
        weights,
        mode,
        cval,
        origin,
        correlate,
        weight_policy(omit_nan, ignore_inf, min_count, false)?,
        ConvolutionHooks {
            after_snapshot: || Ok(()),
            during_detach: move || {
                super::filter::wait_for_python_progress(&inner, Duration::from_secs(5))
            },
        },
    )
    .map(|value| value.into_any())
}
