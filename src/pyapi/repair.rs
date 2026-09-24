use numpy::{
    npyffi::NPY_ORDER, Element, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn,
    PyUntypedArrayMethods, PY_ARRAY_API,
};
use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use reducers::Float;

#[cfg(feature = "test-hooks")]
use std::{sync::Arc, time::Duration};

use super::filter::{parse_modes, python_error, WindowSpec};
use crate::convolution::{ConvolutionConfig, WeightPolicy};
use crate::filter_nd::FilterConfig;
use crate::reduction::{InvalidPolicy, Reduction};

enum RepairMethod<T> {
    Linear(Vec<usize>),
    Median(FilterConfig<T>),
    Mean(ConvolutionConfig<T>),
}

fn run_fixpix<T, A, D>(
    py: Python<'_>,
    array: PyReadonlyArrayDyn<'_, T>,
    mask: Option<PyReadonlyArrayDyn<'_, u8>>,
    method: RepairMethod<T>,
    after_snapshot: A,
    during_detach: D,
) -> PyResult<Py<PyArrayDyn<T>>>
where
    T: Float + Element + Send + Sync,
    A: FnOnce() -> PyResult<()>,
    D: FnOnce() -> Result<(), &'static str> + Send,
{
    let shape = array.shape().to_vec();
    let mut positions = Vec::new();
    if let Some(mask) = mask {
        if mask.shape() != shape {
            return Err(PyValueError::new_err(
                "mask must have the same shape as array",
            ));
        }
        let flags = mask
            .as_slice()
            .map_err(|_| PyValueError::new_err("native mask must be C-contiguous"))?;
        positions = crate::fixpix::masked_positions(flags).map_err(python_error)?;
    }
    // SAFETY: array is a live typed NumPy array borrowed under the GIL.
    // NewCopy returns a fully initialized, independent C-order allocation;
    // it also accepts strided input without a normalization intermediate.
    let pointer =
        unsafe { PY_ARRAY_API.PyArray_NewCopy(py, array.as_array_ptr(), NPY_ORDER::NPY_CORDER) };
    let output =
        unsafe { Bound::from_owned_ptr_or_err(py, pointer)? }.cast_into::<PyArrayDyn<T>>()?;
    {
        let mut write = output
            .try_readwrite()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let result = write
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("native output is not C-contiguous"))?;
        // Output is a private owned snapshot; the original array and mask are
        // no longer borrowed when native work releases the GIL.
        drop(array);
        after_snapshot()?;
        let (probe_error, repair_result) = py.detach(|| match during_detach() {
            Ok(()) => (
                None,
                match &method {
                    RepairMethod::Linear(priority) => {
                        crate::fixpix::fixpix(result, &shape, &positions, priority)
                    }
                    RepairMethod::Median(config) => {
                        crate::filter_nd::repair_median(result, config, &positions)
                    }
                    RepairMethod::Mean(config) => {
                        crate::convolution::repair_masked(result, config, &positions)
                    }
                },
            ),
            Err(message) => (Some(message), Ok(())),
        });
        if let Some(message) = probe_error {
            return Err(PyRuntimeError::new_err(message));
        }
        repair_result.map_err(python_error)?;
    }
    Ok(output.unbind())
}

#[pyfunction]
pub(crate) fn fixpix<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArrayDyn<'py, u8>>,
    priority: Vec<usize>,
) -> PyResult<Py<PyAny>> {
    if let Ok(typed) = array.extract::<PyReadonlyArrayDyn<'py, f32>>() {
        return run_fixpix(
            py,
            typed,
            mask,
            RepairMethod::Linear(priority),
            || Ok(()),
            || Ok(()),
        )
        .map(|value| value.into_any());
    }
    let typed = array.extract::<PyReadonlyArrayDyn<'py, f64>>()?;
    run_fixpix(
        py,
        typed,
        mask,
        RepairMethod::Linear(priority),
        || Ok(()),
        || Ok(()),
    )
    .map(|value| value.into_any())
}

#[pyfunction]
// Keep the native binding's explicit arguments aligned with the Python API.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fixpix_median<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    mask: PyReadonlyArrayDyn<'py, u8>,
    size: Vec<usize>,
    footprint: Option<Vec<bool>>,
    origin: Vec<isize>,
    mode: Vec<String>,
    cval: f64,
    ignore_inf: bool,
    min_count: usize,
) -> PyResult<Py<PyAny>> {
    let reduction = Reduction::Median(InvalidPolicy::Omit {
        ignore_inf,
        min_count,
    });
    let window = WindowSpec {
        size,
        footprint,
        origin,
    };
    if let Ok(typed) = array.extract::<PyReadonlyArrayDyn<'py, f32>>() {
        let config = FilterConfig::from_window(
            window.build(typed.shape().to_vec(), &mode)?,
            cval as f32,
            reduction,
        )
        .map_err(python_error)?;
        return run_fixpix(
            py,
            typed,
            Some(mask),
            RepairMethod::Median(config),
            || Ok(()),
            || Ok(()),
        )
        .map(|value| value.into_any());
    }
    let typed = array.extract::<PyReadonlyArrayDyn<'py, f64>>()?;
    let config = FilterConfig::from_window(
        window.build(typed.shape().to_vec(), &mode)?,
        cval,
        reduction,
    )
    .map_err(python_error)?;
    run_fixpix(
        py,
        typed,
        Some(mask),
        RepairMethod::Median(config),
        || Ok(()),
        || Ok(()),
    )
    .map(|value| value.into_any())
}

#[pyfunction]
// Keep the native binding's explicit arguments aligned with the Python API.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fixpix_mean<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    mask: PyReadonlyArrayDyn<'py, u8>,
    weights: PyReadonlyArrayDyn<'py, f64>,
    mode: Vec<String>,
    cval: f64,
    origin: Vec<isize>,
    ignore_inf: bool,
    min_count: usize,
) -> PyResult<Py<PyAny>> {
    let kernel_shape = weights.shape().to_vec();
    let mut coefficients = Vec::new();
    coefficients
        .try_reserve_exact(weights.len())
        .map_err(|_| PyMemoryError::new_err("unable to allocate kernel snapshot"))?;
    // Logical C order, independent of the weights' memory layout.
    coefficients.extend(weights.as_array().iter().copied());
    drop(weights);
    let modes = parse_modes(&mode)?;
    // The normalized, NaN-omitting correlation that `nancorrelate` builds.
    let policy = WeightPolicy::Normalize {
        omit_nan: true,
        ignore_inf,
        min_count,
    };
    if let Ok(typed) = array.extract::<PyReadonlyArrayDyn<'py, f32>>() {
        let config = ConvolutionConfig::new(
            typed.shape().to_vec(),
            kernel_shape,
            coefficients,
            modes,
            f32::from_f64(cval),
            origin,
            true,
            policy,
        )
        .map_err(python_error)?;
        return run_fixpix(
            py,
            typed,
            Some(mask),
            RepairMethod::Mean(config),
            || Ok(()),
            || Ok(()),
        )
        .map(|value| value.into_any());
    }
    let typed = array.extract::<PyReadonlyArrayDyn<'py, f64>>()?;
    let config = ConvolutionConfig::new(
        typed.shape().to_vec(),
        kernel_shape,
        coefficients,
        modes,
        cval,
        origin,
        true,
        policy,
    )
    .map_err(python_error)?;
    run_fixpix(
        py,
        typed,
        Some(mask),
        RepairMethod::Mean(config),
        || Ok(()),
        || Ok(()),
    )
    .map(|value| value.into_any())
}

#[cfg(feature = "test-hooks")]
fn test_method(shape: &[usize], priority: Vec<usize>, median: bool) -> PyResult<RepairMethod<f64>> {
    if median {
        Ok(RepairMethod::Median(
            FilterConfig::new(
                shape.to_vec(),
                vec![3; shape.len()],
                vec![crate::boundary::BoundaryMode::Reflect; shape.len()],
                f64::NAN,
                Reduction::Median(InvalidPolicy::Omit {
                    ignore_inf: false,
                    min_count: 1,
                }),
            )
            .map_err(python_error)?,
        ))
    } else {
        Ok(RepairMethod::Linear(priority))
    }
}

#[cfg(feature = "test-hooks")]
#[pyfunction(name = "_test_fixpix_after_snapshot", signature = (array, mask, priority, callback, *, median=false))]
pub(crate) fn test_fixpix_after_snapshot<'py>(
    py: Python<'py>,
    array: PyReadonlyArrayDyn<'py, f64>,
    mask: PyReadonlyArrayDyn<'py, u8>,
    priority: Vec<usize>,
    callback: &Bound<'py, PyAny>,
    median: bool,
) -> PyResult<Py<PyAny>> {
    let method = test_method(array.shape(), priority, median)?;
    run_fixpix(
        py,
        array,
        Some(mask),
        method,
        || callback.call0().map(|_| ()),
        || Ok(()),
    )
    .map(|value| value.into_any())
}

#[cfg(feature = "test-hooks")]
#[pyfunction(name = "_test_fixpix_during_detach", signature = (array, mask, priority, probe, *, median=false))]
pub(crate) fn test_fixpix_during_detach<'py>(
    py: Python<'py>,
    array: PyReadonlyArrayDyn<'py, f64>,
    mask: PyReadonlyArrayDyn<'py, u8>,
    priority: Vec<usize>,
    probe: PyRef<'py, super::filter::DetachProbe>,
    median: bool,
) -> PyResult<Py<PyAny>> {
    let method = test_method(array.shape(), priority, median)?;
    let inner = Arc::clone(&probe.inner);
    run_fixpix(
        py,
        array,
        Some(mask),
        method,
        || Ok(()),
        move || super::filter::wait_for_python_progress(&inner, Duration::from_secs(5)),
    )
    .map(|value| value.into_any())
}
