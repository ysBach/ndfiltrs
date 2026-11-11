use numpy::{Element, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyMemoryError, PyOverflowError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use super::filter::{allocate_output, parse_modes, python_error};
use crate::plan::FilterFloat;
use crate::reduction::InvalidPolicy;
use crate::variable::{variable_mean_into, VariableWindow};

fn snapshot<T: Element + Copy>(array: &PyReadonlyArrayDyn<'_, T>) -> PyResult<Vec<T>> {
    let source = array
        .as_slice()
        .map_err(|_| PyValueError::new_err("native input must be C-contiguous"))?;
    let mut copy = Vec::new();
    copy.try_reserve_exact(source.len())
        .map_err(|_| PyMemoryError::new_err("unable to allocate native input snapshot"))?;
    copy.extend_from_slice(source);
    Ok(copy)
}

fn run<T: FilterFloat + Element>(
    py: Python<'_>,
    array: PyReadonlyArrayDyn<'_, T>,
    sizes: Vec<usize>,
    per_axis: bool,
    modes: &[String],
    cval: f64,
    policy: InvalidPolicy,
) -> PyResult<Py<PyAny>> {
    let shape = array.shape().to_vec();
    let input = snapshot(&array)?;
    drop(array);
    let window = if per_axis {
        VariableWindow::per_axis(sizes)
    } else {
        VariableWindow::isotropic(sizes)
    }
    .modes(parse_modes(modes)?);
    let output = allocate_output::<T>(py, &shape)?;
    {
        let mut write = output
            .try_readwrite()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let result = write
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("native output must be C-contiguous"))?;
        py.detach(|| {
            variable_mean_into(&input, &shape, &window, policy, T::from_f64(cval), result)
        })
        .map_err(python_error)?;
    }
    Ok(output.unbind().into_any())
}

/// Mean over per-pixel windows; `sizes` is a validated C-contiguous `uint64` map.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn variable_mean<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    sizes: PyReadonlyArrayDyn<'py, u64>,
    per_axis: bool,
    mode: Vec<String>,
    cval: f64,
    omit_nan: bool,
    ignore_inf: bool,
    min_count: usize,
) -> PyResult<Py<PyAny>> {
    let sizes = snapshot(&sizes)?
        .into_iter()
        .map(usize::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| PyOverflowError::new_err("window size exceeds the native index range"))?;
    let policy = if omit_nan {
        InvalidPolicy::Omit {
            ignore_inf,
            min_count,
        }
    } else {
        InvalidPolicy::Propagate
    };
    if let Ok(typed) = array.extract::<PyReadonlyArrayDyn<'py, f32>>() {
        run(py, typed, sizes, per_axis, &mode, cval, policy)
    } else {
        let typed = array.extract::<PyReadonlyArrayDyn<'py, f64>>()?;
        run(py, typed, sizes, per_axis, &mode, cval, policy)
    }
}
