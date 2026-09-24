use numpy::{Element, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use reducers::Float;

use super::filter::{allocate_output, parse_modes, python_error};
use crate::separable::smooth_into;

struct Options {
    axes: Vec<usize>,
    kernels: Vec<Vec<f64>>,
    mode: Vec<String>,
    cval: f64,
    omit_nan: bool,
    ignore_inf: bool,
    min_count: usize,
}

fn run<T: Float + Element + Send + Sync>(
    py: Python<'_>,
    array: PyReadonlyArrayDyn<'_, T>,
    options: Options,
) -> PyResult<Py<PyAny>> {
    if !array.is_c_contiguous() || !array.is_aligned() {
        return Err(PyValueError::new_err(
            "native input must be aligned and C-contiguous",
        ));
    }
    let shape = array.shape().to_vec();
    let modes = parse_modes(&options.mode)?;
    let mut input = Vec::new();
    {
        let source = array
            .as_slice()
            .map_err(|_| PyValueError::new_err("native input must be C-contiguous"))?;
        input
            .try_reserve_exact(source.len())
            .map_err(|_| PyMemoryError::new_err("unable to allocate input snapshot"))?;
        input.extend_from_slice(source);
    }
    drop(array);
    let output: Bound<'_, PyArrayDyn<T>> = allocate_output(py, &shape)?;
    {
        let mut write = output
            .try_readwrite()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let result = write
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("native output must be C-contiguous"))?;
        py.detach(|| {
            smooth_into(
                &input,
                &shape,
                &options.axes,
                &options.kernels,
                &modes,
                T::from_f64(options.cval),
                options.omit_nan,
                options.ignore_inf,
                options.min_count,
                result,
            )
        })
        .map_err(python_error)?;
    }
    Ok(output.unbind().into_any())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn separable_smooth<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    axes: Vec<usize>,
    kernels: Vec<Vec<f64>>,
    mode: Vec<String>,
    cval: f64,
    omit_nan: bool,
    ignore_inf: bool,
    min_count: usize,
) -> PyResult<Py<PyAny>> {
    let options = Options {
        axes,
        kernels,
        mode,
        cval,
        omit_nan,
        ignore_inf,
        min_count,
    };
    if let Ok(typed) = array.extract::<PyReadonlyArrayDyn<'py, f32>>() {
        run(py, typed, options)
    } else {
        run(
            py,
            array.extract::<PyReadonlyArrayDyn<'py, f64>>()?,
            options,
        )
    }
}
