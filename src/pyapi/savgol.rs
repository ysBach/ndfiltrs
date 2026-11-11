use numpy::{Element, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use reducers::Float;

use super::filter::{allocate_output, parse_modes, python_error};
use crate::reduction::InvalidPolicy;
use crate::savgol::{savgol_into, FitBoundary, PolynomialBasis, SavgolConfig, SavgolKernel};

fn run<T: Float + Element + Send + Sync>(
    py: Python<'_>,
    array: PyReadonlyArrayDyn<'_, T>,
    kernel: &SavgolKernel,
    policy: InvalidPolicy,
    cval: f64,
) -> PyResult<Py<PyAny>> {
    if !array.is_c_contiguous() || !array.is_aligned() {
        return Err(PyValueError::new_err(
            "native input must be aligned and C-contiguous",
        ));
    }
    let shape = array.shape().to_vec();
    let config =
        SavgolConfig::new(&shape, kernel, policy, T::from_f64(cval)).map_err(python_error)?;
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
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let result = write
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("native output must be C-contiguous"))?;
        // The input snapshot and plan own all data used while detached.
        py.detach(|| savgol_into(&input, &config, result))
            .map_err(python_error)?;
    }
    Ok(output.unbind().into_any())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn savgol_filter<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    window: Vec<usize>,
    basis: &str,
    degrees: Vec<usize>,
    derivative: Vec<usize>,
    delta: Vec<f64>,
    weights: Option<Vec<f64>>,
    mode: Vec<String>,
    cval: f64,
    omit_nan: bool,
    ignore_inf: bool,
    min_count: usize,
) -> PyResult<Py<PyAny>> {
    let basis = match (basis, degrees.as_slice()) {
        ("total", &[degree]) => PolynomialBasis::TotalDegree(degree),
        ("per_axis", _) => PolynomialBasis::PerAxisDegree(degrees),
        _ => return Err(PyValueError::new_err("unsupported polynomial basis")),
    };
    let boundaries = mode
        .iter()
        .map(|name| {
            if name == "interp" {
                Ok(FitBoundary::Interp)
            } else {
                parse_modes(std::slice::from_ref(name)).map(|modes| FitBoundary::Pad(modes[0]))
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    let mut kernel = SavgolKernel::new(window, basis)
        .derivative(derivative)
        .delta(delta)
        .boundaries(boundaries);
    if let Some(weights) = weights {
        kernel = kernel.weights(weights);
    }
    let policy = if omit_nan {
        InvalidPolicy::Omit {
            ignore_inf,
            min_count,
        }
    } else if ignore_inf || min_count != 1 {
        return Err(PyValueError::new_err(
            "ignore_inf and min_count require NaN omission",
        ));
    } else {
        InvalidPolicy::Propagate
    };
    if let Ok(typed) = array.extract::<PyReadonlyArrayDyn<'py, f32>>() {
        run(py, typed, &kernel, policy, cval)
    } else {
        run(
            py,
            array.extract::<PyReadonlyArrayDyn<'py, f64>>()?,
            &kernel,
            policy,
            cval,
        )
    }
}
