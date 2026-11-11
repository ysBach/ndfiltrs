use numpy::{Element, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use reducers::Float;

use super::filter::{allocate_output, parse_modes, python_error};
use crate::convolution::{ConvolutionConfig, WeightPolicy};
use crate::variance::{convolve_noise_into, Estimator, NoiseMap, NoiseOutputs, NoiseSpec};

fn snapshot<T: Copy + Element>(array: PyReadonlyArrayDyn<'_, T>, name: &str) -> PyResult<Vec<T>> {
    if !array.is_c_contiguous() || !array.is_aligned() {
        return Err(PyValueError::new_err(format!(
            "native {name} must be aligned and C-contiguous"
        )));
    }
    let borrowed = array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("native {name} must be C-contiguous")))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(borrowed.len())
        .map_err(|_| PyMemoryError::new_err(format!("unable to allocate {name} snapshot")))?;
    values.extend_from_slice(borrowed);
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn run<'py, T>(
    py: Python<'py>,
    array: PyReadonlyArrayDyn<'py, T>,
    noise: &Bound<'py, PyAny>,
    weights: PyReadonlyArrayDyn<'py, f64>,
    modes: Vec<String>,
    cval: f64,
    origin: Vec<isize>,
    correlate: bool,
    spec: NoiseSpec,
    with_snr: bool,
) -> PyResult<Vec<Py<PyAny>>>
where
    T: Float + Element + Send + Sync,
{
    let noise = noise
        .extract::<PyReadonlyArrayDyn<'py, T>>()
        .map_err(|_| PyTypeError::new_err("native noise must have the data dtype"))?;
    let shape = array.shape().to_vec();
    if noise.shape() != shape.as_slice() {
        return Err(PyValueError::new_err("noise must have the data shape"));
    }
    let kernel_shape = weights.shape().to_vec();
    let coefficients = snapshot(weights, "weights")?;
    let config = ConvolutionConfig::new(
        shape.clone(),
        kernel_shape,
        coefficients,
        parse_modes(&modes)?,
        T::from_f64(cval),
        origin,
        correlate,
        WeightPolicy::Sum,
    )
    .map_err(python_error)?;
    spec.validate(&config).map_err(python_error)?;
    let data = snapshot(array, "input")?;
    let noise = snapshot(noise, "noise")?;
    let outputs: Vec<Bound<'py, PyArrayDyn<T>>> = (0..if with_snr { 3 } else { 2 })
        .map(|_| allocate_output(py, &shape))
        .collect::<PyResult<_>>()?;
    {
        let mut writes = outputs
            .iter()
            .map(|output| {
                output
                    .try_readwrite()
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut slices = writes
            .iter_mut()
            .map(|write| {
                write
                    .as_slice_mut()
                    .map_err(|_| PyRuntimeError::new_err("native output is not C-contiguous"))
            })
            .collect::<PyResult<Vec<_>>>()?
            .into_iter();
        let value = slices.next().expect("two outputs");
        let variance = slices.next().expect("two outputs");
        let snr = slices.next();
        // Inputs and coefficients are snapshots, so no Python storage is shared.
        py.detach(|| {
            convolve_noise_into(
                &data,
                &noise,
                &config,
                &spec,
                NoiseOutputs {
                    value,
                    variance,
                    snr,
                },
            )
        })
        .map_err(python_error)?;
    }
    Ok(outputs
        .into_iter()
        .map(|output| output.into_any().unbind())
        .collect())
}

/// Convolve with noise propagation; returns `[value, variance]` or
/// `[value, variance, snr]`.
#[pyfunction(signature = (array, noise, weights, mode, cval, origin, correlate, estimator, inverse_weighting, inverse_noise, omit_nan, ignore_inf, min_count, with_snr))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn convolve_noise<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    noise: &Bound<'py, PyAny>,
    weights: PyReadonlyArrayDyn<'py, f64>,
    mode: Vec<String>,
    cval: f64,
    origin: Vec<isize>,
    correlate: bool,
    estimator: &str,
    inverse_weighting: bool,
    inverse_noise: bool,
    omit_nan: bool,
    ignore_inf: bool,
    min_count: usize,
    with_snr: bool,
) -> PyResult<Vec<Py<PyAny>>> {
    let estimator = match estimator {
        "sum" => Estimator::Sum,
        "mean" => Estimator::Mean,
        "amplitude" => Estimator::Amplitude,
        _ => return Err(PyValueError::new_err("unsupported estimator")),
    };
    let spec = NoiseSpec {
        noise: if inverse_noise {
            NoiseMap::InverseVariance
        } else {
            NoiseMap::Variance
        },
        inverse_weighting,
        estimator,
        omit_nan,
        ignore_inf,
        min_count,
    };
    if let Ok(typed) = array.extract::<PyReadonlyArrayDyn<'py, f32>>() {
        return run(
            py, typed, noise, weights, mode, cval, origin, correlate, spec, with_snr,
        );
    }
    let typed = array.extract::<PyReadonlyArrayDyn<'py, f64>>()?;
    run(
        py, typed, noise, weights, mode, cval, origin, correlate, spec, with_snr,
    )
}
