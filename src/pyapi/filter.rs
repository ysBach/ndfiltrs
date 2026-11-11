use std::os::raw::c_int;

#[cfg(feature = "test-hooks")]
use std::{
    sync::{Arc, Condvar, Mutex},
    time::Duration,
};

use numpy::{
    npyffi::npy_intp, Element, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn,
    PyUntypedArrayMethods, PY_ARRAY_API,
};
use pyo3::exceptions::{PyMemoryError, PyOverflowError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::boundary::BoundaryMode;
use crate::error::Error;
use crate::parallel;
use crate::plan::{FilterFloat, Neighborhood, WindowFilter};
use crate::reduction::{InvalidPolicy, Reduction};
use crate::window::Window;

pub(super) fn parse_modes(modes: &[String]) -> PyResult<Vec<BoundaryMode>> {
    modes
        .iter()
        .map(|mode| match mode.as_str() {
            "reflect" => Ok(BoundaryMode::Reflect),
            "constant" => Ok(BoundaryMode::Constant),
            "nearest" => Ok(BoundaryMode::Nearest),
            "mirror" => Ok(BoundaryMode::Mirror),
            "wrap" => Ok(BoundaryMode::Wrap),
            _ => Err(PyValueError::new_err("unsupported boundary mode")),
        })
        .collect()
}

pub(super) fn python_error(error: Error) -> PyErr {
    match error {
        Error::Memory => PyMemoryError::new_err(error.to_string()),
        Error::Overflow => PyOverflowError::new_err(error.to_string()),
        Error::EmptyAxis | Error::InvalidConfiguration(_) | Error::LengthMismatch => {
            PyValueError::new_err(error.to_string())
        }
    }
}

#[cfg(feature = "test-hooks")]
#[derive(Default)]
pub(super) struct DetachProbeState {
    started: bool,
    progress: bool,
}

#[cfg(feature = "test-hooks")]
pub(super) type DetachProbeInner = Arc<(Mutex<DetachProbeState>, Condvar)>;

#[cfg(feature = "test-hooks")]
#[pyclass(name = "_DetachProbe", frozen)]
pub(crate) struct DetachProbe {
    pub(super) inner: DetachProbeInner,
}

#[cfg(feature = "test-hooks")]
#[pymethods]
impl DetachProbe {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new((Mutex::new(DetachProbeState::default()), Condvar::new())),
        }
    }

    fn wait_until_started(&self, py: Python<'_>, timeout_seconds: f64) -> PyResult<bool> {
        let timeout = Duration::try_from_secs_f64(timeout_seconds)
            .map_err(|_| PyValueError::new_err("timeout must be finite and non-negative"))?;
        let result = py.detach(|| {
            let (state_lock, condition) = &*self.inner;
            let state = state_lock.lock().map_err(|_| ())?;
            let (state, _) = condition
                .wait_timeout_while(state, timeout, |state| !state.started)
                .map_err(|_| ())?;
            Ok::<bool, ()>(state.started)
        });
        result.map_err(|()| PyRuntimeError::new_err("detach probe lock was poisoned"))
    }

    fn signal_progress(&self) -> PyResult<()> {
        let (state_lock, condition) = &*self.inner;
        let mut state = state_lock
            .lock()
            .map_err(|_| PyRuntimeError::new_err("detach probe lock was poisoned"))?;
        state.progress = true;
        condition.notify_all();
        Ok(())
    }
}

#[cfg(feature = "test-hooks")]
pub(super) fn wait_for_python_progress(
    inner: &DetachProbeInner,
    timeout: Duration,
) -> Result<(), &'static str> {
    let (state_lock, condition) = &**inner;
    let mut state = state_lock
        .lock()
        .map_err(|_| "detach probe lock was poisoned")?;
    state.started = true;
    condition.notify_all();
    let (state, _) = condition
        .wait_timeout_while(state, timeout, |state| !state.progress)
        .map_err(|_| "detach probe lock was poisoned")?;
    if state.progress {
        Ok(())
    } else {
        Err("Python worker made no progress while native filtering was detached")
    }
}

struct FilterHooks<A, D> {
    after_snapshot: A,
    during_detach: D,
}

pub(super) fn allocate_output<'py, T: Element>(
    py: Python<'py>,
    shape: &[usize],
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    let mut dimensions = Vec::new();
    dimensions
        .try_reserve_exact(shape.len())
        .map_err(|_| PyMemoryError::new_err("unable to allocate output dimensions"))?;
    for &dimension in shape {
        dimensions.push(
            npy_intp::try_from(dimension)
                .map_err(|_| PyOverflowError::new_err("array dimension is too large"))?,
        );
    }
    let ndim = c_int::try_from(dimensions.len())
        .map_err(|_| PyOverflowError::new_err("array rank is too large"))?;

    // SAFETY: `PyArray_Zeros` receives a new dtype reference and valid
    // dimensions. A non-null return owns initialized, C-contiguous storage;
    // its sole reference stays in this function until filtering has finished.
    let pointer = unsafe {
        PY_ARRAY_API.PyArray_Zeros(
            py,
            ndim,
            dimensions.as_mut_ptr(),
            T::get_dtype(py).into_dtype_ptr(),
            0,
        )
    };
    let output = unsafe { Bound::from_owned_ptr_or_err(py, pointer)? };
    Ok(output.cast_into()?)
}

/// Window geometry received from the Python validation layer.
pub(super) struct WindowSpec {
    pub(super) size: Vec<usize>,
    pub(super) footprint: Option<Vec<bool>>,
    pub(super) origin: Vec<isize>,
}

impl WindowSpec {
    #[cfg(feature = "test-hooks")]
    pub(super) fn rectangular(size: Vec<usize>) -> Self {
        let origin = vec![0; size.len()];
        Self {
            size,
            footprint: None,
            origin,
        }
    }

    fn neighborhood(self, modes: &[String]) -> PyResult<Neighborhood> {
        let neighborhood = match self.footprint {
            Some(mask) => Neighborhood::footprint(self.size, mask),
            None => Neighborhood::rectangular(self.size),
        };
        Ok(neighborhood.origin(self.origin).modes(parse_modes(modes)?))
    }

    pub(super) fn build(self, shape: Vec<usize>, modes: &[String]) -> PyResult<Window> {
        Window::new(
            shape,
            &self.size,
            self.footprint.as_deref(),
            &self.origin,
            parse_modes(modes)?,
        )
        .map_err(python_error)
    }
}

fn run_filter<T, A, D>(
    py: Python<'_>,
    array: PyReadonlyArrayDyn<'_, T>,
    window: WindowSpec,
    modes: Vec<String>,
    cval: T,
    reduction: Reduction,
    hooks: FilterHooks<A, D>,
) -> PyResult<Py<PyArrayDyn<T>>>
where
    T: FilterFloat + Element,
    A: FnOnce() -> PyResult<()>,
    D: FnOnce() -> Result<(), &'static str> + Send,
{
    let FilterHooks {
        after_snapshot,
        during_detach,
    } = hooks;
    let shape = array.shape().to_vec();
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err("native input must be C-contiguous"));
    }
    let neighborhood = window.neighborhood(&modes)?;
    let plan = WindowFilter::new(&shape, &neighborhood, reduction, cval).map_err(python_error)?;
    let mut input = Vec::new();
    {
        let borrowed_input = array
            .as_slice()
            .map_err(|_| PyValueError::new_err("native input must be C-contiguous"))?;
        input
            .try_reserve_exact(borrowed_input.len())
            .map_err(|_| PyMemoryError::new_err("unable to allocate native input snapshot"))?;
        input.extend_from_slice(borrowed_input);
    }
    drop(array);
    after_snapshot()?;
    let output = allocate_output(py, &shape)?;
    {
        let mut write = output
            .try_readwrite()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let output_slice = write
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("native output is not C-contiguous"))?;
        let (probe_error, filter_result) = py.detach(|| match during_detach() {
            Ok(()) => (None, plan.apply_into(&input, output_slice)),
            Err(message) => (Some(message), Ok(())),
        });
        if let Some(message) = probe_error {
            return Err(PyRuntimeError::new_err(message));
        }
        filter_result.map_err(python_error)?;
    }
    Ok(output.unbind())
}

fn no_hooks(
) -> FilterHooks<impl FnOnce() -> PyResult<()>, impl FnOnce() -> Result<(), &'static str> + Send> {
    FilterHooks {
        after_snapshot: || Ok(()),
        during_detach: || Ok(()),
    }
}

/// Apply one generic window reduction.
///
/// `kind` is one of `median`, `median_network`, `median_histogram`, `mean`,
/// `mean_rolling`, `min`, `max`, `var`, `std`, `rank`, or `rank_histogram`.
/// `parameter` is `ddof` for `var`/`std` and the nonnegative rank for the
/// rank kinds; other kinds ignore it.
#[pyfunction(signature = (array, kind, size, footprint, origin, mode, cval, omit_nan, ignore_inf, min_count, parameter))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn window_filter<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    kind: &str,
    size: Vec<usize>,
    footprint: Option<Vec<bool>>,
    origin: Vec<isize>,
    mode: Vec<String>,
    cval: f64,
    omit_nan: bool,
    ignore_inf: bool,
    min_count: usize,
    parameter: usize,
) -> PyResult<Py<PyAny>> {
    let policy = if omit_nan {
        InvalidPolicy::Omit {
            ignore_inf,
            min_count,
        }
    } else {
        InvalidPolicy::Propagate
    };
    let reduction = match kind {
        "median" => Reduction::Median(policy),
        "median_network" => Reduction::NetworkMedian(policy),
        "median_histogram" => Reduction::HistogramMedian(policy),
        "mean" => Reduction::Mean(policy),
        "mean_rolling" => Reduction::RollingMean(policy),
        "min" => Reduction::Minimum(policy),
        "max" => Reduction::Maximum(policy),
        "var" => Reduction::Variance(policy, parameter),
        "std" => Reduction::Std(policy, parameter),
        "rank" => Reduction::Rank(policy, parameter),
        "rank_histogram" => Reduction::HistogramRank(policy, parameter),
        _ => return Err(PyValueError::new_err("unsupported window filter kind")),
    };
    let window = WindowSpec {
        size,
        footprint,
        origin,
    };
    if let Ok(typed) = array.extract::<PyReadonlyArrayDyn<'py, f32>>() {
        return run_filter(py, typed, window, mode, cval as f32, reduction, no_hooks())
            .map(|value| value.into_any());
    }
    let typed = array.extract::<PyReadonlyArrayDyn<'py, f64>>()?;
    run_filter(py, typed, window, mode, cval, reduction, no_hooks()).map(|value| value.into_any())
}

#[cfg(feature = "test-hooks")]
#[pyfunction(name = "_test_median_filter_after_snapshot", signature = (array, size, mode, cval, callback, *, mean=false))]
pub(crate) fn test_median_filter_after_snapshot<'py>(
    py: Python<'py>,
    array: PyReadonlyArrayDyn<'py, f64>,
    size: Vec<usize>,
    mode: Vec<String>,
    cval: f64,
    callback: &Bound<'py, PyAny>,
    mean: bool,
) -> PyResult<Py<PyAny>> {
    run_filter(
        py,
        array,
        WindowSpec::rectangular(size),
        mode,
        cval,
        if mean {
            Reduction::Mean(InvalidPolicy::Propagate)
        } else {
            Reduction::Median(InvalidPolicy::Propagate)
        },
        FilterHooks {
            after_snapshot: || callback.call0().map(|_| ()),
            during_detach: || Ok(()),
        },
    )
    .map(|value| value.into_any())
}

#[cfg(feature = "test-hooks")]
#[pyfunction(name = "_test_median_filter_during_detach", signature = (array, size, mode, cval, probe, *, mean=false))]
pub(crate) fn test_median_filter_during_detach<'py>(
    py: Python<'py>,
    array: PyReadonlyArrayDyn<'py, f64>,
    size: Vec<usize>,
    mode: Vec<String>,
    cval: f64,
    probe: PyRef<'py, DetachProbe>,
    mean: bool,
) -> PyResult<Py<PyAny>> {
    let probe_inner = Arc::clone(&probe.inner);
    run_filter(
        py,
        array,
        WindowSpec::rectangular(size),
        mode,
        cval,
        if mean {
            Reduction::Mean(InvalidPolicy::Propagate)
        } else {
            Reduction::Median(InvalidPolicy::Propagate)
        },
        FilterHooks {
            after_snapshot: || Ok(()),
            during_detach: move || wait_for_python_progress(&probe_inner, Duration::from_secs(5)),
        },
    )
    .map(|value| value.into_any())
}

#[pyfunction]
pub(crate) fn get_num_threads() -> usize {
    parallel::get_num_threads()
}
