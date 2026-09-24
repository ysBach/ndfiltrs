mod convolution;
mod filter;
mod repair;
mod robust;
mod separable;

use pyo3::prelude::*;

#[pymodule]
pub(crate) fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(convolution::convolve, module)?)?;
    module.add_function(wrap_pyfunction!(separable::separable_smooth, module)?)?;
    module.add_function(wrap_pyfunction!(robust::robust_filter, module)?)?;
    module.add_function(wrap_pyfunction!(repair::fixpix, module)?)?;
    module.add_function(wrap_pyfunction!(repair::fixpix_median, module)?)?;
    module.add_function(wrap_pyfunction!(repair::fixpix_mean, module)?)?;
    module.add_function(wrap_pyfunction!(filter::window_filter, module)?)?;
    module.add_function(wrap_pyfunction!(filter::get_num_threads, module)?)?;
    #[cfg(feature = "test-hooks")]
    {
        module.add_class::<filter::DetachProbe>()?;
        module.add_function(wrap_pyfunction!(
            convolution::test_convolve_after_snapshot,
            module
        )?)?;
        module.add_function(wrap_pyfunction!(
            convolution::test_convolve_during_detach,
            module
        )?)?;
        module.add_function(wrap_pyfunction!(
            repair::test_fixpix_after_snapshot,
            module
        )?)?;
        module.add_function(wrap_pyfunction!(repair::test_fixpix_during_detach, module)?)?;
        module.add_function(wrap_pyfunction!(
            filter::test_median_filter_after_snapshot,
            module
        )?)?;
        module.add_function(wrap_pyfunction!(
            filter::test_median_filter_during_detach,
            module
        )?)?;
    }
    Ok(())
}
