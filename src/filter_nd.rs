#[cfg(any(test, feature = "python"))]
use rayon::prelude::*;
use reducers::Float;

#[cfg(any(test, feature = "benchmark", feature = "test-hooks"))]
use crate::boundary::BoundaryMode;
use crate::error::KernelError;
use crate::median_network::NetworkFloat;
use crate::reduction::{InvalidPolicy, Reduction};
use crate::window::Window;

#[derive(Clone, Debug)]
pub(crate) struct FilterConfig<T> {
    window: Window,
    cval: T,
    reduction: Reduction,
    median_network: Option<crate::median_network::Network>,
}

impl<T: Copy> FilterConfig<T> {
    /// Configure a centered rectangular window filter.
    #[cfg(any(test, feature = "benchmark", feature = "test-hooks"))]
    pub(crate) fn new(
        shape: Vec<usize>,
        size: Vec<usize>,
        mode: Vec<BoundaryMode>,
        cval: T,
        reduction: Reduction,
    ) -> Result<Self, KernelError> {
        Self::from_window(Window::rectangular(shape, &size, mode)?, cval, reduction)
    }

    pub(crate) fn from_window(
        window: Window,
        cval: T,
        reduction: Reduction,
    ) -> Result<Self, KernelError> {
        if let InvalidPolicy::Omit { min_count, .. } = reduction.policy() {
            if min_count == 0 || min_count > window.count() {
                return Err(KernelError::InvalidConfiguration(
                    "min_count must be between 1 and the number of window elements",
                ));
            }
        }
        let median_network = if let Reduction::NetworkMedian(policy) = reduction {
            Some(crate::median_network::Network::new(window.count(), policy)?)
        } else {
            None
        };
        Ok(Self {
            window,
            cval,
            reduction,
            median_network,
        })
    }

    pub(crate) fn output_len(&self) -> usize {
        self.window.output_len()
    }
}

pub(crate) fn filter_range<T: NetworkFloat>(
    input: &[T],
    config: &FilterConfig<T>,
    start: usize,
    output: &mut [T],
) -> Result<(), KernelError> {
    let end = start
        .checked_add(output.len())
        .ok_or(KernelError::Overflow)?;
    if input.len() != config.output_len() || end > config.output_len() {
        return Err(KernelError::LengthMismatch);
    }
    let Some(network) = &config.median_network else {
        return filter_offsets(input, config, start..end, output);
    };
    let mut coordinates = vec![0; config.window.rank()];
    let mut retained = Vec::new();
    let mut keys = Vec::new();
    let mut offset = start;
    while offset < end {
        let (interior, width) =
            config
                .window
                .row_run(input.len(), offset, end - offset, &mut coordinates)?;
        if !interior {
            filter_offsets(
                input,
                config,
                offset..offset + width,
                &mut output[offset - start..offset - start + width],
            )?;
            offset += width;
            continue;
        }
        let length = network
            .wires()
            .checked_mul(width)
            .ok_or(KernelError::Overflow)?;
        keys.try_reserve(length.saturating_sub(keys.len()))
            .map_err(|_| KernelError::Memory)?;
        keys.resize(length, T::nan().sort_key());
        retained
            .try_reserve(width.saturating_sub(retained.len()))
            .map_err(|_| KernelError::Memory)?;
        retained.resize(width, 0);
        retained.fill(0);
        config
            .window
            .visit_interior_planes(input, offset, width, |tap, values| {
                crate::median_network::encode_plane(
                    values,
                    &mut keys[tap * width..(tap + 1) * width],
                    config.reduction.policy(),
                    &mut retained,
                );
            });
        network.apply(
            &mut keys,
            config.window.count(),
            config.reduction.policy(),
            &retained,
            &mut output[offset - start..offset - start + width],
        );
        offset += width;
    }
    Ok(())
}

fn filter_offsets<T: Float>(
    input: &[T],
    config: &FilterConfig<T>,
    offsets: impl ExactSizeIterator<Item = usize>,
    output: &mut [T],
) -> Result<(), KernelError> {
    if input.len() != config.output_len() || offsets.len() != output.len() {
        return Err(KernelError::LengthMismatch);
    }
    let mut index = vec![0; config.window.rank()];
    let mut scratch = Vec::new();
    if !matches!(config.reduction, Reduction::Mean(_)) {
        scratch
            .try_reserve_exact(config.window.count())
            .map_err(|_| KernelError::Memory)?;
    }
    for (offset, destination) in offsets.zip(output.iter_mut()) {
        if let Reduction::Mean(policy) = config.reduction {
            let mut mean = crate::mean::Mean::default();
            config
                .window
                .visit(input, offset, config.cval, &mut index, |value| {
                    mean.add(value, policy);
                })?;
            if let Some(result) = mean.finish(policy) {
                *destination = result;
                continue;
            }
            // Only an overflowing finite sum needs the buffered scaling fallback.
            if scratch.capacity() < config.window.count() {
                scratch
                    .try_reserve_exact(config.window.count())
                    .map_err(|_| KernelError::Memory)?;
            }
        }
        scratch.clear();
        config.window.gather(
            input,
            offset,
            config.cval,
            &mut index,
            &mut scratch,
            |value| value,
        )?;
        *destination = config.reduction.finish(&mut scratch);
    }
    Ok(())
}

/// Evaluate only masked windows, excluding every originally masked sample.
#[cfg(any(test, feature = "python"))]
pub(crate) fn repair_median<T: Float + Send + Sync>(
    data: &mut [T],
    config: &FilterConfig<T>,
    positions: &[usize],
) -> Result<(), KernelError> {
    if data.len() != config.output_len()
        || positions.last().is_some_and(|&last| last >= data.len())
        || positions.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(KernelError::LengthMismatch);
    }
    if positions.is_empty() {
        return Ok(());
    }
    let mut estimates = Vec::new();
    estimates
        .try_reserve_exact(positions.len())
        .map_err(|_| KernelError::Memory)?;
    estimates.resize(positions.len(), T::nan());
    // The owned output doubles as the snapshot. All estimates are computed
    // before any are scattered, so repaired values cannot enter a window.
    for &offset in positions {
        data[offset] = T::nan();
    }
    // Each estimate reads only the snapshot, so splitting positions across
    // threads cannot change any result.
    let snapshot: &[T] = data;
    let chunk = positions
        .len()
        .div_ceil(rayon::current_num_threads().max(1));
    estimates
        .par_chunks_mut(chunk)
        .zip(positions.par_chunks(chunk))
        .try_for_each(|(destination, offsets)| {
            filter_offsets(snapshot, config, offsets.iter().copied(), destination)
        })?;
    for (&offset, value) in positions.iter().zip(estimates) {
        data[offset] = value;
    }
    Ok(())
}

#[cfg(any(test, feature = "benchmark"))]
pub(crate) fn filter_nd_serial<T: NetworkFloat>(
    input: &[T],
    config: &FilterConfig<T>,
) -> Result<Vec<T>, KernelError> {
    if input.len() != config.output_len() {
        return Err(KernelError::LengthMismatch);
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(config.output_len())
        .map_err(|_| KernelError::Memory)?;
    output.resize(config.output_len(), T::nan());
    if config.output_len() > 0 {
        filter_range(input, config, 0, &mut output)?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use crate::boundary::BoundaryMode::{Constant, Nearest, Reflect, Wrap};
    use crate::reduction::{InvalidPolicy, Reduction};

    use super::{filter_nd_serial, repair_median, FilterConfig};

    #[test]
    fn mean_uses_shared_windows_and_omission_policy() {
        let config = FilterConfig::new(
            vec![3],
            vec![3],
            vec![Nearest],
            0.0_f64,
            Reduction::Mean(InvalidPolicy::Omit {
                ignore_inf: true,
                min_count: 2,
            }),
        )
        .unwrap();
        let result = filter_nd_serial(&[1.0, f64::NAN, 3.0], &config).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn sparse_repair_never_uses_another_repaired_value() {
        let config = FilterConfig::new(
            vec![4],
            vec![3],
            vec![Constant],
            f64::NAN,
            Reduction::Median(InvalidPolicy::Omit {
                ignore_inf: false,
                min_count: 1,
            }),
        )
        .unwrap();
        let mut data = [1.0, 99.0, 99.0, 9.0];
        repair_median(&mut data, &config, &[1, 2]).unwrap();
        assert_eq!(data, [1.0, 1.0, 9.0, 9.0]);
        assert!(repair_median(&mut data, &config, &[4]).is_err());
        assert!(repair_median(&mut data, &config, &[2, 1]).is_err());
    }

    #[test]
    fn serial_kernel_filters_one_dimensional_values() {
        let config = FilterConfig::new(
            vec![3],
            vec![3],
            vec![Nearest],
            0.0_f64,
            Reduction::Median(InvalidPolicy::Propagate),
        )
        .unwrap();
        assert_eq!(
            filter_nd_serial(&[1.0, 9.0, 2.0], &config).unwrap(),
            vec![1.0, 2.0, 2.0]
        );
    }

    #[test]
    fn serial_kernel_filters_two_dimensional_constant_boundaries() {
        let config = FilterConfig::new(
            vec![2, 2],
            vec![3, 3],
            vec![Constant, Constant],
            10.0_f32,
            Reduction::Median(InvalidPolicy::Propagate),
        )
        .unwrap();
        assert_eq!(
            filter_nd_serial(&[1.0, 2.0, 3.0, 4.0], &config).unwrap(),
            vec![10.0_f32; 4]
        );
    }

    #[test]
    fn serial_kernel_preserves_size_one_axes_in_three_dimensions() {
        let config = FilterConfig::new(
            vec![3, 1, 1],
            vec![3, 1, 1],
            vec![Nearest, Reflect, Wrap],
            0.0_f64,
            Reduction::Median(InvalidPolicy::Propagate),
        )
        .unwrap();
        assert_eq!(
            filter_nd_serial(&[1.0, 9.0, 2.0], &config).unwrap(),
            vec![1.0, 2.0, 2.0]
        );
    }

    #[test]
    fn serial_kernel_applies_omit_policy_and_min_count() {
        let config = FilterConfig::new(
            vec![3],
            vec![3],
            vec![Nearest],
            0.0_f64,
            Reduction::Median(InvalidPolicy::Omit {
                ignore_inf: true,
                min_count: 2,
            }),
        )
        .unwrap();
        let result = filter_nd_serial(&[1.0, f64::NAN, f64::INFINITY], &config).unwrap();
        assert_eq!(result[0], 1.0);
        assert!(result[1].is_nan());
        assert!(result[2].is_nan());
    }

    #[test]
    fn serial_kernel_returns_owned_empty_output() {
        let config = FilterConfig::new(
            vec![0, 3],
            vec![3, 3],
            vec![Reflect, Reflect],
            0.0_f64,
            Reduction::Median(InvalidPolicy::Propagate),
        )
        .unwrap();
        assert_eq!(filter_nd_serial(&[], &config).unwrap(), Vec::<f64>::new());
    }

    #[test]
    fn configuration_rejects_invalid_geometry() {
        assert!(FilterConfig::new(
            vec![],
            vec![],
            vec![],
            0.0_f64,
            Reduction::Median(InvalidPolicy::Propagate),
        )
        .is_err());
        assert!(FilterConfig::new(
            vec![3, 3],
            vec![3],
            vec![Reflect, Reflect],
            0.0_f64,
            Reduction::Median(InvalidPolicy::Propagate),
        )
        .is_err());
        assert!(FilterConfig::new(
            vec![3],
            vec![0],
            vec![Reflect],
            0.0_f64,
            Reduction::Median(InvalidPolicy::Propagate),
        )
        .is_err());
    }

    #[test]
    fn configuration_rejects_min_count_outside_window_volume() {
        for min_count in [0, 4] {
            assert!(FilterConfig::new(
                vec![3],
                vec![3],
                vec![Reflect],
                0.0_f64,
                Reduction::Median(InvalidPolicy::Omit {
                    ignore_inf: false,
                    min_count,
                }),
            )
            .is_err());
        }
    }
}
