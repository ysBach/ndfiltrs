//! Exact separable sliding-window minimum and maximum for rectangular windows
//! (van Herk / Gil-Werman): about three comparisons per sample per axis, for any
//! window length. Results equal the direct window traversal bit for bit.
use rayon::prelude::*;
use reducers::Float;

use crate::boundary::{map_index, BoundaryMode};
use crate::error::KernelError;
use crate::reduction::InvalidPolicy;

#[derive(Clone, Copy)]
struct Summary<T> {
    value: T,
    nan: bool,
}

impl<T: Float> Summary<T> {
    fn empty() -> Self {
        Self {
            value: T::nan(),
            nan: false,
        }
    }

    fn sample(value: T, policy: InvalidPolicy) -> Self {
        match policy {
            InvalidPolicy::Propagate => Self {
                value,
                nan: value.is_nan(),
            },
            InvalidPolicy::Omit { ignore_inf, .. }
                if !value.is_nan() && (!ignore_inf || value.is_finite()) =>
            {
                Self { value, nan: false }
            }
            InvalidPolicy::Omit { .. } => Self::empty(),
        }
    }

    fn combine(self, other: Self, maximum: bool) -> Self {
        if self.nan || other.nan {
            return Self {
                value: T::nan(),
                nan: true,
            };
        }
        if self.value.is_nan() {
            return other;
        }
        if other.value.is_nan() {
            return self;
        }
        let ordering = self.value.total_cmp(other.value);
        let value = if (maximum && ordering == std::cmp::Ordering::Less)
            || (!maximum && ordering == std::cmp::Ordering::Greater)
        {
            other.value
        } else {
            self.value
        };
        Self { value, nan: false }
    }
}

struct LineWorkspace<T> {
    padded: Vec<Summary<T>>,
    prefix: Vec<Summary<T>>,
    suffix: Vec<Summary<T>>,
    padded_counts: Vec<usize>,
    prefix_counts: Vec<usize>,
    suffix_counts: Vec<usize>,
}

impl<T> Default for LineWorkspace<T> {
    fn default() -> Self {
        Self {
            padded: Vec::new(),
            prefix: Vec::new(),
            suffix: Vec::new(),
            padded_counts: Vec::new(),
            prefix_counts: Vec::new(),
            suffix_counts: Vec::new(),
        }
    }
}

fn checked_product(values: &[usize]) -> Result<usize, KernelError> {
    values.iter().try_fold(1usize, |product, &value| {
        product.checked_mul(value).ok_or(KernelError::Overflow)
    })
}

fn reserve<T>(length: usize) -> Result<Vec<T>, KernelError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| KernelError::Memory)?;
    Ok(values)
}

fn resize_summaries<T: Float>(
    values: &mut Vec<Summary<T>>,
    length: usize,
) -> Result<(), KernelError> {
    values.clear();
    if values.capacity() < length {
        values
            .try_reserve_exact(length)
            .map_err(|_| KernelError::Memory)?;
    }
    values.resize(length, Summary::empty());
    Ok(())
}

fn resize_counts(values: &mut Vec<usize>, length: usize) -> Result<(), KernelError> {
    values.clear();
    if values.capacity() < length {
        values
            .try_reserve_exact(length)
            .map_err(|_| KernelError::Memory)?;
    }
    values.resize(length, 0);
    Ok(())
}

/// Apply a rectangular minimum or maximum filter with exact separable passes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn extrema_into<T: Float + Send + Sync>(
    input: &[T],
    shape: &[usize],
    size: &[usize],
    origin: &[isize],
    modes: &[BoundaryMode],
    cval: T,
    maximum: bool,
    policy: InvalidPolicy,
    output: &mut [T],
) -> Result<(), KernelError> {
    let rank = shape.len();
    if rank == 0 {
        return Err(KernelError::InvalidConfiguration(
            "filtering requires at least one dimension",
        ));
    }
    if size.len() != rank || origin.len() != rank || modes.len() != rank {
        return Err(KernelError::InvalidConfiguration(
            "array, window, origin, and mode must have equal rank",
        ));
    }
    if size.contains(&0) {
        return Err(KernelError::InvalidConfiguration(
            "window sizes must be positive",
        ));
    }
    let volume = checked_product(size)?;
    let mut anchors = reserve::<isize>(rank)?;
    for (&window_length, &shift) in size.iter().zip(origin) {
        let window_length = isize::try_from(window_length).map_err(|_| KernelError::Overflow)?;
        if shift < -(window_length / 2) || shift > (window_length - 1) / 2 {
            return Err(KernelError::InvalidConfiguration(
                "origin must satisfy -(size // 2) <= origin <= (size - 1) // 2",
            ));
        }
        anchors.push(window_length / 2 + shift);
    }
    if let InvalidPolicy::Omit { min_count, .. } = policy {
        if min_count == 0 || min_count > volume {
            return Err(KernelError::InvalidConfiguration(
                "min_count must be between 1 and the number of window elements",
            ));
        }
    }

    let output_len = checked_product(shape)?;
    let mut strides = reserve::<usize>(rank)?;
    strides.resize(rank, 1);
    for axis in (0..rank - 1).rev() {
        strides[axis] = strides[axis + 1]
            .checked_mul(shape[axis + 1])
            .ok_or(KernelError::Overflow)?;
    }
    if input.len() != output_len || output.len() != output_len {
        return Err(KernelError::LengthMismatch);
    }
    if output_len == 0 {
        return Ok(());
    }

    let mut previous = reserve::<Summary<T>>(output_len)?;
    previous.extend(
        input
            .iter()
            .copied()
            .map(|value| Summary::sample(value, policy)),
    );
    let mut next = reserve::<Summary<T>>(output_len)?;
    next.resize(output_len, Summary::empty());
    let mut line_results = reserve::<Summary<T>>(output_len)?;
    line_results.resize(output_len, Summary::empty());
    // An empty summary already represents zero retained samples. Counts add
    // information only when more than one retained sample is required.
    let count_tracking = matches!(policy, InvalidPolicy::Omit { min_count, .. } if min_count > 1);
    let mut previous_counts = if count_tracking {
        let mut counts = reserve::<usize>(output_len)?;
        counts.extend(
            input
                .iter()
                .map(|&value| usize::from(retained(value, policy))),
        );
        Some(counts)
    } else {
        None
    };
    let mut next_counts = if count_tracking {
        let mut counts = reserve::<usize>(output_len)?;
        counts.resize(output_len, 0);
        Some(counts)
    } else {
        None
    };
    let mut line_count_results = if count_tracking {
        let mut counts = reserve::<usize>(output_len)?;
        counts.resize(output_len, 0);
        Some(counts)
    } else {
        None
    };

    let mut prior_volume = 1usize;
    for axis in 0..rank {
        let axis_length = shape[axis];
        let window_length = size[axis];
        let stride = strides[axis];
        let padded_length = axis_length
            .checked_add(window_length)
            .and_then(|length| length.checked_sub(1))
            .ok_or(KernelError::Overflow)?;
        let line_count = output_len / axis_length;
        let axis_block = axis_length
            .checked_mul(stride)
            .ok_or(KernelError::Overflow)?;

        if let (Some(previous_counts), Some(line_count_results)) =
            (previous_counts.as_ref(), line_count_results.as_mut())
        {
            line_results
                .par_chunks_mut(axis_length)
                .zip(line_count_results.par_chunks_mut(axis_length))
                .enumerate()
                .try_for_each_init(
                    LineWorkspace::default,
                    |workspace,
                     (line_index, (destinations, count_destinations))|
                     -> Result<(), KernelError> {
                        resize_summaries(&mut workspace.padded, padded_length)?;
                        resize_summaries(&mut workspace.prefix, padded_length)?;
                        resize_summaries(&mut workspace.suffix, padded_length)?;
                        resize_counts(&mut workspace.padded_counts, padded_length)?;
                        resize_counts(&mut workspace.prefix_counts, padded_length)?;
                        resize_counts(&mut workspace.suffix_counts, padded_length)?;

                        let outer_index = line_index / stride;
                        let inner_index = line_index % stride;
                        let base = outer_index * axis_block + inner_index;
                        for (position, sample) in workspace.padded.iter_mut().enumerate() {
                            let coordinate = i128::try_from(position)
                                .map_err(|_| KernelError::Overflow)?
                                .checked_sub(anchors[axis] as i128)
                                .ok_or(KernelError::Overflow)?;
                            let mapped = map_index(coordinate, axis_length, modes[axis])?;
                            *sample = match mapped {
                                Some(mapped) => previous[base + mapped * stride],
                                None => Summary::sample(cval, policy),
                            };
                            workspace.padded_counts[position] = match mapped {
                                Some(mapped) => previous_counts[base + mapped * stride],
                                None => prior_volume
                                    .checked_mul(usize::from(retained(cval, policy)))
                                    .ok_or(KernelError::Overflow)?,
                            };
                        }

                        for position in 0..padded_length {
                            workspace.prefix[position] = if position % window_length == 0 {
                                workspace.padded[position]
                            } else {
                                workspace.prefix[position - 1]
                                    .combine(workspace.padded[position], maximum)
                            };
                            workspace.prefix_counts[position] = if position % window_length == 0 {
                                workspace.padded_counts[position]
                            } else {
                                workspace.prefix_counts[position - 1]
                                    .checked_add(workspace.padded_counts[position])
                                    .ok_or(KernelError::Overflow)?
                            };
                        }
                        for position in (0..padded_length).rev() {
                            workspace.suffix[position] = if position == padded_length - 1
                                || position % window_length == window_length - 1
                            {
                                workspace.padded[position]
                            } else {
                                workspace.padded[position]
                                    .combine(workspace.suffix[position + 1], maximum)
                            };
                            workspace.suffix_counts[position] = if position == padded_length - 1
                                || position % window_length == window_length - 1
                            {
                                workspace.padded_counts[position]
                            } else {
                                workspace.suffix_counts[position + 1]
                                    .checked_add(workspace.padded_counts[position])
                                    .ok_or(KernelError::Overflow)?
                            };
                        }
                        for (position, destination) in destinations.iter_mut().enumerate() {
                            *destination = workspace.suffix[position]
                                .combine(workspace.prefix[position + window_length - 1], maximum);
                            let end = position + window_length - 1;
                            count_destinations[position] =
                                if position / window_length == end / window_length {
                                    workspace.prefix_counts[end]
                                        .checked_sub(workspace.prefix_counts[position])
                                        .and_then(|count| {
                                            count.checked_add(workspace.padded_counts[position])
                                        })
                                        .ok_or(KernelError::Overflow)?
                                } else {
                                    workspace.suffix_counts[position]
                                        .checked_add(workspace.prefix_counts[end])
                                        .ok_or(KernelError::Overflow)?
                                };
                        }
                        Ok(())
                    },
                )?;
        } else {
            line_results
                .par_chunks_mut(axis_length)
                .enumerate()
                .try_for_each_init(
                    LineWorkspace::default,
                    |workspace, (line_index, destinations)| -> Result<(), KernelError> {
                        resize_summaries(&mut workspace.padded, padded_length)?;
                        resize_summaries(&mut workspace.prefix, padded_length)?;
                        resize_summaries(&mut workspace.suffix, padded_length)?;

                        let outer_index = line_index / stride;
                        let inner_index = line_index % stride;
                        let base = outer_index * axis_block + inner_index;
                        for (position, sample) in workspace.padded.iter_mut().enumerate() {
                            let coordinate = i128::try_from(position)
                                .map_err(|_| KernelError::Overflow)?
                                .checked_sub(anchors[axis] as i128)
                                .ok_or(KernelError::Overflow)?;
                            *sample = match map_index(coordinate, axis_length, modes[axis])? {
                                Some(mapped) => previous[base + mapped * stride],
                                None => Summary::sample(cval, policy),
                            };
                        }

                        for position in 0..padded_length {
                            workspace.prefix[position] = if position % window_length == 0 {
                                workspace.padded[position]
                            } else {
                                workspace.prefix[position - 1]
                                    .combine(workspace.padded[position], maximum)
                            };
                        }
                        for position in (0..padded_length).rev() {
                            workspace.suffix[position] = if position == padded_length - 1
                                || position % window_length == window_length - 1
                            {
                                workspace.padded[position]
                            } else {
                                workspace.padded[position]
                                    .combine(workspace.suffix[position + 1], maximum)
                            };
                        }
                        for (position, destination) in destinations.iter_mut().enumerate() {
                            *destination = workspace.suffix[position]
                                .combine(workspace.prefix[position + window_length - 1], maximum);
                        }
                        Ok(())
                    },
                )?;
        }

        debug_assert_eq!(line_count, line_results.len() / axis_length);
        next.par_iter_mut()
            .enumerate()
            .for_each(|(flat, destination)| {
                let outer_index = flat / axis_block;
                let coordinate = (flat / stride) % axis_length;
                let inner_index = flat % stride;
                let line_index = outer_index * stride + inner_index;
                *destination = line_results[line_index * axis_length + coordinate];
            });
        if let (Some(next_counts), Some(line_count_results)) =
            (next_counts.as_mut(), line_count_results.as_ref())
        {
            next_counts
                .par_iter_mut()
                .enumerate()
                .for_each(|(flat, destination)| {
                    let outer_index = flat / axis_block;
                    let coordinate = (flat / stride) % axis_length;
                    let inner_index = flat % stride;
                    let line_index = outer_index * stride + inner_index;
                    *destination = line_count_results[line_index * axis_length + coordinate];
                });
        }
        std::mem::swap(&mut previous, &mut next);
        std::mem::swap(&mut previous_counts, &mut next_counts);
        prior_volume = prior_volume
            .checked_mul(window_length)
            .ok_or(KernelError::Overflow)?;
    }

    match policy {
        InvalidPolicy::Propagate | InvalidPolicy::Omit { min_count: 1, .. } => output
            .par_iter_mut()
            .zip(previous.into_par_iter())
            .for_each(|(destination, summary)| {
                *destination = if summary.nan { T::nan() } else { summary.value };
            }),
        InvalidPolicy::Omit { min_count, .. } => {
            let counts = previous_counts.ok_or(KernelError::Overflow)?;
            output
                .par_iter_mut()
                .zip(previous.into_par_iter())
                .zip(counts.into_par_iter())
                .for_each(|((destination, summary), count)| {
                    *destination = if summary.nan || count < min_count {
                        T::nan()
                    } else {
                        summary.value
                    };
                });
        }
    }
    Ok(())
}

fn retained<T: Float>(value: T, policy: InvalidPolicy) -> bool {
    match policy {
        InvalidPolicy::Propagate => true,
        InvalidPolicy::Omit { ignore_inf, .. } => {
            !value.is_nan() && (!ignore_inf || value.is_finite())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use reducers::Float;

    use crate::boundary::BoundaryMode::{self, Constant, Mirror, Nearest, Reflect, Wrap};
    use crate::filter_nd::{filter_nd_serial, FilterConfig};
    use crate::reduction::{InvalidPolicy, Reduction};
    use crate::window::Window;

    use super::extrema_into;

    trait TestFloat: Float + crate::median_network::NetworkFloat + Send + Sync {
        fn from_bits(bits: u64) -> Self;
        fn bits(self) -> u64;
        fn finite(raw: u64) -> Self;
    }

    impl TestFloat for f32 {
        fn from_bits(bits: u64) -> Self {
            f32::from_bits(bits as u32)
        }

        fn bits(self) -> u64 {
            u64::from(self.to_bits())
        }

        fn finite(raw: u64) -> Self {
            ((raw as i64 % 20001) as f64 / 31.0) as f32
        }
    }

    impl TestFloat for f64 {
        fn from_bits(bits: u64) -> Self {
            f64::from_bits(bits)
        }

        fn bits(self) -> u64 {
            self.to_bits()
        }

        fn finite(raw: u64) -> Self {
            raw as i64 as f64 / 31.0
        }
    }

    #[derive(Clone, Copy)]
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn value<T: TestFloat>(rng: &mut Rng, index: usize) -> T {
        match index % 8 {
            0 => T::from_bits(0),
            1 => T::from_bits(if std::mem::size_of::<T>() == 4 {
                u64::from((-0.0_f32).to_bits())
            } else {
                (-0.0_f64).to_bits()
            }),
            2 => T::from_bits(if std::mem::size_of::<T>() == 4 {
                u64::from(f32::INFINITY.to_bits())
            } else {
                f64::INFINITY.to_bits()
            }),
            3 => T::from_bits(if std::mem::size_of::<T>() == 4 {
                u64::from(f32::NEG_INFINITY.to_bits())
            } else {
                f64::NEG_INFINITY.to_bits()
            }),
            4 => T::from_bits(if std::mem::size_of::<T>() == 4 {
                0x7fc0_0042
            } else {
                0x7ff8_0000_0000_0042
            }),
            5 => T::from_bits(if std::mem::size_of::<T>() == 4 {
                0xffc0_0123
            } else {
                0xfff8_0000_0000_0123
            }),
            _ => T::finite(rng.next()),
        }
    }

    fn compare_case<T: TestFloat>(
        input: &[T],
        shape: &[usize],
        size: &[usize],
        origin: &[isize],
        modes: &[BoundaryMode],
        cval: T,
        policy: InvalidPolicy,
    ) {
        let window = Window::new(shape.to_vec(), size, None, origin, modes.to_vec()).unwrap();
        let config = FilterConfig::from_window(window, cval, Reduction::Minimum(policy)).unwrap();
        let expected = filter_nd_serial(input, &config).unwrap();
        let mut actual = vec![T::nan(); input.len()];
        extrema_into(
            input,
            shape,
            size,
            origin,
            modes,
            cval,
            false,
            policy,
            &mut actual,
        )
        .unwrap();
        assert_eq!(
            actual.iter().map(|value| value.bits()).collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.bits())
                .collect::<Vec<_>>(),
            "shape={shape:?}, size={size:?}, origin={origin:?}, modes={modes:?}, policy={policy:?}"
        );

        let expected = filter_nd_serial(
            input,
            &FilterConfig::from_window(
                Window::new(shape.to_vec(), size, None, origin, modes.to_vec()).unwrap(),
                cval,
                Reduction::Maximum(policy),
            )
            .unwrap(),
        )
        .unwrap();
        extrema_into(
            input,
            shape,
            size,
            origin,
            modes,
            cval,
            true,
            policy,
            &mut actual,
        )
        .unwrap();
        assert_eq!(
            actual.iter().map(|value| value.bits()).collect::<Vec<_>>(),
            expected.iter().map(|value| value.bits()).collect::<Vec<_>>(),
            "maximum: shape={shape:?}, size={size:?}, origin={origin:?}, modes={modes:?}, policy={policy:?}"
        );
    }

    fn policies(volume: usize) -> [InvalidPolicy; 5] {
        [
            InvalidPolicy::Propagate,
            InvalidPolicy::Omit {
                ignore_inf: false,
                min_count: 1,
            },
            InvalidPolicy::Omit {
                ignore_inf: false,
                min_count: volume,
            },
            InvalidPolicy::Omit {
                ignore_inf: true,
                min_count: 1,
            },
            InvalidPolicy::Omit {
                ignore_inf: true,
                min_count: volume,
            },
        ]
    }

    fn cval<T: TestFloat>(index: usize) -> T {
        match index % 4 {
            0 => T::finite(431),
            1 => T::from_bits(if std::mem::size_of::<T>() == 4 {
                u64::from(f32::INFINITY.to_bits())
            } else {
                f64::INFINITY.to_bits()
            }),
            2 => T::from_bits(if std::mem::size_of::<T>() == 4 {
                u64::from(f32::NEG_INFINITY.to_bits())
            } else {
                f64::NEG_INFINITY.to_bits()
            }),
            _ => T::from_bits(if std::mem::size_of::<T>() == 4 {
                0xffc0_0123
            } else {
                0xfff8_0000_0000_0123
            }),
        }
    }

    fn exercise<T: TestFloat>() {
        let all_modes = [Reflect, Constant, Nearest, Mirror, Wrap];
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut case_index = 0usize;

        // Exhaust every legal one-dimensional size, origin, mode, and short axis.
        for axis_length in 1..=5 {
            for window_length in 1..=6 {
                let low = -(window_length as isize / 2);
                let high = (window_length as isize - 1) / 2;
                for origin in low..=high {
                    for &mode in &all_modes {
                        let shape = [axis_length];
                        let size = [window_length];
                        let modes = [mode];
                        for policy in policies(window_length) {
                            for cval_index in 0..4 {
                                let mut input = Vec::with_capacity(axis_length);
                                for index in 0..axis_length {
                                    input.push(value::<T>(&mut rng, index + case_index));
                                }
                                compare_case(
                                    &input,
                                    &shape,
                                    &size,
                                    &[origin],
                                    &modes,
                                    cval(case_index + cval_index),
                                    policy,
                                );
                                case_index += 1;
                            }
                        }
                    }
                }
            }
        }

        // Check every origin tuple for equal-sided windows in ranks two and three.
        for (shape, rank) in [([1usize, 2, 1], 2usize), ([1usize, 1, 2], 3usize)] {
            for window_length in 1..=6 {
                let low = -(window_length as isize / 2);
                let high = (window_length as isize - 1) / 2;
                let mut origins = vec![low; rank];
                let mut cursor = vec![low; rank];
                loop {
                    let selected_shape = &shape[..rank];
                    let size = vec![window_length; rank];
                    let mut input = Vec::with_capacity(selected_shape.iter().product());
                    for index in 0..selected_shape.iter().product() {
                        input.push(value::<T>(&mut rng, index + case_index));
                    }
                    for mode in all_modes {
                        let modes = vec![mode; rank];
                        for policy in policies(size.iter().product()) {
                            compare_case(
                                &input,
                                selected_shape,
                                &size,
                                &origins,
                                &modes,
                                cval(case_index),
                                policy,
                            );
                            case_index += 1;
                        }
                    }

                    let mut axis = rank;
                    let mut advanced = false;
                    while axis > 0 {
                        axis -= 1;
                        if cursor[axis] < high {
                            cursor[axis] += 1;
                            for reset in cursor.iter_mut().skip(axis + 1) {
                                *reset = low;
                            }
                            advanced = true;
                            break;
                        }
                    }
                    if !advanced {
                        break;
                    }
                    origins.copy_from_slice(&cursor);
                }
            }
        }

        // Random mixed-axis geometry exercises combinations absent from the exhaustive sets.
        for rank in 1..=3 {
            for _ in 0..180 {
                let shape = (0..rank)
                    .map(|_| 1 + (rng.next() as usize % 3))
                    .collect::<Vec<_>>();
                let size = (0..rank)
                    .map(|_| 1 + (rng.next() as usize % 6))
                    .collect::<Vec<_>>();
                let origin = size
                    .iter()
                    .map(|&length| {
                        let low = -(length as isize / 2);
                        let choices = length;
                        low + (rng.next() as usize % choices) as isize
                    })
                    .collect::<Vec<_>>();
                let modes = (0..rank)
                    .map(|_| all_modes[rng.next() as usize % all_modes.len()])
                    .collect::<Vec<_>>();
                let length = shape.iter().product();
                let mut input = Vec::with_capacity(length);
                for index in 0..length {
                    input.push(value::<T>(&mut rng, index + case_index));
                }
                let volume = size.iter().product();
                let policy_set = policies(volume);
                compare_case(
                    &input,
                    &shape,
                    &size,
                    &origin,
                    &modes,
                    cval(case_index),
                    policy_set[case_index % policy_set.len()],
                );
                case_index += 1;
            }
        }

        for (shape, size) in [
            (&[0][..], &[3][..]),
            (&[0, 2][..], &[4, 2][..]),
            (&[1, 0, 2][..], &[2, 3, 4][..]),
        ] {
            let origin = vec![0; shape.len()];
            let modes = vec![Constant; shape.len()];
            compare_case(
                &[],
                shape,
                size,
                &origin,
                &modes,
                cval::<T>(case_index),
                InvalidPolicy::Omit {
                    ignore_inf: true,
                    min_count: 1,
                },
            );
            case_index += 1;
        }
    }

    #[test]
    fn exhaustive_and_random_comparisons_match_f32_direct_path() {
        exercise::<f32>();
    }

    #[test]
    fn exhaustive_and_random_comparisons_match_f64_direct_path() {
        exercise::<f64>();
    }

    #[test]
    fn invalid_geometry_and_lengths_return_kernel_errors() {
        let mut output = [0.0_f64; 2];
        let error = extrema_into(
            &[1.0, 2.0],
            &[2],
            &[3],
            &[2],
            &[Reflect],
            0.0,
            false,
            InvalidPolicy::Propagate,
            &mut output,
        );
        assert!(matches!(
            error,
            Err(crate::error::KernelError::InvalidConfiguration(_))
        ));
        assert!(extrema_into(
            &[1.0, 2.0],
            &[2],
            &[3],
            &[0],
            &[Reflect],
            0.0,
            false,
            InvalidPolicy::Omit {
                ignore_inf: false,
                min_count: 4
            },
            &mut output,
        )
        .is_err());
        assert!(extrema_into(
            &[1.0],
            &[2],
            &[3],
            &[0],
            &[Reflect],
            0.0,
            false,
            InvalidPolicy::Propagate,
            &mut output,
        )
        .is_err());
    }

    #[test]
    #[ignore = "manual release timing comparison for a 2048 by 2048 array"]
    fn timing_2048_square_size_15_matches_direct_path() {
        let shape = [2048, 2048];
        let size = [15, 15];
        let origin = [0, 0];
        let modes = [Reflect, Reflect];
        let input = (0..shape.iter().product())
            .map(|index| ((index * 104729 % 1_000_003) as f64) / 997.0)
            .collect::<Vec<_>>();
        let config = FilterConfig::from_window(
            Window::new(shape.to_vec(), &size, None, &origin, modes.to_vec()).unwrap(),
            f64::INFINITY,
            Reduction::Minimum(InvalidPolicy::Propagate),
        )
        .unwrap();
        let started = Instant::now();
        let expected = filter_nd_serial(&input, &config).unwrap();
        let direct_elapsed = started.elapsed();
        let mut actual = vec![0.0; input.len()];
        let started = Instant::now();
        extrema_into(
            &input,
            &shape,
            &size,
            &origin,
            &modes,
            f64::INFINITY,
            false,
            InvalidPolicy::Propagate,
            &mut actual,
        )
        .unwrap();
        let extrema_elapsed = started.elapsed();
        assert!(actual
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
        eprintln!("2048x2048 size=15: direct={direct_elapsed:?}, extrema={extrema_elapsed:?}");
    }
}
