//! Benchmark-only comparison surface; excluded from production wheels.

use reducers::Float;

use crate::boundary::{map_index, BoundaryMode};
use crate::error::KernelError;
use crate::filter_nd::{filter_nd_serial, FilterConfig};
use crate::reduction::{InvalidPolicy, Reduction};

/// Value policy used by the benchmark-only implementations.
#[derive(Clone, Copy, Debug)]
pub enum BenchmarkPolicy {
    /// Propagate NaNs and retain infinities.
    Propagate,
    /// Omit NaNs and optionally infinities, enforcing a retained count.
    Omit { ignore_inf: bool, min_count: usize },
}

fn modes(names: [&str; 2]) -> Result<Vec<BoundaryMode>, KernelError> {
    names
        .into_iter()
        .map(|name| match name {
            "reflect" => Ok(BoundaryMode::Reflect),
            "constant" => Ok(BoundaryMode::Constant),
            "nearest" => Ok(BoundaryMode::Nearest),
            "mirror" => Ok(BoundaryMode::Mirror),
            "wrap" => Ok(BoundaryMode::Wrap),
            _ => Err(KernelError::InvalidConfiguration("unsupported mode")),
        })
        .collect()
}

fn policy(value: BenchmarkPolicy) -> InvalidPolicy {
    match value {
        BenchmarkPolicy::Propagate => InvalidPolicy::Propagate,
        BenchmarkPolicy::Omit {
            ignore_inf,
            min_count,
        } => InvalidPolicy::Omit {
            ignore_inf,
            min_count,
        },
    }
}

/// Run rank-two input through the production generic N-D traversal.
pub fn generic_2d<T: crate::median_network::NetworkFloat>(
    input: &[T],
    shape: [usize; 2],
    size: [usize; 2],
    mode: [&str; 2],
    cval: T,
    value_policy: BenchmarkPolicy,
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    let config = FilterConfig::new(
        shape.to_vec(),
        size.to_vec(),
        modes(mode)?,
        cval,
        Reduction::Median(policy(value_policy)),
    )?;
    Ok(filter_nd_serial(input, &config)?)
}

/// Benchmark-only specialized rank-two candidate.
pub fn specialized_2d<T: Float>(
    input: &[T],
    shape: [usize; 2],
    size: [usize; 2],
    mode: [&str; 2],
    cval: T,
    value_policy: BenchmarkPolicy,
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    if size.iter().any(|&length| length == 0 || length % 2 == 0) {
        return Err(Box::new(KernelError::InvalidConfiguration(
            "window sizes must be positive odd integers",
        )));
    }
    let output_len = shape[0]
        .checked_mul(shape[1])
        .ok_or(KernelError::Overflow)?;
    if input.len() != output_len {
        return Err(Box::new(KernelError::LengthMismatch));
    }
    let window_volume = size[0].checked_mul(size[1]).ok_or(KernelError::Overflow)?;
    if let BenchmarkPolicy::Omit { min_count, .. } = value_policy {
        if min_count == 0 || min_count > window_volume {
            return Err(Box::new(KernelError::InvalidConfiguration(
                "min_count must be between 1 and the window volume",
            )));
        }
    }
    let parsed = modes(mode)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|_| KernelError::Memory)?;
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(window_volume)
        .map_err(|_| KernelError::Memory)?;
    for row in 0..shape[0] {
        for column in 0..shape[1] {
            scratch.clear();
            for window_row in 0..size[0] {
                let row_coordinate = i128::try_from(row)? + i128::try_from(window_row)?
                    - i128::try_from(size[0] / 2)?;
                let mapped_row = map_index(row_coordinate, shape[0], parsed[0])?;
                for window_column in 0..size[1] {
                    let column_coordinate = i128::try_from(column)?
                        + i128::try_from(window_column)?
                        - i128::try_from(size[1] / 2)?;
                    let mapped_column = map_index(column_coordinate, shape[1], parsed[1])?;
                    let value = match (mapped_row, mapped_column) {
                        (Some(source_row), Some(source_column)) => {
                            let flat = source_row
                                .checked_mul(shape[1])
                                .and_then(|base| base.checked_add(source_column))
                                .ok_or(KernelError::Overflow)?;
                            input[flat]
                        }
                        _ => cval,
                    };
                    scratch.push(value);
                }
            }
            output.push(Reduction::Median(policy(value_policy)).finish(&mut scratch));
        }
    }
    Ok(output)
}

fn verify_candidate<T, Bits>(
    input: &[T],
    shape: [usize; 2],
    size: [usize; 2],
    mode: [&str; 2],
    cval: T,
    value_policy: BenchmarkPolicy,
    bits: Bits,
) -> Result<(), Box<dyn std::error::Error>>
where
    T: crate::median_network::NetworkFloat,
    Bits: Fn(T) -> u64,
{
    let generic = generic_2d(input, shape, size, mode, cval, value_policy)?;
    let specialized = specialized_2d(input, shape, size, mode, cval, value_policy)?;
    if generic.len() != specialized.len() {
        return Err("generic and specialized outputs have different lengths".into());
    }
    for (index, (generic_value, specialized_value)) in
        generic.iter().zip(specialized.iter()).enumerate()
    {
        if bits(*generic_value) != bits(*specialized_value) {
            return Err(format!("generic and specialized outputs differ at index {index}").into());
        }
    }
    Ok(())
}

/// Verify bitwise parity for a benchmark-only `f32` candidate case.
pub fn verify_candidate_f32(
    input: &[f32],
    shape: [usize; 2],
    size: [usize; 2],
    mode: [&str; 2],
    cval: f32,
    value_policy: BenchmarkPolicy,
) -> Result<(), Box<dyn std::error::Error>> {
    verify_candidate(input, shape, size, mode, cval, value_policy, |value| {
        u64::from(value.to_bits())
    })
}

/// Verify bitwise parity for a benchmark-only `f64` candidate case.
pub fn verify_candidate_f64(
    input: &[f64],
    shape: [usize; 2],
    size: [usize; 2],
    mode: [&str; 2],
    cval: f64,
    value_policy: BenchmarkPolicy,
) -> Result<(), Box<dyn std::error::Error>> {
    verify_candidate(input, shape, size, mode, cval, value_policy, |value| {
        value.to_bits()
    })
}

#[cfg(test)]
mod tests {
    use super::{
        generic_2d, specialized_2d, verify_candidate_f32, verify_candidate_f64, BenchmarkPolicy,
    };

    #[test]
    fn benchmark_preflight_accepts_matching_f32_and_f64_cases() {
        verify_candidate_f32(
            &[1.0, f32::NAN, f32::INFINITY, -0.0, 5.0, 6.0],
            [2, 3],
            [3, 5],
            ["constant", "mirror"],
            -2.0,
            BenchmarkPolicy::Omit {
                ignore_inf: true,
                min_count: 2,
            },
        )
        .unwrap();
        verify_candidate_f64(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            [5, 3],
            ["nearest", "wrap"],
            0.0,
            BenchmarkPolicy::Propagate,
        )
        .unwrap();
    }

    #[test]
    fn specialized_candidate_is_bitwise_equal_for_all_modes_and_policies() {
        let input = [
            1.0_f64,
            -0.0,
            f64::NAN,
            3.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        for mode in ["reflect", "constant", "nearest", "mirror", "wrap"] {
            for value_policy in [
                BenchmarkPolicy::Propagate,
                BenchmarkPolicy::Omit {
                    ignore_inf: false,
                    min_count: 1,
                },
                BenchmarkPolicy::Omit {
                    ignore_inf: true,
                    min_count: 2,
                },
            ] {
                let generic =
                    generic_2d(&input, [2, 3], [3, 5], [mode, mode], -2.0, value_policy).unwrap();
                let specialized =
                    specialized_2d(&input, [2, 3], [3, 5], [mode, mode], -2.0, value_policy)
                        .unwrap();
                assert_eq!(
                    generic
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    specialized
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn specialized_candidate_matches_f32_anisotropic_and_singleton_windows() {
        let input = [1.0_f32, -0.0, f32::NAN, f32::INFINITY, 5.0, 6.0];
        for size in [[1, 1], [1, 5], [3, 1], [5, 3]] {
            let generic = generic_2d(
                &input,
                [2, 3],
                size,
                ["constant", "mirror"],
                f32::NAN,
                BenchmarkPolicy::Omit {
                    ignore_inf: true,
                    min_count: 1,
                },
            )
            .unwrap();
            let specialized = specialized_2d(
                &input,
                [2, 3],
                size,
                ["constant", "mirror"],
                f32::NAN,
                BenchmarkPolicy::Omit {
                    ignore_inf: true,
                    min_count: 1,
                },
            )
            .unwrap();
            assert_eq!(
                generic
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                specialized
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn specialized_candidate_rejects_invalid_min_count_like_generic_nd() {
        let input = [1.0_f64, 2.0, 3.0, 4.0];
        for min_count in [0, 10] {
            let policy = BenchmarkPolicy::Omit {
                ignore_inf: false,
                min_count,
            };
            assert!(
                generic_2d(&input, [2, 2], [3, 3], ["reflect", "reflect"], 0.0, policy,).is_err()
            );
            assert!(
                specialized_2d(&input, [2, 2], [3, 3], ["reflect", "reflect"], 0.0, policy,)
                    .is_err()
            );
        }
    }
}
