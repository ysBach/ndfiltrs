use crate::median_network::NetworkFloat;
use rayon::prelude::*;

use crate::error::KernelError;
use crate::filter_nd::{filter_range, FilterConfig};

#[cfg(test)]
pub(crate) fn filter_nd<T: NetworkFloat>(
    input: &[T],
    config: &FilterConfig<T>,
) -> Result<Vec<T>, KernelError> {
    let output_len = config.output_len();
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|_| KernelError::Memory)?;
    output.resize(output_len, T::nan());
    filter_nd_into(input, config, &mut output)?;
    Ok(output)
}

pub(crate) fn filter_nd_into<T: NetworkFloat>(
    input: &[T],
    config: &FilterConfig<T>,
    output: &mut [T],
) -> Result<(), KernelError> {
    let output_len = config.output_len();
    if output.len() != output_len || input.len() != output_len {
        return Err(KernelError::LengthMismatch);
    }
    if output_len == 0 {
        return Ok(());
    }
    let threads = rayon::current_num_threads().max(1);
    if threads == 1 || output_len <= 1 {
        return filter_range(input, config, 0, output);
    }
    let chunk_len = output_len.div_ceil(threads);
    output
        .par_chunks_mut(chunk_len)
        .enumerate()
        .try_for_each(|(chunk_index, chunk)| {
            let start = chunk_index
                .checked_mul(chunk_len)
                .ok_or(KernelError::Overflow)?;
            filter_range(input, config, start, chunk)
        })
}

#[cfg(feature = "python")]
pub(crate) fn get_num_threads() -> usize {
    rayon::current_num_threads()
}

#[cfg(test)]
mod tests {
    use crate::boundary::BoundaryMode::{Mirror, Reflect};
    use crate::filter_nd::{filter_nd_serial, FilterConfig};
    use crate::reduction::{InvalidPolicy, Reduction};

    use super::{filter_nd, filter_nd_into};

    #[test]
    fn network_matches_selection_when_chunks_split_interior_rows() {
        let mut input: Vec<f64> = (0..21 * 23)
            .map(|i| ((i * 37 % 101) as f64 - 50.0) / 8.0)
            .collect();
        for (offset, value) in [f64::NAN, -0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY]
            .into_iter()
            .enumerate()
        {
            input[37 + offset * 43] = value;
        }
        for policy in [
            InvalidPolicy::Propagate,
            InvalidPolicy::Omit {
                ignore_inf: false,
                min_count: 2,
            },
            InvalidPolicy::Omit {
                ignore_inf: true,
                min_count: 12,
            },
        ] {
            let direct = FilterConfig::new(
                vec![21, 23],
                vec![3, 4],
                vec![Reflect, Mirror],
                0.0,
                Reduction::Median(policy),
            )
            .unwrap();
            let network = FilterConfig::new(
                vec![21, 23],
                vec![3, 4],
                vec![Reflect, Mirror],
                0.0,
                Reduction::NetworkMedian(policy),
            )
            .unwrap();
            let expected: Vec<_> = filter_nd_serial(&input, &direct)
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect();
            for threads in [1, 2, 3, 4, 7] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let actual: Vec<_> = pool
                    .install(|| filter_nd(&input, &network))
                    .unwrap()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect();
                assert_eq!(actual, expected, "threads={threads}, policy={policy:?}");
            }
        }
    }

    #[test]
    fn parallel_output_is_bitwise_identical_to_serial_output() {
        let config = FilterConfig::new(
            vec![3, 4],
            vec![3, 3],
            vec![Reflect, Mirror],
            0.0_f64,
            Reduction::Median(InvalidPolicy::Omit {
                ignore_inf: false,
                min_count: 1,
            }),
        )
        .unwrap();
        let input = [
            1.0,
            -0.0,
            f64::NAN,
            3.0,
            4.0,
            f64::INFINITY,
            6.0,
            7.0,
            8.0,
            9.0,
            f64::NEG_INFINITY,
            11.0,
        ];
        let serial = filter_nd_serial(&input, &config).unwrap();
        let parallel = filter_nd(&input, &config).unwrap();
        assert_eq!(
            parallel
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            serial
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn parallel_kernel_fills_caller_owned_output_and_checks_length() {
        let config = FilterConfig::new(
            vec![3],
            vec![3],
            vec![Reflect],
            0.0_f64,
            Reduction::Median(InvalidPolicy::Propagate),
        )
        .unwrap();
        let mut output = [0.0; 3];
        filter_nd_into(&[1.0, 9.0, 2.0], &config, &mut output).unwrap();
        assert_eq!(output, [1.0, 2.0, 2.0]);
        assert!(filter_nd_into(&[1.0, 9.0, 2.0], &config, &mut [0.0; 2]).is_err());
    }
}
