//! Exact selection with a generated, data-independent comparator network.

use reducers::Float;

use crate::error::KernelError;
use crate::reduction::InvalidPolicy;

/// IEEE floats with reversible integer keys in `total_cmp` order.
pub trait NetworkFloat: Float {
    type Key: Copy + Ord + Send + Sync;
    fn sort_key(self) -> Self::Key;
    fn from_sort_key(key: Self::Key) -> Self;
}

macro_rules! network_float {
    ($float:ty, $signed:ty, $unsigned:ty) => {
        impl NetworkFloat for $float {
            type Key = $signed;

            #[inline]
            fn sort_key(self) -> Self::Key {
                let bits = self.to_bits() as $signed;
                bits ^ (((bits >> (<$signed>::BITS - 1)) as $unsigned >> 1) as $signed)
            }

            #[inline]
            fn from_sort_key(key: Self::Key) -> Self {
                // The sign bit is unchanged, so the same XOR is its inverse.
                let bits = key ^ (((key >> (<$signed>::BITS - 1)) as $unsigned >> 1) as $signed);
                Self::from_bits(bits as $unsigned)
            }
        }
    };
}

network_float!(f32, i32, u32);
network_float!(f64, i64, u64);

/// Encode input ordering once per tap and count retained values per output.
pub(crate) fn encode_plane<T: NetworkFloat>(
    values: &[T],
    keys: &mut [T::Key],
    policy: InvalidPolicy,
    retained: &mut [usize],
) {
    for ((&value, key), valid) in values.iter().zip(keys).zip(retained.iter_mut()) {
        if value.is_nan()
            || (matches!(
                policy,
                InvalidPolicy::Omit {
                    ignore_inf: true,
                    ..
                }
            ) && !value.is_finite())
        {
            // Positive canonical NaN sorts above every retained value.
            *key = T::nan().sort_key();
        } else {
            *key = value.sort_key();
            *valid += 1;
        }
    }
}

type Comparator = (usize, usize, bool, bool);

fn sort(
    start: usize,
    length: usize,
    count: usize,
    comparisons: &mut Vec<Comparator>,
) -> Result<(), KernelError> {
    if length > 1 && start < count {
        let half = length / 2;
        sort(start, half, count, comparisons)?;
        sort(start + half, half, count, comparisons)?;
        merge(start, length, 1, count, comparisons)?;
    }
    Ok(())
}

fn merge(
    start: usize,
    length: usize,
    stride: usize,
    count: usize,
    comparisons: &mut Vec<Comparator>,
) -> Result<(), KernelError> {
    if start >= count {
        return Ok(());
    }
    let step = stride * 2;
    if step < length {
        merge(start, length, step, count, comparisons)?;
        merge(start + stride, length, step, count, comparisons)?;
        for lower in (start + stride..start + length - stride).step_by(step) {
            if lower + stride < count {
                comparisons
                    .try_reserve(1)
                    .map_err(|_| KernelError::Memory)?;
                comparisons.push((lower, lower + stride, true, true));
            }
        }
    } else if start + stride < count {
        comparisons
            .try_reserve(1)
            .map_err(|_| KernelError::Memory)?;
        comparisons.push((start, start + stride, true, true));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(crate) struct Network {
    wires: usize,
    comparisons: Vec<Comparator>,
}

impl Network {
    pub(crate) fn new(count: usize, policy: InvalidPolicy) -> Result<Self, KernelError> {
        if count == 0 {
            return Err(KernelError::InvalidConfiguration(
                "median needs at least one tap",
            ));
        }
        let wires = count
            .checked_next_power_of_two()
            .ok_or(KernelError::Overflow)?;
        let mut comparisons = Vec::new();
        // Odd-even merges compare only ascending wires. Symbolic +infinity
        // padding therefore stays beyond `count` and needs no stored values.
        sort(0, wires, count, &mut comparisons)?;
        // A propagated-NaN median needs only the middle output wire(s).
        // Omission can change the rank, so all retained ranks remain live.
        let mut live = Vec::new();
        live.try_reserve_exact(count)
            .map_err(|_| KernelError::Memory)?;
        live.resize(count, false);
        match policy {
            InvalidPolicy::Propagate => {
                live[count / 2] = true;
                live[(count - 1) / 2] = true;
            }
            InvalidPolicy::Omit { .. } => live[..count].fill(true),
        }
        comparisons.reverse();
        comparisons.retain_mut(|(left, right, keep_left, keep_right)| {
            let (left, right) = (*left, *right);
            if !live[left] && !live[right] {
                false
            } else {
                *keep_left = live[left];
                *keep_right = live[right];
                live[left] = true;
                live[right] = true;
                true
            }
        });
        comparisons.reverse();
        Ok(Self {
            wires: count,
            comparisons,
        })
    }

    pub(crate) fn wires(&self) -> usize {
        self.wires
    }

    /// Sort independent columns of a tap-major buffer and select their medians.
    pub(crate) fn apply<T: NetworkFloat>(
        &self,
        keys: &mut [T::Key],
        count: usize,
        policy: InvalidPolicy,
        retained: &[usize],
        output: &mut [T],
    ) {
        let width = output.len();
        for &(lower, upper, keep_lower, keep_upper) in &self.comparisons {
            let (head, tail) = keys.split_at_mut(upper * width);
            let low = &mut head[lower * width..(lower + 1) * width];
            let high = &mut tail[..width];
            if keep_lower && keep_upper {
                for (left, right) in low.iter_mut().zip(high) {
                    let a = *left;
                    let b = *right;
                    *left = a.min(b);
                    *right = a.max(b);
                }
            } else if keep_lower {
                for (left, right) in low.iter_mut().zip(high) {
                    *left = (*left).min(*right);
                }
            } else {
                for (left, right) in low.iter_mut().zip(high) {
                    *right = (*left).max(*right);
                }
            }
        }
        for (column, (destination, &valid)) in output.iter_mut().zip(retained.iter()).enumerate() {
            let missing = match policy {
                InvalidPolicy::Propagate => valid != count,
                InvalidPolicy::Omit { min_count, .. } => valid < min_count,
            };
            *destination = if missing || valid == 0 {
                T::nan()
            } else if valid % 2 == 1 {
                T::from_sort_key(keys[(valid / 2) * width + column])
            } else {
                let mut middle = [
                    T::from_sort_key(keys[(valid / 2 - 1) * width + column]),
                    T::from_sort_key(keys[(valid / 2) * width + column]),
                ];
                T::from_f64(reducers::reducers_1d::median_valid_in_place(&mut middle))
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{encode_plane, Network, NetworkFloat};
    use crate::error::KernelError;
    use crate::reduction::InvalidPolicy;

    #[test]
    fn unrepresentable_padding_is_rejected_before_allocation() {
        assert!(matches!(
            Network::new(usize::MAX, InvalidPolicy::Propagate),
            Err(KernelError::Overflow)
        ));
    }

    #[test]
    fn integer_keys_preserve_ieee_bits_and_total_order() {
        macro_rules! check {
            ($float:ty, $bits:ty) => {{
                let mut state = 0x4ad3_81f7_d058_92b1u64;
                let mut previous = <$float>::NEG_INFINITY;
                for value in [
                    -<$float>::MAX,
                    -<$float>::MIN_POSITIVE,
                    -<$float>::from_bits(1),
                    -0.0,
                    0.0,
                    <$float>::from_bits(1),
                    <$float>::MIN_POSITIVE,
                    <$float>::MAX,
                    <$float>::INFINITY,
                    <$float>::NAN,
                ]
                .into_iter()
                .chain((0..10000).map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    <$float>::from_bits(state as $bits)
                })) {
                    assert_eq!(
                        <$float>::from_sort_key(value.sort_key()).to_bits(),
                        value.to_bits()
                    );
                    assert_eq!(
                        previous.total_cmp(&value),
                        previous.sort_key().cmp(&value.sort_key())
                    );
                    previous = value;
                }
            }};
        }
        check!(f32, u32);
        check!(f64, u64);
    }

    #[test]
    fn zero_one_principle_checks_generated_selection_for_odd_and_even_counts() {
        // Every binary input checks the selection topology independently of a
        // sorting implementation, including non-power-of-two padding.
        for count in 1..=12 {
            let width = 1usize << count;
            for policy in [
                InvalidPolicy::Propagate,
                InvalidPolicy::Omit {
                    ignore_inf: false,
                    min_count: 1,
                },
            ] {
                let network = Network::new(count, policy).unwrap();
                let mut values = vec![f64::NAN; network.wires() * width];
                for tap in 0..count {
                    for column in 0..width {
                        values[tap * width + column] = ((column >> tap) & 1) as f64;
                    }
                }
                let mut output = vec![0.0; width];
                let mut keys = vec![0; count * width];
                let mut retained = vec![0; width];
                for (plane, key_plane) in
                    values.chunks_exact(width).zip(keys.chunks_exact_mut(width))
                {
                    encode_plane(plane, key_plane, policy, &mut retained);
                }
                network.apply(&mut keys, count, policy, &retained, &mut output);
                for (column, value) in output.into_iter().enumerate() {
                    let ones = column.count_ones() as usize;
                    let expected = match (2 * ones).cmp(&count) {
                        std::cmp::Ordering::Less => 0.0,
                        std::cmp::Ordering::Equal => 0.5,
                        std::cmp::Ordering::Greater => 1.0,
                    };
                    assert_eq!(value, expected, "count={count}, bits={column:b}");
                }
            }
        }
    }
}
