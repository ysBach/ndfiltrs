//! Exact finite sums in units of the smallest binary64 subnormal.
//!
//! The signed width covers `usize::MAX` finite binary64 terms of either sign.
//! Callers must preserve that magnitude bound, including during merges. Counts
//! and nonfinite-value policies belong to the caller. Exact cancellation yields
//! positive zero; a negative nonzero result rounded to zero stays negative.

use std::cmp::Ordering;

const UNIT_EXP: i32 = f64::MIN_EXP - f64::MANTISSA_DIGITS as i32;
const MAGNITUDE_BITS: usize = (f64::MAX_EXP - UNIT_EXP) as usize;
const WORD_COUNT: usize = (MAGNITUDE_BITS + usize::BITS as usize + 1).div_ceil(64);

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// An IEEE binary format supported by the directly rounded mean.
pub(crate) trait ExactFloat: sealed::Sealed + reducers::Float {
    const PRECISION: u32;
    /// Exponent of the smallest positive subnormal.
    const MIN_EXP: i32;
    /// Exponent of the largest finite power of two.
    const MAX_EXP: i32;
    fn from_bits(bits: u64) -> Self;
}

impl ExactFloat for f32 {
    const PRECISION: u32 = f32::MANTISSA_DIGITS;
    const MIN_EXP: i32 = f32::MIN_EXP - f32::MANTISSA_DIGITS as i32;
    const MAX_EXP: i32 = f32::MAX_EXP - 1;

    fn from_bits(bits: u64) -> Self {
        f32::from_bits(bits as u32)
    }
}

impl ExactFloat for f64 {
    const PRECISION: u32 = f64::MANTISSA_DIGITS;
    const MIN_EXP: i32 = f64::MIN_EXP - f64::MANTISSA_DIGITS as i32;
    const MAX_EXP: i32 = f64::MAX_EXP - 1;

    fn from_bits(bits: u64) -> Self {
        f64::from_bits(bits)
    }
}

/// A little-endian, fixed-width two's-complement sum of finite binary64 values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExactSum {
    #[cfg(test)]
    words: [u64; WORD_COUNT],
}

#[cfg(test)]
impl Default for ExactSum {
    fn default() -> Self {
        Self {
            words: [0; WORD_COUNT],
        }
    }
}

impl ExactSum {
    /// Encode one finite value into a proven sufficient signed word span.
    ///
    /// `words` is overwritten. Its offset is in units of 64 binary64-subnormal
    /// bits. The caller must include all nonzero magnitude bits and a sign bit;
    /// `boxcar::word_span` proves these bounds for inputs and constant padding.
    pub(crate) fn encode_finite_words(value: f64, low_word: usize, words: &mut [u64]) {
        assert!(value.is_finite(), "exact sums require finite values");
        assert!(low_word <= WORD_COUNT && words.len() <= WORD_COUNT - low_word);
        words.fill(0);
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as usize;
        let fraction = bits & ((1_u64 << 52) - 1);
        let significand = fraction | if exponent == 0 { 0 } else { 1_u64 << 52 };
        if significand == 0 {
            return;
        }
        // A stored span may begin above the unnormalized exponent because the
        // significand has trailing zeros. Remove them before rebasing the shift.
        let trailing = significand.trailing_zeros() as usize;
        let shift = exponent.saturating_sub(1) + trailing;
        assert!(shift >= low_word * 64, "word span omits nonzero low bits");
        let local_shift = shift - low_word * 64;
        let significand = significand >> trailing;
        let magnitude_bits = local_shift + (64 - significand.leading_zeros()) as usize;
        assert!(
            magnitude_bits < words.len() * 64,
            "word span needs a sign bit"
        );
        let payload = (significand as u128) << (local_shift % 64);
        let index = local_shift / 64;
        words[index] = payload as u64;
        if payload >> 64 != 0 {
            words[index + 1] = (payload >> 64) as u64;
        }
        if bits >> 63 != 0 {
            let mut carry = true;
            for word in words {
                let (next, overflow) = (!*word).overflowing_add(u64::from(carry));
                *word = next;
                carry = overflow;
            }
        }
    }

    /// Add a finite value without rounding. Panics for NaN or infinity.
    #[cfg(test)]
    pub(crate) fn add_f64(&mut self, value: f64) {
        self.update_f64(value, false);
    }

    /// Remove a finite value without rounding. Panics for NaN or infinity.
    #[cfg(test)]
    pub(crate) fn remove_f64(&mut self, value: f64) {
        self.update_f64(value, true);
    }

    #[cfg(test)]
    fn update_f64(&mut self, value: f64, remove: bool) {
        assert!(value.is_finite(), "exact sums require finite values");
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as usize;
        let fraction = bits & ((1_u64 << 52) - 1);
        let (significand, shift) = if exponent == 0 {
            (fraction, 0)
        } else {
            (fraction | (1_u64 << 52), exponent - 1)
        };
        if significand == 0 {
            return;
        }
        let subtract = (bits >> 63 != 0) ^ remove;
        let payload = (significand as u128) << (shift % 64);
        let mut index = shift / 64;
        let mut carry = false;
        for word in [payload as u64, (payload >> 64) as u64] {
            let (next, first) = if subtract {
                self.words[index].overflowing_sub(word)
            } else {
                self.words[index].overflowing_add(word)
            };
            let (next, second) = if subtract {
                next.overflowing_sub(u64::from(carry))
            } else {
                next.overflowing_add(u64::from(carry))
            };
            self.words[index] = next;
            carry = first || second;
            index += 1;
        }
        while carry && index < WORD_COUNT {
            let (next, overflow) = if subtract {
                self.words[index].overflowing_sub(1)
            } else {
                self.words[index].overflowing_add(1)
            };
            self.words[index] = next;
            carry = overflow;
            index += 1;
        }
    }

    /// Add another exact sum; the caller must preserve the magnitude bound.
    #[cfg(test)]
    pub(crate) fn merge(&mut self, other: &Self) {
        self.combine(other, false);
    }

    /// Subtract another exact sum; the caller must preserve the magnitude bound.
    #[cfg(test)]
    pub(crate) fn subtract(&mut self, other: &Self) {
        self.combine(other, true);
    }

    #[cfg(test)]
    fn combine(&mut self, other: &Self, subtract: bool) {
        let mut carry = false;
        for (target, &source) in self.words.iter_mut().zip(&other.words) {
            let (next, first) = if subtract {
                target.overflowing_sub(source)
            } else {
                target.overflowing_add(source)
            };
            let (next, second) = if subtract {
                next.overflowing_sub(u64::from(carry))
            } else {
                next.overflowing_add(u64::from(carry))
            };
            *target = next;
            carry = first || second;
        }
    }

    /// Expose little-endian two's-complement words in units of `2^-1074`.
    #[cfg(test)]
    pub(crate) fn words(&self) -> &[u64] {
        &self.words
    }

    /// Decode a signed word span, with zero words below it and sign extension above.
    ///
    /// The span must include its sign bit; its highest word is not an unsigned
    /// magnitude. An empty span denotes zero. Panics if the span does not fit.
    #[cfg(test)]
    pub(crate) fn from_words(words: &[u64], low_word: usize) -> Self {
        assert!(low_word <= WORD_COUNT && words.len() <= WORD_COUNT - low_word);
        let mut result = Self::default();
        let end = low_word + words.len();
        result.words[low_word..end].copy_from_slice(words);
        if words.last().is_some_and(|word| word >> 63 != 0) {
            result.words[end..].fill(u64::MAX);
        }
        result
    }

    /// Divide by a positive count, rounding once to `T`, nearest with ties to even.
    ///
    #[cfg(test)]
    pub(crate) fn mean<T: ExactFloat>(&self, count: usize) -> T {
        Self::mean_words(&self.words, 0, count)
    }

    /// Round the mean of a compact signed sum without expanding its word span.
    ///
    /// Words are little-endian in units of `2^-1074`, with zero words below
    /// `low_word` and sign extension above the supplied span. The highest word
    /// must include the sign bit. Empty spans denote zero. Rounding is nearest
    /// with ties to even, directly to `T`; out-of-range results can overflow.
    /// Zero counts and spans exceeding the accumulator panic.
    ///
    /// Floating-point division seeds a quotient that integer corrections make
    /// exact before rounding. The numerator has at most
    /// `T::PRECISION + usize::BITS` bits (117 for binary64 on a 64-bit target).
    /// Counts need not be exactly representable as floats.
    pub(crate) fn mean_words<T: ExactFloat>(words: &[u64], low_word: usize, count: usize) -> T {
        assert!(low_word <= WORD_COUNT && words.len() <= WORD_COUNT - low_word);
        assert!(count != 0, "an exact mean requires a positive count");
        if words.last().is_some_and(|word| word >> 63 != 0) {
            // If W occupies B bits, sign extension means W - 2^B. Its
            // magnitude is 2^B - W, even for the most negative signed word.
            // Negate only this span; the implicit low zero words remain zero.
            let mut scratch = [0_u64; WORD_COUNT];
            let magnitude = &mut scratch[..words.len()];
            let mut carry = true;
            for (destination, &word) in magnitude.iter_mut().zip(words) {
                let (next, overflow) = (!word).overflowing_add(u64::from(carry));
                *destination = next;
                carry = overflow;
            }
            Self::round_mean(magnitude, low_word, true, count)
        } else {
            Self::round_mean(words, low_word, false, count)
        }
    }

    fn round_mean<T: ExactFloat>(
        magnitude: &[u64],
        low_word: usize,
        negative: bool,
        count: usize,
    ) -> T {
        let fraction_bits = T::PRECISION - 1;
        let min_normal = T::MIN_EXP + fraction_bits as i32;
        let infinity_exponent = (T::MAX_EXP - min_normal + 2) as u64;
        let sign = if negative {
            (infinity_exponent + 1) << fraction_bits
        } else {
            0
        };
        let Some(high_word) = magnitude.iter().rposition(|&word| word != 0) else {
            return T::from_bits(0);
        };
        let bits =
            (low_word + high_word) * 64 + (64 - magnitude[high_word].leading_zeros()) as usize;
        let count_bits = (usize::BITS - count.leading_zeros()) as usize;
        let unit_shift = (T::MIN_EXP - UNIT_EXP) as usize;
        let mut shift = bits
            .saturating_sub(T::PRECISION as usize + count_bits)
            .max(unit_shift);
        let high = high_u128(magnitude, low_word, shift);
        let divisor = count as u128;
        // The exact ratio is below 2^(precision+1). Rounded conversions and
        // division can reach that endpoint, but cannot exceed it: the numerator
        // rounds to at most 2^(precision+count_bits), and the denominator to at
        // least 2^(count_bits-1). Thus the seed product is below 2^118 on a
        // 64-bit target and fits u128. Neither correction can overflow: the
        // downward loop has a positive quotient, and upward steps stay <= high.
        let mut quotient = ((high as f64) / (count as f64)) as u128;
        let mut product = quotient * divisor;
        while product > high {
            quotient -= 1;
            product -= divisor;
        }
        while high - product >= divisor {
            quotient += 1;
            product += divisor;
        }
        let mut remainder = high - product;

        // The initial quotient has at most precision+1 bits. If it has the
        // extra bit, halve the numerator without another division:
        // floor((q*n+r)/2) = floor(q/2)*n + floor(((q%2)*n+r)/2).
        if quotient >= (1_u128 << T::PRECISION) {
            remainder = ((quotient & 1) * divisor + remainder) >> 1;
            quotient >>= 1;
            shift += 1;
        }

        // The exact fractional remainder is (r + low/2^shift)/n. Only
        // 2*r == n or 2*r == n-1 can need information from discarded bits.
        let odd = quotient & 1 != 0;
        let round_up = match (2 * remainder).cmp(&divisor) {
            Ordering::Greater => true,
            Ordering::Equal => any_below(magnitude, low_word, shift) || odd,
            Ordering::Less => {
                2 * remainder + 1 == divisor
                    && shift != 0
                    && bit_at(magnitude, low_word, shift - 1)
                    && (any_below(magnitude, low_word, shift - 1) || odd)
            }
        };
        quotient += u128::from(round_up);
        if quotient == 0 {
            return T::from_bits(sign);
        }
        if quotient == (1_u128 << T::PRECISION) {
            quotient >>= 1;
            shift += 1;
        }
        let top = (127 - quotient.leading_zeros()) as i32;
        let exponent = UNIT_EXP + shift as i32 + top;
        if exponent > T::MAX_EXP {
            return T::from_bits(sign | (infinity_exponent << fraction_bits));
        }
        if exponent < min_normal {
            debug_assert_eq!(shift, unit_shift);
            return T::from_bits(sign | quotient as u64);
        }
        debug_assert_eq!(top, fraction_bits as i32);
        let exponent_field = (exponent - min_normal + 1) as u64;
        let fraction = quotient as u64 & ((1_u64 << fraction_bits) - 1);
        T::from_bits(sign | (exponent_field << fraction_bits) | fraction)
    }
}

/// Read a magnitude word, including the implicit zero words outside its span.
fn word_at(words: &[u64], low_word: usize, index: usize) -> u64 {
    index
        .checked_sub(low_word)
        .and_then(|local| words.get(local))
        .copied()
        .unwrap_or(0)
}

/// Extract the high integer after a global shift; the caller proves u128 fits.
/// Reading virtual words also covers a left shift of the compact magnitude
/// when `shift` is below `low_word * 64`.
fn high_u128(words: &[u64], low_word: usize, shift: usize) -> u128 {
    let index = shift / 64;
    let offset = shift % 64;
    let mut high = (word_at(words, low_word, index) >> offset) as u128;
    high |= (word_at(words, low_word, index + 1) as u128) << (64 - offset);
    if offset != 0 {
        high |= (word_at(words, low_word, index + 2) as u128) << (128 - offset);
    }
    high
}

fn bit_at(words: &[u64], low_word: usize, bit: usize) -> bool {
    word_at(words, low_word, bit / 64) & (1_u64 << (bit % 64)) != 0
}

fn any_below(words: &[u64], low_word: usize, bits: usize) -> bool {
    let bits = bits.saturating_sub(low_word * 64);
    let whole = (bits / 64).min(words.len());
    let tail = bits % 64;
    words[..whole].iter().any(|&word| word != 0)
        || (tail != 0
            && words
                .get(whole)
                .is_some_and(|word| word & ((1_u64 << tail) - 1) != 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum(values: &[f64]) -> ExactSum {
        let mut result = ExactSum::default();
        for &value in values {
            result.add_f64(value);
        }
        result
    }

    #[test]
    fn compact_encoding_matches_full_accumulation_across_exponent_boundaries() {
        for exponent in 0..0x7ff_u64 {
            for fraction in [0, 1, 1_u64 << 51, (1_u64 << 52) - 1] {
                let positive = f64::from_bits((exponent << 52) | fraction);
                let magnitude = sum(&[positive]);
                let magnitude_words = magnitude.words();
                let low = magnitude_words
                    .iter()
                    .position(|&word| word != 0)
                    .unwrap_or(0);
                let high = magnitude_words
                    .iter()
                    .rposition(|&word| word != 0)
                    .unwrap_or(0);
                let end = high + 1 + usize::from(magnitude_words[high] >> 63 != 0);
                for value in [positive, -positive] {
                    let reference = sum(&[value]);
                    // Both the tight signed span and a wider span must fully
                    // overwrite earlier state, including sign-extension words.
                    for limit in [end, WORD_COUNT] {
                        let mut encoded = vec![0x55aa_55aa_55aa_55aa; limit - low];
                        ExactSum::encode_finite_words(value, low, &mut encoded);
                        assert_eq!(encoded, reference.words()[low..limit]);
                        assert_eq!(ExactSum::from_words(&encoded, low), reference);
                    }
                }
            }
        }
    }

    #[test]
    fn cancellation_and_removal_recover_extreme_finite_values() {
        let tiny = f64::from_bits(1);
        let mut value = sum(&[f64::MAX, f64::MAX, tiny]);
        value.remove_f64(f64::MAX);
        value.remove_f64(f64::MAX);
        assert_eq!(value.mean::<f64>(1).to_bits(), 1);
        assert_eq!(sum(&[f64::MAX, 1.0, -f64::MAX]).mean::<f64>(1), 1.0);
        assert_eq!(sum(&[-f64::MAX, -tiny, f64::MAX]).mean::<f64>(1), -tiny);
        assert_eq!(sum(&[f64::MAX, f64::MAX]).mean::<f64>(2), f64::MAX);
        assert_eq!(sum(&[-f64::MAX, -f64::MAX]).mean::<f64>(2), -f64::MAX);
        value.remove_f64(tiny);
        assert_eq!(value, ExactSum::default());
    }

    #[test]
    fn merges_and_subtractions_keep_low_bits() {
        let tiny = f64::from_bits(1);
        let first = sum(&[f64::MAX, tiny]);
        let second = sum(&[-f64::MAX, 3.0 * tiny]);
        let mut combined = first.clone();
        combined.merge(&second);
        assert_eq!(combined.mean::<f64>(1).to_bits(), 4);
        combined.subtract(&second);
        assert_eq!(combined, first);
        combined.subtract(&first);
        assert_eq!(combined, ExactSum::default());
        combined.subtract(&second);
        combined.merge(&second);
        assert_eq!(combined, ExactSum::default());
    }

    #[test]
    fn compact_words_reconstruct_signed_values_and_zero_low_words() {
        for value in [1.0, -1.0, f64::MAX, -f64::MAX, f64::from_bits(1)] {
            let original = sum(&[value]);
            assert_eq!(ExactSum::from_words(original.words(), 0), original);
        }
        for value in [1.0, -1.0] {
            let original = sum(&[value]);
            assert_eq!(
                ExactSum::from_words(&original.words()[16..17], 16),
                original
            );
        }
        assert_eq!(ExactSum::from_words(&[], WORD_COUNT), ExactSum::default());
        // A signed -1 at word 2 means -2^128 lattice units = -2^-946.
        let negative = ExactSum::from_words(&[u64::MAX], 2);
        assert_eq!(negative.mean::<f64>(1), -f64::from_bits(77_u64 << 52));
    }

    fn assert_compact_matches_full(words: &[u64], low_word: usize, count: usize) {
        let full = ExactSum::from_words(words, low_word);
        assert_eq!(
            ExactSum::mean_words::<f64>(words, low_word, count).to_bits(),
            full.mean::<f64>(count).to_bits(),
            "binary64: words={words:?}, low_word={low_word}, count={count}"
        );
        assert_eq!(
            ExactSum::mean_words::<f32>(words, low_word, count).to_bits(),
            full.mean::<f32>(count).to_bits(),
            "binary32: words={words:?}, low_word={low_word}, count={count}"
        );
    }

    #[test]
    fn compact_means_match_sign_extended_decoding_at_every_word_offset() {
        // Include redundant sign words, a carry through low zero words, and
        // the most negative signed span, whose unsigned magnitude keeps bit 63.
        let spans: &[&[u64]] = &[
            &[],
            &[0],
            &[1],
            &[u64::MAX],
            &[1_u64 << 63],
            &[0, 1],
            &[0, u64::MAX],
            &[u64::MAX, 0],
            &[u64::MAX, u64::MAX],
            &[1, 0, 1],
            &[0, 0, 1_u64 << 63],
        ];
        for &words in spans {
            for low_word in 0..=WORD_COUNT - words.len() {
                for count in [1, 2, 3, 7, usize::MAX / 2 + 1, usize::MAX] {
                    assert_compact_matches_full(words, low_word, count);
                }
            }
        }
    }

    #[test]
    fn compact_ieee_extrema_match_full_sums() {
        for value in [
            0.0,
            1.0,
            f64::from_bits(1),
            f64::from_bits((1_u64 << 52) - 1),
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::from(f32::from_bits(1)),
            f64::from(f32::MIN_POSITIVE),
            f64::from(f32::MAX),
        ] {
            for sign in [1.0, -1.0] {
                let original = sum(&[sign * value, sign * value, sign * value]);
                let words = original.words();
                let low = words
                    .iter()
                    .position(|&word| word != 0)
                    .unwrap_or(words.len());
                let mut end = words.len();
                while end > low + 1 {
                    let extension = if words[end - 2] >> 63 != 0 {
                        u64::MAX
                    } else {
                        0
                    };
                    if words[end - 1] != extension {
                        break;
                    }
                    end -= 1;
                }
                for count in [1, 2, 3, 7, usize::MAX] {
                    assert_compact_matches_full(&words[low..end], low, count);
                }
            }
        }
    }

    #[test]
    fn compact_mean_restores_precision_from_implicit_low_zero_words() {
        // Three ones have local magnitude 3*2^50 at word 16. The initial
        // binary64 quotient needs bits below that word, so extraction shifts
        // this local magnitude left rather than discarding its low bits.
        let magnitude = 3_u64 << 50;
        assert_eq!(ExactSum::mean_words::<f64>(&[magnitude], 16, 3), 1.0);
        assert_eq!(
            ExactSum::mean_words::<f64>(&[magnitude.wrapping_neg()], 16, 3),
            -1.0
        );
    }

    #[test]
    fn binary64_midpoints_round_to_even_with_both_signs() {
        let one = 1.0_f64.to_bits();
        for sign in [1.0, -1.0] {
            assert_eq!(
                sum(&[sign, sign * f64::from_bits(one + 1)]).mean::<f64>(2),
                sign
            );
            assert_eq!(
                sum(&[
                    sign * f64::from_bits(one + 1),
                    sign * f64::from_bits(one + 2)
                ])
                .mean::<f64>(2),
                sign * f64::from_bits(one + 2)
            );
        }
    }

    #[test]
    fn discarded_half_and_sticky_bits_resolve_odd_and_even_divisors() {
        let tiny = f64::from_bits(1);
        let next = f64::from_bits(1.0_f64.to_bits() + 1);
        // In each case the exact mean is 1 + 2^-53 + adjustment/count.
        for (count, terms) in [
            (2, vec![1.0, 1.0, 2.0_f64.powi(-52)]),
            (3, vec![1.0, 1.0, 1.0, 3.0 * 2.0_f64.powi(-53)]),
        ] {
            for (adjustment, expected) in [(-tiny, 1.0), (0.0, 1.0), (tiny, next)] {
                let mut value = sum(&terms);
                value.add_f64(adjustment);
                assert_eq!(value.mean::<f64>(count), expected);
                let mut negative = ExactSum::default();
                negative.subtract(&value);
                assert_eq!(negative.mean::<f64>(count), -expected);
            }
        }
    }

    #[test]
    fn binary64_subnormals_and_signed_underflow_are_rounded_once() {
        let tiny = f64::from_bits(1);
        assert_eq!(sum(&[tiny]).mean::<f64>(2).to_bits(), 0);
        assert_eq!(sum(&[-tiny]).mean::<f64>(2).to_bits(), 1_u64 << 63);
        assert_eq!(sum(&[3.0 * tiny]).mean::<f64>(2).to_bits(), 2);
        assert_eq!(sum(&[2.0 * tiny]).mean::<f64>(3).to_bits(), 1);
        assert_eq!(sum(&[tiny]).mean::<f64>(3).to_bits(), 0);
        assert_eq!(sum(&[-0.0, 0.0]).mean::<f64>(1).to_bits(), 0);
        // Midpoint between largest subnormal and smallest normal, latter even.
        assert_eq!(
            sum(&[f64::from_bits((1_u64 << 52) - 1), f64::MIN_POSITIVE]).mean::<f64>(2),
            f64::MIN_POSITIVE
        );
    }

    #[test]
    fn binary32_avoids_double_rounding_and_handles_subnormal_ties() {
        let midpoint = sum(&[1.0, 2.0_f64.powi(-24), 2.0_f64.powi(-80)]);
        assert_eq!(midpoint.mean::<f32>(1).to_bits(), 1.0_f32.to_bits() + 1);
        assert_eq!(
            (midpoint.mean::<f64>(1) as f32).to_bits(),
            1.0_f32.to_bits()
        );
        let half_tiny = 2.0_f64.powi(-150);
        assert_eq!(sum(&[half_tiny]).mean::<f32>(1).to_bits(), 0);
        assert_eq!(sum(&[-half_tiny]).mean::<f32>(1).to_bits(), 1_u32 << 31);
        assert_eq!(sum(&[3.0 * half_tiny]).mean::<f32>(1).to_bits(), 2);
        assert_eq!(
            sum(&[half_tiny, f64::from_bits(1)])
                .mean::<f32>(1)
                .to_bits(),
            1
        );
        assert_eq!(
            sum(&[
                f64::from(f32::from_bits((1_u32 << 23) - 1)),
                f64::from(f32::MIN_POSITIVE)
            ])
            .mean::<f32>(2),
            f32::MIN_POSITIVE
        );
    }

    #[test]
    fn target_overflow_and_maximum_count() {
        assert_eq!(sum(&[f64::MAX, f64::MAX]).mean::<f64>(1), f64::INFINITY);
        assert_eq!(
            sum(&[-f64::MAX, -f64::MAX]).mean::<f64>(1),
            f64::NEG_INFINITY
        );
        assert_eq!(sum(&[f64::MAX]).mean::<f32>(1), f32::INFINITY);
        let mut many = ExactSum::default();
        let mut term = sum(&[f64::MAX]);
        // Binary expansion of usize::MAX: exact sum of the largest supported
        // number of MAX terms, without iterating over those terms.
        for bit in 0..usize::BITS {
            many.merge(&term);
            if bit + 1 < usize::BITS {
                let duplicate = term.clone();
                term.merge(&duplicate);
            }
        }
        assert_eq!(many.mean::<f64>(usize::MAX), f64::MAX);
        // A count-sized integer in subnormal units must divide to one unit,
        // even when the count itself is not exactly representable in f64.
        let units = ExactSum::from_words(&[usize::MAX as u64, 0], 0);
        assert_eq!(units.mean::<f64>(usize::MAX).to_bits(), 1);
    }

    #[test]
    fn overflow_midpoints_use_discarded_bits() {
        let tiny = f64::from_bits(1);
        let mut limit64 = sum(&[f64::MAX, 2.0_f64.powi(970)]);
        assert_eq!(limit64.mean::<f64>(1), f64::INFINITY);
        limit64.remove_f64(tiny);
        assert_eq!(limit64.mean::<f64>(1), f64::MAX);
        let mut limit32 = sum(&[f64::from(f32::MAX), 2.0_f64.powi(103)]);
        assert_eq!(limit32.mean::<f32>(1), f32::INFINITY);
        limit32.remove_f64(tiny);
        assert_eq!(limit32.mean::<f32>(1), f32::MAX);
    }

    #[test]
    fn a_repeated_constant_is_preserved_across_the_binary64_exponent_range() {
        // Cover exponent transitions and significand endpoints. Multiplicity
        // three exercises division without constructing a rounded reference sum.
        for exponent in 0..0x7ff_u64 {
            for fraction in [0, 1, (1_u64 << 51) - 1, (1_u64 << 52) - 1] {
                for sign in [0, 1_u64 << 63] {
                    let bits = sign | (exponent << 52) | fraction;
                    let value = f64::from_bits(bits);
                    let expected = if value == 0.0 { 0 } else { bits };
                    assert_eq!(
                        sum(&[value, value, value]).mean::<f64>(3).to_bits(),
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn exact_dyadic_numerators_match_native_ieee_division() {
        // Both native operands are exact: small integers times powers of two,
        // divided by exactly representable counts. Hardware division is an
        // independent rounding oracle, including normal/subnormal transitions.
        for numerator in -31..=31 {
            for count in [1, 2, 3, 5, 7, 31, 127, 65535] {
                for unit in [
                    f64::from_bits(1),
                    f64::MIN_POSITIVE,
                    2.0_f64.powi(-500),
                    1.0,
                    2.0_f64.powi(500),
                    2.0_f64.powi(970),
                ] {
                    let value = f64::from(numerator) * unit;
                    assert_eq!(
                        sum(&[value]).mean::<f64>(count).to_bits(),
                        (value / count as f64).to_bits()
                    );
                }
                for unit in [
                    f32::from_bits(1),
                    f32::MIN_POSITIVE,
                    2.0_f32.powi(-50),
                    1.0,
                    2.0_f32.powi(50),
                    2.0_f32.powi(100),
                ] {
                    let value = numerator as f32 * unit;
                    assert_eq!(
                        sum(&[f64::from(value)]).mean::<f32>(count).to_bits(),
                        (value / count as f32).to_bits()
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "finite")]
    fn rejects_nonfinite_addends() {
        ExactSum::default().add_f64(f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "positive count")]
    fn rejects_zero_count() {
        ExactSum::default().mean::<f64>(0);
    }
}
