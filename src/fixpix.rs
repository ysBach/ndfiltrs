//! Mask-directed interpolation preserving `astroimred.fixpix` write order.

use std::collections::HashSet;

use reducers::Float;

use crate::error::KernelError;

/// Compact nonzero mask bytes in C order, skipping empty machine words.
#[cfg(any(test, feature = "python"))]
pub(crate) fn masked_positions(flags: &[u8]) -> Result<Vec<usize>, KernelError> {
    const WIDTH: usize = std::mem::size_of::<usize>();
    let mut positions = Vec::new();
    let mut chunks = flags.chunks_exact(WIDTH);
    for (word_index, chunk) in chunks.by_ref().enumerate() {
        if usize::from_ne_bytes(chunk.try_into().expect("machine-word chunk")) == 0 {
            continue;
        }
        for (byte_index, &flag) in chunk.iter().enumerate() {
            if flag != 0 {
                positions.try_reserve(1).map_err(|_| KernelError::Memory)?;
                positions.push(word_index * WIDTH + byte_index);
            }
        }
    }
    let remainder = chunks.remainder();
    let start = flags.len() - remainder.len();
    for (index, &flag) in remainder.iter().enumerate() {
        if flag != 0 {
            positions.try_reserve(1).map_err(|_| KernelError::Memory)?;
            positions.push(start + index);
        }
    }
    Ok(positions)
}

#[derive(Clone, Copy, Debug)]
struct Span {
    start: usize,
    length: usize,
    stride: usize,
    axis: usize,
    coordinate: usize,
    extent: usize,
}

/// Repair an owned C-order array from strictly increasing masked offsets.
pub(crate) fn fixpix<T: Float>(
    data: &mut [T],
    shape: &[usize],
    positions: &[usize],
    priority: &[usize],
) -> Result<(), KernelError> {
    if shape.is_empty() || priority.len() != shape.len() {
        return Err(KernelError::InvalidConfiguration(
            "priority must contain every array axis exactly once",
        ));
    }
    for (index, &axis) in priority.iter().enumerate() {
        if axis >= shape.len() || priority[..index].contains(&axis) {
            return Err(KernelError::InvalidConfiguration(
                "priority must contain every array axis exactly once",
            ));
        }
    }
    let mut strides = Vec::new();
    strides
        .try_reserve_exact(shape.len())
        .map_err(|_| KernelError::Memory)?;
    strides.resize(shape.len(), 0);
    let mut total = 1_usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = total;
        total = total
            .checked_mul(shape[axis])
            .ok_or(KernelError::Overflow)?;
    }
    if total != data.len() {
        return Err(KernelError::LengthMismatch);
    }
    if positions.last().is_some_and(|&last| last >= total)
        || positions.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(KernelError::InvalidConfiguration(
            "masked offsets must be sorted, unique, and within the array",
        ));
    }
    // With no good anchors, every axis run spans the full axis and is skipped.
    if positions.is_empty() || positions.len() == total {
        return Ok(());
    }

    let count = positions.len();
    let mut selected = Vec::new();
    selected
        .try_reserve_exact(count)
        .map_err(|_| KernelError::Memory)?;
    selected.resize(
        count,
        Span {
            start: 0,
            length: usize::MAX,
            stride: 0,
            axis: 0,
            coordinate: 0,
            extent: 0,
        },
    );
    let mut order = Vec::new();
    order
        .try_reserve_exact(count)
        .map_err(|_| KernelError::Memory)?;

    for &axis in priority {
        let stride = strides[axis];
        let extent = shape[axis];
        order.clear();
        for (index, &offset) in positions.iter().enumerate() {
            // Remove this axis's coordinate to obtain a unique line key.
            let coordinate = (offset / stride) % extent;
            let line = offset - coordinate * stride;
            order.push((line, offset, index));
        }
        // C-order already groups lines parallel to the contiguous axis.
        if stride != 1 {
            order.sort_unstable();
        }
        let mut first = 0;
        while first < count {
            let mut stop = first + 1;
            while stop < count
                && order[stop].0 == order[first].0
                && order[stop].1 - order[stop - 1].1 == stride
            {
                stop += 1;
            }
            let span = Span {
                start: order[first].1,
                length: stop - first,
                stride,
                axis,
                coordinate: (order[first].1 / stride) % extent,
                extent,
            };
            for &(_, _, index) in &order[first..stop] {
                // Strict comparison preserves the first priority axis on ties.
                if span.length < selected[index].length {
                    selected[index] = span;
                }
            }
            first = stop;
        }
    }
    drop(order);

    // Anchors are outside the ORIGINAL mask: all executions of a given span
    // write identical values. Keep its LAST C-order trigger, then execute the
    // retained spans in that same order. Keeping the first or repairing each
    // masked coordinate independently would change crossing-run overwrites.
    let mut seen = HashSet::new();
    seen.try_reserve(count).map_err(|_| KernelError::Memory)?;
    for span in selected.iter_mut().rev() {
        if span.length == span.extent || !seen.insert((span.axis, span.start)) {
            span.length = 0;
        }
    }
    drop(seen);
    for span in selected {
        if span.length == 0 {
            continue;
        }
        let end = span.start + (span.length - 1) * span.stride;
        let left = if span.coordinate == 0 {
            end + span.stride
        } else {
            span.start - span.stride
        };
        let right = if span.coordinate + span.length == span.extent {
            left
        } else {
            end + span.stride
        };
        let a = data[left].to_f64();
        let b = data[right].to_f64();
        // Preserve the legacy f64 operation order, including nonfinite values,
        // overflow, signed zeros, and a single cast back to the input dtype.
        let slope = (b - a) / (span.length + 1) as f64;
        for distance in 1..=span.length {
            data[span.start + (distance - 1) * span.stride] =
                T::from_f64(slope * distance as f64 + a);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{fixpix, masked_positions};

    #[test]
    fn mask_compaction_handles_word_edges_tails_and_noncanonical_bytes() {
        let width = std::mem::size_of::<usize>();
        for length in 0..=3 * width + 1 {
            for start in 0..=length {
                let mut flags = vec![0; length];
                for (index, flag) in flags.iter_mut().enumerate().skip(start) {
                    if index % 3 == 0 {
                        *flag = 255;
                    }
                }
                let expected: Vec<_> = flags
                    .iter()
                    .enumerate()
                    .filter_map(|(index, &flag)| (flag != 0).then_some(index))
                    .collect();
                assert_eq!(masked_positions(&flags).unwrap(), expected);
            }
        }
    }

    #[test]
    fn interpolates_interior_and_edge_runs() {
        let mut data = [0.0_f64, 90.0, 90.0, 6.0, 90.0];
        fixpix(&mut data, &[5], &[1, 2, 4], &[0]).unwrap();
        assert_eq!(data, [0.0, 2.0, 4.0, 6.0, 6.0]);
    }

    #[test]
    fn full_mask_and_empty_input_are_unchanged() {
        let mut data = [1.0_f32, 2.0];
        fixpix(&mut data, &[2], &[0, 1], &[0]).unwrap();
        assert_eq!(data, [1.0, 2.0]);
        fixpix::<f64>(&mut [], &[0, 3], &[], &[1, 0]).unwrap();
    }

    #[test]
    fn validates_shape_priority_and_offsets() {
        for priority in [vec![], vec![0, 0], vec![0, 2]] {
            assert!(fixpix(&mut [0.0_f64; 4], &[2, 2], &[], &priority).is_err());
        }
        for offsets in [vec![4], vec![1, 1], vec![2, 1]] {
            assert!(fixpix(&mut [0.0_f64; 4], &[2, 2], &offsets, &[1, 0]).is_err());
        }
        assert!(fixpix(&mut [0.0_f64; 4], &[3], &[], &[0]).is_err());
        assert!(fixpix::<f64>(&mut [], &[usize::MAX, 2], &[], &[0, 1]).is_err());
    }
}
