//! Neighborhood geometry shared by the window filters.
//!
//! A window is a list of taps: integer offsets from the output position.
//! Taps come from a rectangular size or a boolean footprint, in row-major
//! footprint order, shifted by `origin` with SciPy's convention
//! (tap `k` samples `center + k - (size / 2 + origin)`).

use crate::boundary::{map_index, BoundaryMode};
use crate::error::KernelError;

#[derive(Clone, Debug)]
pub(crate) struct Window {
    shape: Vec<usize>,
    strides: Vec<usize>,
    modes: Vec<BoundaryMode>,
    /// Per-tap, per-axis offsets, stored tap-major (`count * rank` values).
    taps: Vec<isize>,
    /// Per-tap flat offsets, valid when the whole window is inside the array.
    flat_taps: Vec<isize>,
    /// Smallest and largest offset per axis.
    low: Vec<isize>,
    high: Vec<isize>,
    count: usize,
    output_len: usize,
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

impl Window {
    /// Build a window from a footprint shape, an optional C-order boolean
    /// footprint (`None` selects every element), and per-axis origins.
    pub(crate) fn new(
        shape: Vec<usize>,
        window_shape: &[usize],
        footprint: Option<&[bool]>,
        origin: &[isize],
        modes: Vec<BoundaryMode>,
    ) -> Result<Self, KernelError> {
        let rank = shape.len();
        if rank == 0 {
            return Err(KernelError::InvalidConfiguration(
                "filtering requires at least one dimension",
            ));
        }
        if window_shape.len() != rank || origin.len() != rank || modes.len() != rank {
            return Err(KernelError::InvalidConfiguration(
                "array, window, origin, and mode must have equal rank",
            ));
        }
        if window_shape.contains(&0) {
            return Err(KernelError::InvalidConfiguration(
                "window sizes must be positive",
            ));
        }
        let volume = checked_product(window_shape)?;
        if footprint.is_some_and(|values| values.len() != volume) {
            return Err(KernelError::LengthMismatch);
        }
        let mut anchors = reserve::<isize>(rank)?;
        for (&length, &shift) in window_shape.iter().zip(origin) {
            let length = isize::try_from(length).map_err(|_| KernelError::Overflow)?;
            if shift < -(length / 2) || shift > (length - 1) / 2 {
                return Err(KernelError::InvalidConfiguration(
                    "origin must satisfy -(size // 2) <= origin <= (size - 1) // 2",
                ));
            }
            anchors.push(length / 2 + shift);
        }
        let output_len = checked_product(&shape)?;
        let mut strides = reserve::<usize>(rank)?;
        strides.resize(rank, 1);
        for axis in (0..rank - 1).rev() {
            strides[axis] = strides[axis + 1]
                .checked_mul(shape[axis + 1])
                .ok_or(KernelError::Overflow)?;
        }
        let count = footprint.map_or(volume, |values| values.iter().filter(|&&on| on).count());
        if count == 0 {
            return Err(KernelError::InvalidConfiguration(
                "footprint must select at least one element",
            ));
        }
        let mut taps = reserve::<isize>(count.checked_mul(rank).ok_or(KernelError::Overflow)?)?;
        let mut flat_taps = reserve::<isize>(count)?;
        let mut low = reserve::<isize>(rank)?;
        low.resize(rank, isize::MAX);
        let mut high = reserve::<isize>(rank)?;
        high.resize(rank, isize::MIN);
        let mut index = reserve::<usize>(rank)?;
        index.resize(rank, 0);
        for element in 0..volume {
            if footprint.is_none_or(|values| values[element]) {
                let mut flat = 0isize;
                for axis in 0..rank {
                    let offset = isize::try_from(index[axis]).map_err(|_| KernelError::Overflow)?
                        - anchors[axis];
                    let stride =
                        isize::try_from(strides[axis]).map_err(|_| KernelError::Overflow)?;
                    flat = offset
                        .checked_mul(stride)
                        .and_then(|step| flat.checked_add(step))
                        .ok_or(KernelError::Overflow)?;
                    low[axis] = low[axis].min(offset);
                    high[axis] = high[axis].max(offset);
                    taps.push(offset);
                }
                flat_taps.push(flat);
            }
            for axis in (0..rank).rev() {
                index[axis] += 1;
                if index[axis] < window_shape[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
        Ok(Self {
            shape,
            strides,
            modes,
            taps,
            flat_taps,
            low,
            high,
            count,
            output_len,
        })
    }

    /// Build a centered rectangular window.
    #[cfg(any(test, feature = "benchmark", feature = "test-hooks"))]
    pub(crate) fn rectangular(
        shape: Vec<usize>,
        size: &[usize],
        modes: Vec<BoundaryMode>,
    ) -> Result<Self, KernelError> {
        let origin = vec![0isize; size.len()];
        Self::new(shape, size, None, &origin, modes)
    }

    /// Number of selected taps.
    pub(crate) fn count(&self) -> usize {
        self.count
    }

    pub(crate) fn output_len(&self) -> usize {
        self.output_len
    }

    pub(crate) fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Split a row into mapped-boundary and fully interior output runs.
    pub(crate) fn row_run(
        &self,
        input_len: usize,
        center: usize,
        limit: usize,
        index: &mut [usize],
    ) -> Result<(bool, usize), KernelError> {
        let interior = self.prepare(input_len, center, index)?;
        let last = self.rank() - 1;
        let row_left = self.shape[last] - index[last];
        if !interior {
            let other_axes_inside = (0..last).all(|axis| {
                let coordinate = index[axis] as isize;
                coordinate + self.low[axis] >= 0
                    && coordinate + self.high[axis] < self.shape[axis] as isize
            });
            let left_edge = index[last] as isize + self.low[last];
            let width = if other_axes_inside && left_edge < 0 {
                row_left.min((-left_edge) as usize)
            } else {
                row_left
            };
            return Ok((false, limit.min(width)));
        }
        // Stop before crossing either the row or the last interior coordinate.
        let end = (self.shape[last] as isize - self.high[last].max(0)) as usize;
        Ok((true, limit.min(end - index[last])))
    }

    /// Visit each contiguous tap plane of an already validated interior run.
    pub(crate) fn visit_interior_planes<T>(
        &self,
        input: &[T],
        center: usize,
        width: usize,
        mut visit: impl FnMut(usize, &[T]),
    ) {
        for (tap, &offset) in self.flat_taps.iter().enumerate() {
            let source = center.wrapping_add_signed(offset);
            visit(tap, &input[source..source + width]);
        }
    }

    /// Append the window samples around flat position `center` to `values`.
    ///
    /// Samples beyond a `constant` boundary use `cval`. `index` is
    /// caller-owned scratch with one entry per axis.
    pub(crate) fn gather<T: Copy, U>(
        &self,
        input: &[T],
        center: usize,
        cval: T,
        index: &mut [usize],
        values: &mut Vec<U>,
        convert: impl Fn(T) -> U,
    ) -> Result<(), KernelError> {
        if self.prepare(input.len(), center, index)? {
            let base = center as isize;
            values.extend(
                self.flat_taps
                    .iter()
                    .map(|&offset| convert(input[(base + offset) as usize])),
            );
            Ok(())
        } else {
            self.visit_mapped(input, cval, index, |value| values.push(convert(value)))
        }
    }

    /// Visit samples in tap order, without a temporary window buffer.
    pub(crate) fn visit<T: Copy>(
        &self,
        input: &[T],
        center: usize,
        cval: T,
        index: &mut [usize],
        mut consume: impl FnMut(T),
    ) -> Result<(), KernelError> {
        if self.prepare(input.len(), center, index)? {
            let base = center as isize;
            for &offset in &self.flat_taps {
                consume(input[(base + offset) as usize]);
            }
            Ok(())
        } else {
            self.visit_mapped(input, cval, index, consume)
        }
    }

    /// Decode the center and test whether all taps avoid boundary mapping.
    fn prepare(
        &self,
        input_len: usize,
        center: usize,
        index: &mut [usize],
    ) -> Result<bool, KernelError> {
        if center >= self.output_len || input_len != self.output_len || index.len() != self.rank() {
            return Err(KernelError::LengthMismatch);
        }
        let mut remainder = center;
        let mut interior = true;
        for (axis, coordinate) in index.iter_mut().enumerate() {
            *coordinate = remainder / self.strides[axis];
            remainder %= self.strides[axis];
            // Both bounds fit in isize: indices and offsets are below isize::MAX.
            let position = *coordinate as isize;
            interior &= position + self.low[axis] >= 0
                && position + self.high[axis] < self.shape[axis] as isize;
        }
        Ok(interior)
    }

    fn visit_mapped<T: Copy>(
        &self,
        input: &[T],
        cval: T,
        index: &[usize],
        mut consume: impl FnMut(T),
    ) -> Result<(), KernelError> {
        for tap in self.taps.chunks_exact(self.rank()) {
            let mut source = 0usize;
            let mut outside = false;
            for axis in 0..self.rank() {
                let coordinate = index[axis] as i128 + tap[axis] as i128;
                match map_index(coordinate, self.shape[axis], self.modes[axis])? {
                    Some(mapped) => source += mapped * self.strides[axis],
                    None => {
                        outside = true;
                        break;
                    }
                }
            }
            consume(if outside { cval } else { input[source] });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Window;
    use crate::boundary::BoundaryMode::{Constant, Nearest, Reflect};

    fn samples(window: &Window, input: &[f64], center: usize) -> Vec<f64> {
        let mut index = vec![0; window.rank()];
        let mut values = Vec::new();
        window
            .gather(input, center, -1.0, &mut index, &mut values, |value| value)
            .unwrap();
        values
    }

    #[test]
    fn rectangular_taps_follow_row_major_order_and_boundaries() {
        let input: Vec<f64> = (0..9).map(f64::from).collect();
        let window = Window::rectangular(vec![3, 3], &[3, 3], vec![Constant, Nearest]).unwrap();
        assert_eq!(window.count(), 9);
        assert_eq!(
            samples(&window, &input, 4),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(
            samples(&window, &input, 0),
            vec![-1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 3.0, 3.0, 4.0]
        );
    }

    #[test]
    fn footprint_and_origin_select_shifted_taps() {
        let input: Vec<f64> = (0..6).map(f64::from).collect();
        // Even footprint [1, 0, 1, 1]: anchor 2 + origin -1 = 1, offsets -1, 1, 2.
        let window = Window::new(
            vec![6],
            &[4],
            Some(&[true, false, true, true]),
            &[-1],
            vec![Reflect],
        )
        .unwrap();
        assert_eq!(window.count(), 3);
        assert_eq!(samples(&window, &input, 2), vec![1.0, 3.0, 4.0]);
        assert_eq!(samples(&window, &input, 5), vec![4.0, 5.0, 4.0]);
    }

    #[test]
    fn invalid_geometry_is_rejected() {
        assert!(Window::rectangular(vec![], &[], vec![]).is_err());
        assert!(Window::rectangular(vec![3], &[0], vec![Reflect]).is_err());
        assert!(Window::new(vec![3], &[3], Some(&[false; 3]), &[0], vec![Reflect]).is_err());
        assert!(Window::new(vec![3], &[3], Some(&[true; 2]), &[0], vec![Reflect]).is_err());
        assert!(Window::new(vec![3], &[3], None, &[2], vec![Reflect]).is_err());
        assert!(Window::new(vec![3], &[4], None, &[-2], vec![Reflect]).is_ok());
        assert!(Window::new(vec![3], &[4], None, &[2], vec![Reflect]).is_err());
    }
}
