use crate::error::KernelError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BoundaryMode {
    Reflect,
    Constant,
    Nearest,
    Mirror,
    Wrap,
}

pub(crate) fn map_index(
    index: i128,
    length: usize,
    mode: BoundaryMode,
) -> Result<Option<usize>, KernelError> {
    if length == 0 {
        return Err(KernelError::EmptyAxis);
    }
    let length = i128::try_from(length).map_err(|_| KernelError::Overflow)?;
    if (0..length).contains(&index) {
        return usize::try_from(index)
            .map(Some)
            .map_err(|_| KernelError::Overflow);
    }
    let mapped = match mode {
        BoundaryMode::Constant => return Ok(None),
        BoundaryMode::Nearest => index.clamp(0, length - 1),
        BoundaryMode::Wrap => index.rem_euclid(length),
        BoundaryMode::Reflect if length == 1 => 0,
        BoundaryMode::Reflect => {
            let period = length.checked_mul(2).ok_or(KernelError::Overflow)?;
            let reduced = index.rem_euclid(period);
            if reduced < length {
                reduced
            } else {
                period - 1 - reduced
            }
        }
        BoundaryMode::Mirror if length == 1 => 0,
        BoundaryMode::Mirror => {
            let period = (length - 1).checked_mul(2).ok_or(KernelError::Overflow)?;
            let reduced = index.rem_euclid(period);
            if reduced < length {
                reduced
            } else {
                period - reduced
            }
        }
    };
    usize::try_from(mapped)
        .map(Some)
        .map_err(|_| KernelError::Overflow)
}

#[cfg(test)]
mod tests {
    use super::{map_index, BoundaryMode};

    #[test]
    fn boundary_modes_map_literal_coordinates() {
        let coordinates = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8];
        let cases = [
            (
                BoundaryMode::Reflect,
                vec![3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0]
                    .into_iter()
                    .map(Some)
                    .collect::<Vec<_>>(),
            ),
            (
                BoundaryMode::Mirror,
                vec![1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2]
                    .into_iter()
                    .map(Some)
                    .collect::<Vec<_>>(),
            ),
            (
                BoundaryMode::Nearest,
                vec![0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3]
                    .into_iter()
                    .map(Some)
                    .collect::<Vec<_>>(),
            ),
            (
                BoundaryMode::Wrap,
                vec![3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0]
                    .into_iter()
                    .map(Some)
                    .collect::<Vec<_>>(),
            ),
        ];
        for (mode, expected) in cases {
            let actual = coordinates
                .iter()
                .map(|&index| map_index(index, 4, mode).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "mode={mode:?}");
        }
    }

    #[test]
    fn constant_marks_only_out_of_bounds_coordinates() {
        let actual = (-2..=3)
            .map(|index| map_index(index, 2, BoundaryMode::Constant).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(actual, vec![None, None, Some(0), Some(1), None, None]);
    }

    #[test]
    fn nonconstant_modes_map_length_two_axes() {
        let coordinates = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6];
        let cases = [
            (
                BoundaryMode::Reflect,
                vec![0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            ),
            (
                BoundaryMode::Mirror,
                vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            ),
            (
                BoundaryMode::Nearest,
                vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            ),
            (BoundaryMode::Wrap, vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        ];
        for (mode, expected) in cases {
            let actual = coordinates
                .iter()
                .map(|&index| map_index(index, 2, mode).unwrap().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "mode={mode:?}");
        }
    }

    #[test]
    fn singleton_axes_are_defined_for_every_nonconstant_mode() {
        for mode in [
            BoundaryMode::Reflect,
            BoundaryMode::Nearest,
            BoundaryMode::Mirror,
            BoundaryMode::Wrap,
        ] {
            for index in [-1_000_000, -1, 0, 1, 1_000_000] {
                assert_eq!(map_index(index, 1, mode).unwrap(), Some(0));
            }
        }
    }

    #[test]
    fn empty_axes_are_rejected_before_mapping() {
        assert!(map_index(0, 0, BoundaryMode::Reflect).is_err());
    }

    #[test]
    fn periodic_modes_map_coordinates_after_many_periods() {
        assert_eq!(
            map_index(1_000_000_003, 4, BoundaryMode::Reflect).unwrap(),
            Some(3)
        );
        assert_eq!(
            map_index(-1_000_000_003, 4, BoundaryMode::Mirror).unwrap(),
            Some(1)
        );
        assert_eq!(
            map_index(-1_000_000_003, 4, BoundaryMode::Wrap).unwrap(),
            Some(1)
        );
    }
}
