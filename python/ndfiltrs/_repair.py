"""Mask-directed repair of numerical arrays."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core
from ._api import BoundaryMode
from ._linear import KernelArguments, _kernel_arguments
from ._validation import (
    _integer_index,
    _normalize_window,
    validate_array,
    validate_filter_arguments,
)


def fixpix(
    array: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    method: Literal["linear", "median", "mean"] | None = None,
    priority: Sequence[SupportsIndex] | None = None,
    size: int | tuple[int, ...] | None = None,
    footprint: ArrayLike | None = None,
    kernel: ArrayLike | None = None,
    origin: int | tuple[int, ...] | None = None,
    mode: BoundaryMode | tuple[BoundaryMode, ...] | None = None,
    cval: float | None = None,
    ignore_inf: bool | None = None,
    min_count: int | None = None,
    copy: bool = True,
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Repair masked pixels by linear interpolation or a local median or mean.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` data with rank at least
        one. Input data are never modified.
    mask : array-like or None, optional
        Boolean or numeric array with exactly the same shape as `array`.
        Nonzero entries identify bad pixels. `None` means no bad pixels.
        NaNs and infinities in `array` are not automatically masked.
    method : {'linear', 'median', 'mean'} or None, optional
        Repair method. `None` (default) selects `'mean'` when `kernel` is
        given and `'linear'` otherwise.

        `'linear'` preserves the array semantics
        of ``astroimred.fixpix``. `'median'` uses the median of unmasked
        neighbors and evaluates only masked windows.

        `'mean'` uses the mean of unmasked neighbors, like Astropy's
        ``interpolate_replace_nans``. A supplied `kernel` provides the weights.
    priority : sequence of int or None, optional
        Linear only: permutation of all axes, highest priority first.
        This breaks ties between equally short masked runs. Default is reversed
        axis order. The function rejects booleans, duplicate axes, and negative axes.
    size : int or tuple of int or None, optional
        Median and mean: positive window size, scalar or one entry per axis.
        `None` selects `3` unless `footprint` or `kernel` is given.
    footprint : array-like or None, optional
        Median and mean: window with the array's rank. Nonzero entries select samples.
        Pass at most one of `size`, `footprint`, and `kernel`.
    kernel : array-like or None, optional
        Mean only: finite nonnegative weights with the array's rank, for example
        ``kernels.gaussian(1.5, ndim=2)``. Element `k` weights the neighbor at
        offset `k - center`, as in `nancorrelate`. `None` gives equal weights.
    origin : int or tuple of int or None, optional
        Median and mean: window shift per axis, as in `median_filter`.
        `None` selects `0`.
    mode : str or tuple of str or None, optional
        Median and mean: `'reflect'`, `'constant'`, `'nearest'`, `'mirror'`, or
        `'wrap'`, scalar or one per axis. `None` selects `'reflect'`.
    cval : float or None, optional
        Median and mean: constant boundary value. `None` selects NaN, so exterior
        samples are omitted with `'constant'` boundaries.
    ignore_inf : bool or None, optional
        Median and mean: omit infinities when `True`. `None` selects `False`.
        NaNs and originally masked samples are always omitted.
    min_count : int or None, optional
        Median and mean: minimum retained window samples. `None` selects `1`.
        Insufficient support produces NaN. Boundary repetitions and retained
        constant samples count separately, as in ``nanmedian_filter()``.
    copy : bool, optional
        Default `True` always returns an independent copy. `False` permits
        returning ``np.asarray(array)`` when the mask is absent or empty.
        This can alias input storage and retain its layout and read-only flag.

        With bad pixels present, repair still returns a new array. This option
        never enables in-place repair.

    Returns
    -------
    numpy.ndarray
        Same shape and dtype as `array`. Unmasked pixels are unchanged.
        Output is owned, writable, and C-contiguous except when `copy=False`
        returns the input without repair.

    Notes
    -----
    The function validates all options even when no pixels need repair. An option that
    does not apply to the chosen method raises `ValueError` unless it is `None`.

    Linear repair selects each pixel's shortest original masked run and
    interpolates the whole run. Crossing runs preserve repeated C-order writes
    from ``astroimred.fixpix``. Edge runs use their sole good anchor twice.
    Full-axis runs perform no write and do not use another axis as a fallback.

    Arithmetic uses `float64` and casts once to the input dtype, preserving
    legacy nonfinite, signed-zero, and overflow behavior.

    Median repair uses only original unmasked samples: repaired values never
    feed another repair. Even retained counts use the same overflow-safe
    midpoint and signed-zero ordering as ``nanmedian_filter()``.

    Mean repair also uses only original unmasked samples. It divides each
    weighted sum by the weights of the samples it keeps. The result equals
    ``nancorrelate()`` of the data with masked pixels set to NaN. For a
    symmetric kernel this is Astropy's ``interpolate_replace_nans`` with an
    explicit mask.

    Mean repair evaluates only masked pixels. Its cost scales with the number
    of bad pixels rather than the image size. The averages omit unmasked NaNs,
    but those pixels stay NaN in the output.
    Include them in `mask` (for example ``mask | np.isnan(array)``) to repair them.

    Native repair snapshots input data and mask positions before releasing
    the GIL. Concurrent mutation during normalization or snapshots is unsupported.
    FITS and CCDData metadata remain the caller's responsibility.
    """
    value = validate_array(array)
    if method is None:
        method = "linear" if kernel is None else "mean"
    if not isinstance(method, str) or method not in ("linear", "median", "mean"):
        raise ValueError("method must be 'linear', 'median', or 'mean'")
    if not isinstance(copy, (bool, np.bool_)):
        raise TypeError("copy must be a boolean")
    options = {
        "priority": (priority, ("linear",)),
        "size": (size, ("median", "mean")),
        "footprint": (footprint, ("median", "mean")),
        "kernel": (kernel, ("mean",)),
        "origin": (origin, ("median", "mean")),
        "mode": (mode, ("median", "mean")),
        "cval": (cval, ("median", "mean")),
        "ignore_inf": (ignore_inf, ("median", "mean")),
        "min_count": (min_count, ("median", "mean")),
    }
    for name, (option, methods) in options.items():
        if option is not None and method not in methods:
            allowed = " or ".join(repr(entry) for entry in methods)
            raise ValueError(f"{name} requires method={allowed}")
    if ignore_inf is not None and not isinstance(ignore_inf, (bool, np.bool_)):
        raise TypeError("ignore_inf must be a boolean")
    arguments = None
    kernel_arguments: KernelArguments | None = None
    axes = tuple(range(value.ndim - 1, -1, -1))
    if method == "linear":
        if priority is not None:
            if isinstance(priority, (str, bytes)):
                raise TypeError("priority must be a sequence of integer axes")
            try:
                axes = tuple(
                    _integer_index(
                        axis, "priority entries must be integers, not booleans"
                    )
                    for axis in priority
                )
            except TypeError:
                raise TypeError("priority must be a sequence of integer axes") from None
            if len(axes) != value.ndim or set(axes) != set(range(value.ndim)):
                raise ValueError("priority must contain every array axis exactly once")
    elif method == "median":
        arguments = validate_filter_arguments(
            value,
            size,
            "reflect" if mode is None else mode,
            np.nan if cval is None else cval,
            footprint=footprint,
            origin=0 if origin is None else origin,
            min_count=1 if min_count is None else min_count,
            normalize_array=False,
        )
    else:
        if kernel is None:
            shape, selected, _, _ = _normalize_window(size, footprint, 0, value.ndim)
            weights = (
                np.ones(shape)
                if selected is None
                else np.reshape(np.asarray(selected, dtype=np.float64), shape)
            )
        elif size is not None or footprint is not None:
            raise ValueError("pass at most one of size, footprint, and kernel")
        else:
            weights = kernel
        # Validate kernel and options exactly as ``nancorrelate()`` does, even
        # when nothing needs repair. Only the rank and dtype of the probe matter.
        kernel_arguments = _kernel_arguments(
            value[tuple(slice(0, 1) for _ in range(value.ndim))],
            weights,
            "reflect" if mode is None else mode,
            np.nan if cval is None else cval,
            0 if origin is None else origin,
            normalized=True,
            ignore_inf=bool(ignore_inf),
            min_count=1 if min_count is None else min_count,
        )
    bad = None
    if mask is not None:
        if np.ma.isMaskedArray(mask):
            raise TypeError("masked arrays are not supported as masks")
        bad = np.asarray(mask)
        if bad.shape != value.shape:
            raise ValueError("mask must have the same shape as array")
        if bad.dtype.kind not in "buifc":
            raise TypeError("mask must have a boolean or numeric dtype")
    if bad is None or not np.any(bad):
        return value.copy(order="C") if copy else value
    # Read bytes in Rust: noncanonical true bytes in bool views are not valid
    # Rust bool representations. Normalize only after the no-repair return.
    bad = np.ascontiguousarray(bad, dtype=np.bool_).view(np.uint8)
    if kernel_arguments is not None:
        return _core.fixpix_mean(
            value,
            bad,
            kernel_arguments.weights,
            kernel_arguments.filtering.mode,
            kernel_arguments.filtering.cval,
            kernel_arguments.origin,
            bool(ignore_inf),
            kernel_arguments.min_count,
        )
    if arguments is not None:
        return _core.fixpix_median(
            value,
            bad,
            arguments.size,
            arguments.footprint,
            arguments.origin,
            arguments.mode,
            arguments.cval,
            bool(ignore_inf),
            arguments.min_count,
        )
    return _core.fixpix(value, bad, axes)
