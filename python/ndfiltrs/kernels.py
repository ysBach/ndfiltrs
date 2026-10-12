"""Pure-NumPy factories for sampled N-dimensional convolution kernels."""

from __future__ import annotations

import itertools
import math
import numbers
import operator
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _integer(value: object, name: str) -> int:
    """Return an integer index while rejecting booleans and non-integers."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not a boolean")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _real(value: object, name: str, *, positive: bool) -> float:
    """Return a finite real number with the requested sign constraint."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, not a boolean")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _real_values(value: object, name: str) -> tuple[tuple[float, ...], bool]:
    """Read a real scalar or one-dimensional sequence of real values."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must contain real numbers, not booleans")
    if isinstance(value, numbers.Real):
        return (_real(value, name, positive=True),), True
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a real number or a sequence of them")
    try:
        entries = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be a real number or a sequence of them") from exc
    if not entries:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(_real(entry, name, positive=True) for entry in entries), False


def _dimension(ndim: object | None, inferred: int | None) -> int:
    """Validate or infer a positive kernel dimension."""
    if ndim is None:
        if inferred is None:
            raise TypeError("ndim is required when the parameter is scalar")
        result = inferred
    else:
        result = _integer(ndim, "ndim")
    if result < 1:
        raise ValueError("ndim must be at least one")
    if inferred is not None and result != inferred:
        raise ValueError("parameter lengths and rotation dimensions must match ndim")
    return result


def _rotation_matrix(rotation: ArrayLike | None) -> NDArray[np.float64] | None:
    """Validate an orthogonal matrix whose columns are principal axes."""
    if rotation is None:
        return None
    raw = np.asarray(rotation)
    if (
        raw.ndim != 2
        or raw.shape[0] < 1
        or raw.shape[0] != raw.shape[1]
        or not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.complexfloating)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise TypeError("rotation must be a square real numeric matrix")
    matrix = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("rotation must contain only finite values")
    identity = np.eye(matrix.shape[0], dtype=np.float64)
    # allclose allows the round-off produced by floating-point rotations.
    if not np.allclose(matrix.T @ matrix, identity):
        raise ValueError("rotation must be orthogonal")
    return matrix


def _geometry(
    widths: object,
    ndim: int | None,
    rotation: ArrayLike | None,
) -> tuple[tuple[float, ...], NDArray[np.float64]]:
    """Resolve principal-axis widths and an orthogonal rotation matrix."""
    values, scalar = _real_values(widths, "widths")
    matrix = _rotation_matrix(rotation)
    inferred: int | None = None
    if matrix is not None:
        inferred = matrix.shape[0]
    elif not scalar:
        inferred = len(values)
    dimension = _dimension(ndim, inferred)
    if not scalar and len(values) != dimension:
        raise ValueError("widths must have one entry per principal axis")
    selected_widths = values * dimension if scalar else values
    if matrix is None:
        matrix = np.eye(dimension, dtype=np.float64)
    elif matrix.shape != (dimension, dimension):
        raise ValueError("rotation shape must match ndim")
    return selected_widths, matrix


def _radii(radius: object, ndim: int) -> tuple[int, ...]:
    """Validate a nonnegative scalar or per-axis support radius."""
    if isinstance(radius, (bool, np.bool_)):
        raise TypeError("radius must contain integers, not booleans")
    if isinstance(radius, numbers.Integral):
        values = (_integer(radius, "radius"),) * ndim
    elif isinstance(radius, (str, bytes)):
        raise TypeError("radius must be an integer or a sequence of integers")
    else:
        try:
            entries = tuple(radius)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                "radius must be an integer or a sequence of integers"
            ) from exc
        if len(entries) != ndim:
            raise ValueError("radius must have one entry per axis")
        values = tuple(_integer(entry, "radius entries") for entry in entries)
    if any(value < 0 for value in values):
        raise ValueError("radius entries must be nonnegative")
    return values


def _sampling(oversample: object, normalize: object) -> tuple[int, bool]:
    """Validate the sampling factor and normalization flag."""
    factor = _integer(oversample, "oversample")
    if factor < 1:
        raise ValueError("oversample must be at least one")
    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError("normalize must be a boolean")
    return factor, bool(normalize)


def _finish(kernel: NDArray[np.float64], normalize: bool) -> NDArray[np.float64]:
    """Normalize sampled weights when requested and return C-contiguous float64."""
    result = np.asarray(kernel, dtype=np.float64)
    if normalize:
        with np.errstate(over="ignore", invalid="ignore"):
            total = float(np.sum(result, dtype=np.float64))
        if not math.isfinite(total) or total <= 0:
            raise ValueError("kernel weights must have a positive finite sum")
        result = result / total
    return np.ascontiguousarray(result, dtype=np.float64)


def _subpixel_shifts(oversample: int, ndim: int) -> itertools.product:
    """Yield every subpixel-center offset in each array axis."""
    shifts = tuple((index + 0.5) / oversample - 0.5 for index in range(oversample))
    return itertools.product(shifts, repeat=ndim)


def _axis_offsets(radii: tuple[int, ...]) -> list[NDArray[np.float64]]:
    """Return broadcastable pixel offsets from the kernel center, axis 0 first."""
    dimension = len(radii)
    return [
        np.arange(-extent, extent + 1, dtype=np.float64).reshape(
            tuple(2 * extent + 1 if axis == index else 1 for axis in range(dimension))
        )
        for index, extent in enumerate(radii)
    ]


def _sample(
    radii: tuple[int, ...],
    oversample: int,
    profile: str,
    *,
    widths: tuple[float, ...] | None = None,
    rotation: NDArray[np.float64] | None = None,
    parameter: float | None = None,
    inner_radius: float | None = None,
    outer_radius: float | None = None,
) -> NDArray[np.float64]:
    """Average a radial or ellipsoidal profile over subpixel centers."""
    dimension = len(radii)
    shape = tuple(2 * extent + 1 for extent in radii)
    axis_offsets = _axis_offsets(radii)
    accumulated = np.zeros(shape, dtype=np.float64)
    for shifts in _subpixel_shifts(oversample, dimension):
        coordinates = [
            offset + shift for offset, shift in zip(axis_offsets, shifts, strict=True)
        ]
        if profile in ("ball", "ring"):
            distance2 = np.zeros(shape, dtype=np.float64)
            for coordinate in coordinates:
                distance2 += coordinate * coordinate
            if profile == "ball":
                assert outer_radius is not None
                values = distance2 <= outer_radius * outer_radius
            else:
                assert inner_radius is not None and outer_radius is not None
                values = (distance2 >= inner_radius * inner_radius) & (
                    distance2 <= outer_radius * outer_radius
                )
            accumulated += values
            continue

        assert widths is not None and rotation is not None and parameter is not None
        distance2 = np.zeros(shape, dtype=np.float64)
        for principal_axis, width in enumerate(widths):
            coordinate = np.zeros(shape, dtype=np.float64)
            for array_axis, values in enumerate(coordinates):
                coordinate += rotation[array_axis, principal_axis] * values
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                coordinate /= width
                distance2 += coordinate * coordinate
        if profile == "gaussian":
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                values = np.exp(-0.5 * distance2)
        elif profile == "moffat":
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                values = np.power(1.0 + distance2, -parameter)
        else:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                values = np.power(
                    1.0 + distance2 / parameter,
                    -(parameter + dimension) / 2.0,
                )
        accumulated += values
    accumulated /= oversample**dimension
    return accumulated


def _smallest_support(
    kernel: NDArray[np.float64], radii: tuple[int, ...]
) -> NDArray[np.float64]:
    """Crop zero outer layers so every nonzero sample remains in the kernel."""
    locations = np.argwhere(kernel != 0)
    if locations.size == 0:
        raise ValueError(
            "no subpixel sample lies inside the region; widen it or increase oversample"
        )
    support = tuple(
        int(np.max(np.abs(locations[:, axis] - extent)))
        for axis, extent in enumerate(radii)
    )
    return kernel[
        tuple(
            slice(extent - keep, extent + keep + 1)
            for extent, keep in zip(radii, support, strict=True)
        )
    ]


def box(
    size: int | Sequence[int],
    ndim: int | None = None,
    *,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Return a uniform N-dimensional box kernel.

    Parameters
    ----------
    size : int or sequence of int
        Pixel length on each axis. A scalar requires `ndim`.
        Lengths may be even.
    ndim : int, optional
        Number of axes when `size` is scalar.
    normalize : bool, default=True
        Divide the weights by their sum.

    Returns
    -------
    numpy.ndarray
        C-contiguous `float64` weights.
    """
    if isinstance(size, (bool, np.bool_)):
        raise TypeError("size must contain integers, not booleans")
    if isinstance(size, numbers.Integral):
        if ndim is None:
            raise TypeError("ndim is required when size is scalar")
        dimension = _dimension(ndim, None)
        sizes = (_integer(size, "size"),) * dimension
    elif isinstance(size, (str, bytes)):
        raise TypeError("size must be an integer or a sequence of integers")
    else:
        try:
            entries = tuple(size)
        except TypeError as exc:
            raise TypeError(
                "size must be an integer or a sequence of integers"
            ) from exc
        if not entries:
            raise ValueError("size must contain at least one value")
        dimension = _dimension(ndim, len(entries))
        sizes = tuple(_integer(entry, "size entries") for entry in entries)
    if any(length < 1 for length in sizes):
        raise ValueError("size entries must be positive")
    _, normalize = _sampling(1, normalize)
    return _finish(np.ones(sizes, dtype=np.float64), normalize)


def tophat(
    radius: float,
    ndim: int,
    *,
    oversample: int = 1,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Return pixel fractions inside an N-dimensional ball.

    This is Astropy's top-hat (disk) kernel. For morphological top-hat
    filtering, see `white_tophat` and `black_tophat`.

    Parameters
    ----------
    radius : float
        Ball radius in pixels. The pixel fraction includes points on the ball
        boundary.
    ndim : int
        Number of array axes, with axis 0 first.
    oversample : int, default=1
        Average this many subpixel centers per axis.
    normalize : bool, default=True
        Divide the pixel fractions by their sum.

    Returns
    -------
    numpy.ndarray
        C-contiguous `float64` weights with the smallest sampled support.
    """
    radius_value = _real(radius, "radius", positive=False)
    dimension = _dimension(ndim, None)
    factor, normalize = _sampling(oversample, normalize)
    extent = max(0, math.floor(radius_value + 0.5 - 0.5 / factor))
    radii = (extent,) * dimension
    sampled = _sample(
        radii,
        factor,
        "ball",
        outer_radius=radius_value,
    )
    return _finish(_smallest_support(sampled, radii), normalize)


def ring(
    radius_in: float,
    radius_out: float,
    ndim: int,
    *,
    oversample: int = 1,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Return sampled pixel fractions in an inclusive N-dimensional shell.

    Parameters
    ----------
    radius_in, radius_out : float
        Inner and outer shell radii in pixels, with both boundaries included.
        Astropy `Ring2DKernel(radius_in, width)` uses
        `radius_out = radius_in + width`.
    ndim : int
        Number of array axes, with axis 0 first.
    oversample : int, default=1
        Average this many subpixel centers per axis.
    normalize : bool, default=True
        Divide the sampled fractions by their sum.

    Returns
    -------
    numpy.ndarray
        C-contiguous `float64` weights with the smallest sampled support.
    """
    inner = _real(radius_in, "radius_in", positive=False)
    outer = _real(radius_out, "radius_out", positive=False)
    if outer < inner:
        raise ValueError("radius_out must be greater than or equal to radius_in")
    dimension = _dimension(ndim, None)
    factor, normalize = _sampling(oversample, normalize)
    extent = max(0, math.floor(outer + 0.5 - 0.5 / factor))
    radii = (extent,) * dimension
    sampled = _sample(
        radii,
        factor,
        "ring",
        inner_radius=inner,
        outer_radius=outer,
    )
    return _finish(_smallest_support(sampled, radii), normalize)


def gaussian(
    sigma: float | Sequence[float],
    *,
    ndim: int | None = None,
    rotation: ArrayLike | None = None,
    radius: int | Sequence[int] | None = None,
    truncate: float = 4.0,
    oversample: int = 1,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Return a sampled Gaussian kernel with optional rotated principal axes.

    Parameters
    ----------
    sigma : float or sequence of float
        Positive principal-axis standard deviations in pixels. A scalar needs
        `ndim` unless `rotation` supplies it.
    ndim : int, optional
        Number of array axes. If omitted, the function infers it from a width
        sequence or `rotation`.
    rotation : array-like, optional
        Orthogonal matrix whose columns are principal axes in array-axis order.
        The function checks orthogonality with `numpy.allclose` to allow
        floating-point round-off. Without a matrix, principal axes align with
        array axes.
    radius : int or sequence of int, optional
        Nonnegative support radius per array axis. By default each radius is
        `ceil(truncate * sqrt(Sigma_ii))`, where
        `Sigma = rotation @ diag(sigma**2) @ rotation.T`.
    truncate : float, default=4.0
        Nonnegative support extent in marginal standard deviations.
    oversample : int, default=1
        Average this many subpixel centers per axis.
    normalize : bool, default=True
        Divide the sampled profile by its sum.

    Returns
    -------
    numpy.ndarray
        C-contiguous `float64` weights on an odd-sized support.
    """
    widths, matrix = _geometry(sigma, ndim, rotation)
    dimension = len(widths)
    truncation = _real(truncate, "truncate", positive=False)
    if radius is None:
        marginal_widths = tuple(
            math.hypot(
                *(
                    matrix[axis, principal] * widths[principal]
                    for principal in range(dimension)
                )
            )
            for axis in range(dimension)
        )
        radii = tuple(math.ceil(truncation * width) for width in marginal_widths)
    else:
        radii = _radii(radius, dimension)
    factor, normalize = _sampling(oversample, normalize)
    return _finish(
        _sample(
            radii, factor, "gaussian", widths=widths, rotation=matrix, parameter=1.0
        ),
        normalize,
    )


def moffat(
    alpha: float | Sequence[float],
    beta: float,
    *,
    ndim: int | None = None,
    rotation: ArrayLike | None = None,
    radius: int | Sequence[int],
    oversample: int = 1,
    normalize: bool = True,
) -> NDArray[np.float64]:
    r"""Return a sampled Moffat kernel with explicitly finite support.

    Parameters
    ----------
    alpha : float or sequence of float
        Positive core widths in pixels along the principal axes. A scalar
        applies to every axis. Scale each principal coordinate by its core
        width. Let `d` be the Euclidean distance of those scaled coordinates.
        The profile is ``(1 + d**2)**(-beta)``.
    beta : float
        Positive power-law exponent.
    ndim : int, optional
        Number of array axes. If omitted, the function infers it from
        `rotation` when supplied.
    rotation : array-like, optional
        Orthogonal matrix whose columns are principal axes in array-axis order.
        The function checks orthogonality with `numpy.allclose` to allow
        floating-point round-off.
    radius : int or sequence of int
        Required nonnegative support radius per array axis. It sets the finite
        support for this heavy-tailed profile.
    oversample : int, default=1
        Average this many subpixel centers per axis.
    normalize : bool, default=True
        Divide the sampled profile by its sum.

    Returns
    -------
    numpy.ndarray
        C-contiguous `float64` weights on an odd-sized support.

    Notes
    -----
    Along each principal axis, the full width at half maximum is
    ``2 * alpha * sqrt(2**(1 / beta) - 1)``. Astropy's
    `Moffat2DKernel(gamma, alpha)` maps to `alpha=gamma`, `beta=alpha` here.
    """
    exponent = _real(beta, "beta", positive=True)
    widths, matrix = _geometry(alpha, ndim, rotation)
    radii = _radii(radius, len(widths))
    factor, normalize = _sampling(oversample, normalize)
    return _finish(
        _sample(
            radii,
            factor,
            "moffat",
            widths=widths,
            rotation=matrix,
            parameter=exponent,
        ),
        normalize,
    )


def student_t(
    scale: float | Sequence[float],
    df: float,
    *,
    ndim: int | None = None,
    rotation: ArrayLike | None = None,
    radius: int | Sequence[int],
    oversample: int = 1,
    normalize: bool = True,
) -> NDArray[np.float64]:
    r"""Return a sampled multivariate Student-t kernel with finite support.

    Parameters
    ----------
    scale : float or sequence of float
        Positive principal-axis scales in pixels. A scalar needs `ndim` unless
        `rotation` supplies it.
    df : float
        Positive degrees of freedom.
    ndim : int, optional
        Number of array axes. If omitted, the function infers it from a scale
        sequence or `rotation`.
    rotation : array-like, optional
        Orthogonal matrix whose columns are principal axes in array-axis order.
        The function checks orthogonality with `numpy.allclose` to allow
        floating-point round-off.
    radius : int or sequence of int
        Required nonnegative support radius per array axis. It sets the finite
        support for this heavy-tailed profile.
    oversample : int, default=1
        Average this many subpixel centers per axis.
    normalize : bool, default=True
        Divide the sampled profile by its sum.

    Returns
    -------
    numpy.ndarray
        C-contiguous `float64` weights on an odd-sized support.

    Notes
    -----
    `df=1` gives a multivariate Cauchy profile (the Lorentzian in 1-D).
    It matches a Moffat profile
    with ``alpha**2 = df * scale**2`` and ``beta = (df + ndim) / 2``.
    """
    degrees = _real(df, "df", positive=True)
    widths, matrix = _geometry(scale, ndim, rotation)
    radii = _radii(radius, len(widths))
    factor, normalize = _sampling(oversample, normalize)
    return _finish(
        _sample(
            radii,
            factor,
            "student_t",
            widths=widths,
            rotation=matrix,
            parameter=degrees,
        ),
        normalize,
    )


def sample(
    function: Callable[..., ArrayLike],
    radius: int | Sequence[int],
    *,
    ndim: int | None = None,
    oversample: int = 1,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Evaluate any function on a pixel grid and return it as kernel weights.

    Use this for profiles without a dedicated factory: Sersic, Airy, Voigt,
    King, exponential, window functions, or any Astropy functional model.

    Parameters
    ----------
    function : callable
        The function calls ``function(*offsets)`` with one `float64` array per
        axis. Axis 0 comes first. Each array holds pixel offsets from the kernel
        center. The arrays broadcast to the kernel shape. The result must also
        broadcast to that shape and contain only finite values.
    radius : int or sequence of int
        Half-width per axis. The kernel has ``2 * radius + 1`` pixels per axis.
        A scalar requires `ndim`.
    ndim : int, optional
        Number of axes when `radius` is scalar.
    oversample : int, default=1
        Average ``oversample**ndim`` evaluations per pixel, at subpixel
        centers ``(j + 0.5) / oversample - 0.5``.
    normalize : bool, default=True
        Divide the weights by their sum. The sum must be positive when
        `normalize=True`.

    Returns
    -------
    numpy.ndarray
        C-contiguous `float64` weights.

    Notes
    -----
    Kernel construction calls the function once for each subpixel shift.
    Filtering never calls it. Identical weights have the same cost for the
    same filter. A dense sampled kernel does not benefit from a specialized
    separable filter.

    Astropy 2-D models expect `x` along axis 1 and `y` along axis 0. Pass a
    model to `sample` as ``sample(lambda y, x: model(x, y), radius=(10, 10))``.
    Set `normalize=False` for zero-sum kernels such as a Ricker wavelet.

    Examples
    --------
    >>> import numpy as np
    >>> from ndfiltrs import kernels
    >>> exponential = kernels.sample(
    ...     lambda y, x: np.exp(-np.hypot(y, x) / 2.0), radius=6, ndim=2
    ... )
    >>> exponential.shape
    (13, 13)
    """
    if not callable(function):
        raise TypeError("function must be callable")
    try:
        inferred = None if isinstance(radius, numbers.Integral) else len(radius)
    except TypeError:
        raise TypeError("radius must be an integer or a sequence of integers") from None
    dimension = _dimension(ndim, inferred)
    radii = _radii(radius, dimension)
    factor, normalize = _sampling(oversample, normalize)
    shape = tuple(2 * extent + 1 for extent in radii)
    offsets = _axis_offsets(radii)
    accumulated = np.zeros(shape, dtype=np.float64)
    for shifts in _subpixel_shifts(factor, dimension):
        values = np.asarray(
            function(
                *(offset + shift for offset, shift in zip(offsets, shifts, strict=True))
            ),
            dtype=np.float64,
        )
        try:
            accumulated += np.broadcast_to(values, shape)
        except ValueError as error:
            raise ValueError(
                f"function output with shape {values.shape} does not broadcast "
                f"to the kernel shape {shape}"
            ) from error
    if not np.isfinite(accumulated).all():
        raise ValueError("function values must be finite")
    return _finish(accumulated / factor**dimension, normalize)
