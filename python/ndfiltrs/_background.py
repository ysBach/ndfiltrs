"""Mesh (grid) background and RMS estimation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core
from ._robust import _finite_real, _pearson
from ._validation import _integer_index, validate_array

FloatArray = NDArray[np.float32] | NDArray[np.float64]


class MeshBackground(NamedTuple):
    """Result of `mesh_background`.

    Attributes
    ----------
    background, rms : numpy.ndarray
        Full-resolution maps with the shape and dtype of the input.
    mesh_background, mesh_rms : numpy.ndarray
        `float64` mesh grids after fill and median filter.
    mesh_valid : numpy.ndarray
        Boolean grid. `True` where the mesh passed the validity rule (before
        fill).
    mesh_finite_count, mesh_retained_count : numpy.ndarray
        `int64` grids: finite unmasked samples, and samples used by the final
        statistics.
    global_background, global_rms : float
        Medians of the background and RMS grids, respectively.
    """

    background: FloatArray
    rms: FloatArray
    mesh_background: NDArray[np.float64]
    mesh_rms: NDArray[np.float64]
    mesh_valid: NDArray[np.bool_]
    mesh_finite_count: NDArray[np.int64]
    mesh_retained_count: NDArray[np.int64]
    global_background: float
    global_rms: float


def _sizes(value: object, ndim: int, name: str, *, odd: bool) -> tuple[int, ...]:
    """Broadcast a positive integer or validate one entry per axis."""
    message = f"{name} must be an integer or a sequence of integers"
    if isinstance(value, np.ndarray) and value.ndim == 1:
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        sizes = tuple(_integer_index(entry, message) for entry in value)
        if len(sizes) != ndim:
            raise ValueError(f"{name} must have one entry per array dimension")
    else:
        sizes = (_integer_index(value, message),) * ndim
    if any(size <= 0 for size in sizes):
        raise ValueError(f"{name} entries must be positive")
    if odd and any(size % 2 == 0 for size in sizes):
        raise ValueError(f"{name} entries must be odd")
    return sizes


def _choice(value: object, name: str, options: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in options:
        raise ValueError(f"{name} must be one of {options}")
    return value


def mesh_background(
    data: ArrayLike,
    box_size: int | Sequence[int],
    *,
    mask: ArrayLike | None = None,
    filter_size: int | Sequence[int] = 3,
    estimator: Literal["sextractor", "sigma_clip"] = "sextractor",
    sigma: float = 3.0,
    sigma_lower: float | None = None,
    sigma_upper: float | None = None,
    maxiters: int | None = 10,
    center: Literal["median", "mean"] = "median",
    spread: Literal["std", "mad"] = "std",
    statistic: Literal["mode", "mean", "median"] = "mode",
    mode_estimator: Literal["sex", "mmm", "iraf"] = "sex",
    pearson: float | None = None,
    rms: Literal["std", "mad"] = "std",
    min_valid_fraction: float = 0.5,
    valid_count: Literal["finite", "retained"] = "finite",
    valid_base: Literal["mesh", "box"] = "mesh",
    fill: Literal["nearest", "idw"] = "nearest",
    idw_neighbors: int = 10,
    idw_power: float = 1.0,
    filter_edge: Literal["shrink", "truncate"] = "shrink",
    filter_threshold: float | None = None,
    threshold_kind: Literal["deviation", "above"] = "deviation",
    interpolation: Literal["spline", "sextractor"] = "spline",
) -> MeshBackground:
    """Estimate smooth background and RMS maps from a grid of meshes.

    The array is divided into boxes ("meshes"). Each mesh gets a robust
    background and RMS from its finite, unmasked samples. Meshes with too few
    samples are replaced from valid meshes. The mesh grid is median filtered
    and a cubic spline maps it back to full resolution. The defaults follow
    SExtractor; ``docs/background.md`` gives the photutils settings and lists
    every difference from SExtractor, sep, and photutils.

    Parameters
    ----------
    data : array-like
        Aligned, native-endian `float32` or `float64` data with rank at least
        one. NaNs and infinities are never used.
    box_size : int or sequence of int
        Mesh size, scalar or one entry per axis (SExtractor ``BACK_SIZE``, sep
        ``bh, bw``, photutils ``box_size``). The last mesh along an axis is
        partial when the size does not divide the axis length.
    mask : array-like or None, optional
        Array with the shape of `data`. Nonzero entries exclude the sample.
        For sep's ``maskthresh`` use ``mask > maskthresh``.
    filter_size : int or sequence of int, optional
        Odd median-filter size in meshes, scalar or one per axis. Default is
        `3`. `1` on every axis disables the filter.
    estimator : {'sextractor', 'sigma_clip'}, optional
        Mesh statistics. `'sextractor'` (default) runs SExtractor's clipping
        sequence on exact samples and returns the SExtractor mode and the
        standard deviation of the survivors. `'sigma_clip'` clips once as
        `sigma_clipped_filter` and returns `statistic` and `rms`.
    sigma, sigma_lower, sigma_upper : float, optional
        `'sigma_clip'` only: nonnegative clipping multipliers. `None` sides use
        `sigma`. Default is `3.0`.
    maxiters : int or None, optional
        `'sigma_clip'` only: maximum clipping passes. Default is `10`. `None`
        clips until no sample is rejected.
    center : {'median', 'mean'}, optional
        `'sigma_clip'` only: clipping center. Default is `'median'`.
    spread : {'std', 'mad'}, optional
        `'sigma_clip'` only: clipping spread, population standard deviation
        or Gaussian-consistent MAD. Default is `'std'`.
    statistic : {'mode', 'mean', 'median'}, optional
        `'sigma_clip'` only: background statistic of the survivors. Default
        is `'mode'`.
    mode_estimator : {'sex', 'mmm', 'iraf'}, optional
        Mode rule for ``statistic='mode'``, as in `mode_filter`. Default is
        `'sex'`.
    pearson : float or None, optional
        Pearson factor (at least `1`) of the mode rule, also used by
        `'sextractor'`. `None` selects `2.5` for `'sextractor'` and `'sex'`,
        and `3` for `'mmm'` and `'iraf'`.
    rms : {'std', 'mad'}, optional
        `'sigma_clip'` only: RMS statistic of the survivors. Default is
        `'std'`.
    min_valid_fraction : float, optional
        A mesh is valid when its count is at least this fraction of its base
        and positive. Default is `0.5` (SExtractor). photutils uses
        ``1 - exclude_percentile / 100``.
    valid_count : {'finite', 'retained'}, optional
        Count finite unmasked samples (default, SExtractor) or clipping
        survivors (photutils).
    valid_base : {'mesh', 'box'}, optional
        Multiply the fraction by the mesh's own pixel count (default,
        SExtractor) or by the full box volume (photutils).
    fill : {'nearest', 'idw'}, optional
        Replacement for invalid meshes. `'nearest'` (default) averages all
        valid meshes at the smallest distance. `'idw'` weights the
        `idw_neighbors` nearest valid meshes by ``1 / distance**idw_power``,
        including every mesh tied with the last one.
    idw_neighbors : int, optional
        `'idw'` only: number of neighbors. Default is `10` (photutils).
    idw_power : float, optional
        `'idw'` only: nonnegative distance exponent. Default is `1.0`.
    filter_edge : {'shrink', 'truncate'}, optional
        Filter window at grid edges. `'shrink'` (default, SExtractor) keeps
        the window centered by reducing its half width. `'truncate'`
        (photutils) drops positions outside the grid.
    filter_threshold : float or None, optional
        `None` (default) filters every mesh. Otherwise see `threshold_kind`.
    threshold_kind : {'deviation', 'above'}, optional
        `'deviation'` (default, SExtractor ``BACK_FILTTHRESH`` and sep
        ``fthresh``) replaces a mesh when ``|median - value| >=
        filter_threshold``. `'above'` (photutils) replaces meshes with
        ``value > filter_threshold``. The RMS mesh is replaced at the same
        meshes.
    interpolation : {'spline', 'sextractor'}, optional
        `'spline'` (default) is a natural cubic spline along each axis with
        nodes at mesh centers. `'sextractor'` (2-D only) targets the
        interpolation behavior of sep 1.4.1.

    Returns
    -------
    MeshBackground
        Named tuple of full-resolution maps, mesh grids, and global medians.
        Maps and grids are all NaN when no mesh is valid.

    Notes
    -----
    Statistics are computed in `float64`; maps are cast once to the input
    dtype. Results do not depend on the number of threads.

    Examples
    --------
    >>> import numpy as np
    >>> import ndfiltrs as nf
    >>> rng = np.random.default_rng(0)
    >>> image = 100 + rng.normal(0, 5, (256, 256))
    >>> image[100:110, 100:110] += 1000
    >>> result = nf.mesh_background(image, 64)
    >>> result.mesh_background.shape
    (4, 4)
    >>> bool(abs(result.global_background - 100) < 1)
    True
    """
    value = validate_array(data)
    ndim = value.ndim
    boxes = _sizes(box_size, ndim, "box_size", odd=False)
    filters = _sizes(filter_size, ndim, "filter_size", odd=True)
    if mask is not None:
        if np.ma.isMaskedArray(mask):
            raise TypeError("mask must not be a masked array")
        flags = np.asarray(mask)
        if flags.dtype.kind not in "biuf":
            raise TypeError("mask must have a boolean or real numeric dtype")
        if flags.shape != value.shape:
            raise ValueError("mask must have the same shape as data")
        mask = np.ascontiguousarray(flags != 0)
    estimator = _choice(estimator, "estimator", ("sextractor", "sigma_clip"))
    lower = _finite_real(sigma if sigma_lower is None else sigma_lower, "sigma_lower")
    upper = _finite_real(sigma if sigma_upper is None else sigma_upper, "sigma_upper")
    if maxiters is not None:
        maxiters = _integer_index(maxiters, "maxiters must be an integer or None")
        if maxiters < 0:
            raise ValueError("maxiters must be nonnegative")
    mode_estimator = _choice(mode_estimator, "mode_estimator", ("sex", "mmm", "iraf"))
    # The SExtractor estimator always uses the 'sex' rule.
    factor = _pearson("sex" if estimator == "sextractor" else mode_estimator, pearson)
    neighbors = _integer_index(idw_neighbors, "idw_neighbors must be an integer")
    threshold = (
        None
        if filter_threshold is None
        else _finite_real(filter_threshold, "filter_threshold")
    )
    value = np.require(value, dtype=value.dtype, requirements=("C", "A"))
    outputs = _core.mesh_background(
        value,
        mask,
        boxes,
        filters,
        estimator,
        lower,
        upper,
        maxiters,
        _choice(center, "center", ("median", "mean")),
        _choice(spread, "spread", ("std", "mad")),
        _choice(statistic, "statistic", ("mode", "mean", "median")),
        mode_estimator,
        factor,
        _choice(rms, "rms", ("std", "mad")),
        _finite_real(min_valid_fraction, "min_valid_fraction"),
        _choice(valid_count, "valid_count", ("finite", "retained")),
        _choice(valid_base, "valid_base", ("mesh", "box")),
        _choice(fill, "fill", ("nearest", "idw")),
        neighbors,
        _finite_real(idw_power, "idw_power"),
        _choice(filter_edge, "filter_edge", ("shrink", "truncate")),
        threshold,
        _choice(threshold_kind, "threshold_kind", ("deviation", "above")),
        _choice(interpolation, "interpolation", ("spline", "sextractor")),
    )
    return MeshBackground(*outputs)
