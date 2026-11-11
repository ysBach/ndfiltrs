"""Per-window NumPy least-squares reference for Savitzky-Golay filters."""

from __future__ import annotations

import itertools
import math

import numpy as np
from numpy.typing import NDArray


def basis_exponents(
    window: tuple[int, ...], basis: str, degree: int | tuple[int, ...]
) -> list[tuple[int, ...]]:
    """Return monomial exponents; unfiltered axes keep exponent zero."""
    if basis == "total":
        bounds = [degree if length > 1 else 0 for length in window]
        return [
            exponents
            for exponents in itertools.product(*(range(b + 1) for b in bounds))
            if sum(exponents) <= degree
        ]
    return list(itertools.product(*(range(d + 1) for d in degree)))


def _map_index(index: int, length: int, mode: str) -> int | None:
    """Map one coordinate with SciPy ndimage boundary rules."""
    if 0 <= index < length:
        return index
    if mode == "constant":
        return None
    if mode == "nearest":
        return min(max(index, 0), length - 1)
    if mode == "wrap":
        return index % length
    if length == 1:
        return 0
    if mode == "reflect":
        reduced = index % (2 * length)
        return reduced if reduced < length else 2 * length - 1 - reduced
    period = 2 * (length - 1)
    reduced = index % period
    return reduced if reduced < length else period - reduced


def savgol_reference(
    array: NDArray[np.float64],
    window: tuple[int, ...],
    basis: str,
    degree: int | tuple[int, ...],
    deriv: tuple[int, ...],
    delta: tuple[float, ...],
    modes: tuple[str, ...],
    cval: float = 0.0,
    *,
    omit: bool = True,
    ignore_inf: bool = False,
    min_count: int = 1,
    weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Fit each window with ``numpy.linalg.lstsq`` after an explicit rank test.

    All tuple arguments have one entry per array axis. A window whose
    weighted design has ``s_min <= s_max * max(n, r) * eps`` gives NaN,
    because ``lstsq`` alone returns a minimum-norm answer for such windows.
    """
    half = [length // 2 for length in window]
    scale = [1.0 / h if h else 1.0 for h in half]
    exponents = np.array(basis_exponents(window, basis, degree), dtype=float)
    offsets = np.array(
        list(itertools.product(*(range(-h, h + 1) for h in half))), dtype=int
    )
    w = np.ones(len(offsets)) if weights is None else np.asarray(weights).ravel()
    w = w / w.max()
    factor = math.prod(
        (s / d) ** k for s, d, k in zip(scale, delta, deriv, strict=True)
    )
    output = np.empty(array.shape)
    for position in itertools.product(*(range(n) for n in array.shape)):
        centers = [
            min(max(i, h), n - 1 - h) if mode == "interp" else i
            for i, h, n, mode in zip(position, half, array.shape, modes, strict=True)
        ]
        samples = np.empty(len(offsets))
        for row, offset in enumerate(offsets):
            source = []
            for c, d, n, mode in zip(centers, offset, array.shape, modes, strict=True):
                source.append(
                    _map_index(c + d, n, "nearest" if mode == "interp" else mode)
                )
            samples[row] = cval if None in source else array[tuple(source)]
        used = w > 0
        invalid = np.isnan(samples) | (ignore_inf & np.isinf(samples))
        if not omit and np.any(invalid & used):
            output[position] = np.nan
            continue
        keep = used & ~invalid
        t = offsets[keep] * scale
        design = np.prod(t[:, None, :] ** exponents[None, :, :], axis=2)
        root = np.sqrt(w[keep])[:, None]
        n, r = design.shape
        if n < max(min_count, r):
            output[position] = np.nan
            continue
        singular = np.linalg.svd(root * design, compute_uv=False)
        if (
            singular.max() == 0
            or singular.min() <= singular.max() * max(n, r) * np.finfo(float).eps
        ):
            output[position] = np.nan
            continue
        with np.errstate(invalid="ignore"):
            coefficients = np.linalg.lstsq(
                root * design, root[:, 0] * samples[keep], rcond=None
            )[0]
        at = (np.array(position) - np.array(centers)) * scale
        functional = np.ones(len(exponents)) * factor
        for axis, k in enumerate(deriv):
            power = exponents[:, axis]
            falling = np.array([math.perm(int(p), k) if p >= k else 0 for p in power])
            functional *= falling * np.where(
                power >= k, at[axis] ** np.maximum(power - k, 0), 0
            )
        output[position] = functional @ coefficients
    return output
