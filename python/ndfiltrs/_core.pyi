from typing import Literal

import numpy as np
from numpy.typing import NDArray

BoundaryMode = Literal["reflect", "constant", "nearest", "mirror", "wrap"]

def fixpix(
    array: NDArray[np.float32] | NDArray[np.float64],
    mask: NDArray[np.uint8] | None,
    priority: tuple[int, ...],
) -> NDArray[np.float32] | NDArray[np.float64]: ...
def fixpix_median(
    array: NDArray[np.float32] | NDArray[np.float64],
    mask: NDArray[np.uint8],
    size: tuple[int, ...],
    footprint: list[bool] | None,
    origin: tuple[int, ...],
    mode: tuple[BoundaryMode, ...],
    cval: float,
    ignore_inf: bool,
    min_count: int,
) -> NDArray[np.float32] | NDArray[np.float64]: ...
def fixpix_mean(
    array: NDArray[np.float32] | NDArray[np.float64],
    mask: NDArray[np.uint8],
    weights: NDArray[np.float64],
    mode: tuple[BoundaryMode, ...],
    cval: float,
    origin: tuple[int, ...],
    ignore_inf: bool,
    min_count: int,
) -> NDArray[np.float32] | NDArray[np.float64]: ...
def window_filter(
    array: NDArray[np.float32] | NDArray[np.float64],
    kind: Literal["median", "mean", "min", "max", "var", "std", "rank"],
    size: tuple[int, ...],
    footprint: list[bool] | None,
    origin: tuple[int, ...],
    mode: tuple[BoundaryMode, ...],
    cval: float,
    omit_nan: bool,
    ignore_inf: bool,
    min_count: int,
    parameter: int,
) -> NDArray[np.float32] | NDArray[np.float64]: ...
def convolve(
    array: NDArray[np.float32] | NDArray[np.float64],
    weights: NDArray[np.float64],
    mode: tuple[BoundaryMode, ...],
    cval: float,
    origin: tuple[int, ...],
    correlate: bool,
    omit_nan: bool,
    ignore_inf: bool,
    min_count: int,
    normalize: bool = False,
) -> NDArray[np.float32] | NDArray[np.float64]: ...
def get_num_threads() -> int: ...
def separable_smooth(
    array: NDArray[np.float32] | NDArray[np.float64],
    axes: tuple[int, ...],
    kernels: list[list[float]],
    mode: tuple[str, ...],
    cval: float,
    omit_nan: bool,
    ignore_inf: bool,
    min_count: int,
) -> NDArray[np.float32] | NDArray[np.float64]: ...
def robust_filter(
    array: NDArray[np.float32] | NDArray[np.float64],
    size: tuple[int, ...],
    footprint: list[bool] | None,
    origin: tuple[int, ...],
    mode: tuple[str, ...],
    cval: float,
    kind: str,
    parameter: float,
    omit_nan: bool,
    ignore_inf: bool,
    min_count: int,
    sigma_lower: float,
    sigma_upper: float,
    maxiters: int | None,
    center: str,
    spread: str,
    estimator: str = "sex",
    modify_sample_size: bool = False,
    pearson: float = 2.5,
) -> NDArray[np.float32] | NDArray[np.float64]: ...
