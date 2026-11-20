# Variable window means

`variable_mean_filter` and `nanvariable_mean_filter` average each pixel over
its own rectangular window. A second integer array, with the input's shape,
gives the window of every pixel. Use them when the right smoothing scale
changes across the image, for example a window that grows with distance from
a comet nucleus.

```python
import numpy as np
import ndfiltrs as nf

y, x = np.indices(image.shape)
rho = np.hypot(x - x0, y - y0)  # distance from the nucleus
radius = np.floor(1 + 0.05 * rho).astype(int)  # choose the scale law yourself
smooth = nf.nanvariable_mean_filter(image, radius=radius, mode="nearest")
```

## Window maps: `size` or `radius`

Pass exactly one of them. Both need an integer dtype. Round or clip a
floating-point map before the call: the filter does not round for you.

- `radius=r` gives the centered window size `2 * r + 1`. `r = 0` keeps the
  pixel itself. Entries must be nonnegative.
- `size=s` gives the window size directly. Entries must be positive. Pixel `i`
  reads indices `i - s // 2` through `i - s // 2 + s - 1`, as
  `mean_filter(size=s)` does with `origin=0`. An even size therefore reaches
  one index further toward lower indices.

Map shapes:

- `array.shape`: one size per pixel, used on every axis (square or cubic
  windows).
- `(array.ndim, *array.shape)`: one size per pixel and per axis. `size[a]` is
  the map for axis `a`, so windows can be elongated.

With a constant map, the result equals `nanmean_filter` with that fixed size,
up to the rounding described below.

## Boundaries and missing samples

Boundary modes, `cval`, NaN omission, `ignore_inf`, and infinity rules follow
[the neighborhood-statistic semantics](semantics.md). Windows can be wider than
the array. Each boundary repetition and each `cval` sample counts as one
sample. `min_count` must be at least 1. A pixel whose window has fewer than
`min_count` retained samples, including a window smaller than `min_count`,
gives NaN.

Each output pixel uses its own window (a gather operation). The filter does
not spread each input pixel over its own window, so it does not conserve total
flux when the window size changes.

## Accuracy

The filter uses summed-area tables (integral images). Each table entry holds
the sum of all inputs with smaller coordinates on every axis. A window sum is
then a signed combination of `2**ndim` table entries for a window inside the
array, and of a few more entries for a window that crosses a boundary.
Boundary modes map onto the same tables, so the array is not padded.

The tables are float64 for both input dtypes. Each axis pass uses a
compensated prefix sum and rounds the entries to float64. Subtracting large
table entries cancels most of their digits, so the absolute error of a window
sum depends on the prefix sums, not on the window sum.

- Derived first-order bound for a window inside the array: absolute error of
  the window sum at most about `2**-53 * 2**ndim * ndim * sum(abs(array))`.
- Measured example: a 4096×4096 float64 image with level 1e4 and noise 10, size
  3 everywhere. The largest absolute error of the mean against the exact
  `nanmean_filter(method="boxcar")` was 1.0e-5, a relative error of 1.0e-9.
  For the float32 copy of the same image, the results were identical.

**Huge finite values damage distant windows.** A table entry includes every
input with smaller coordinates, so one large finite value enters most of the
table. Its rounding error then appears in windows that do not contain it.
Measured example: a 512×512 float64 image with level 1e3 and unit noise, one
pixel set to 1e20 near the top-left corner, size 3 everywhere. Windows hundreds
of pixels away were wrong by up to 4.6e3, against `nanmean_filter`. The
fixed-size filters keep such a value inside its own windows. Set sentinel
values such as 1e30 or saturated pixels to NaN before you call the filter.

The error does not reach float32 results in the 4096×4096 example, because the float64
error is far below one float32 rounding step. For float64 data that needs
correctly rounded fixed-size means, use `nanmean_filter(method="boxcar")`.
A finite input whose absolute sum overflows float64 raises `ValueError`.

## Cost

- Time: one pass per axis to build the tables, then a fixed amount of work per
  pixel. The time does not depend on the window sizes.
- Memory: one float64 table with `prod(array.shape + 1)` entries. Each
  nonfinite category present in the input (NaN, `+inf`, `-inf`) adds one
  64-bit count table of the same size. The filter also copies the input and
  the window map.

Measured median wall times, complete Python calls, `mode="reflect"`, input
with level 1000 and unit noise:

| Input | Window map | ndfiltrs, 1 thread (ms) | ndfiltrs, 4 threads (ms) | NumPy table (ms) |
| --- | --- | ---: | ---: | ---: |
| float32 2048×2048 | radius 1–33, grows with distance from the center | 77 | 62 | 107 |
| float64 2048×2048 | radius 1–33, grows with distance from the center | 74 | 61 | 106 |
| float64 2048×2048, 1% NaN | same, `nanvariable_mean_filter` | 150 | 108 | 207 |
| float64 2048×2048 | radius 0–15, random | 75 | 61 | 107 |
| float64 128×128×128 | radius 1–9, grows with distance from the center | 77 | 43 | 122 |

The NumPy reference pads the input by the largest radius, builds plain
float64 tables with `np.cumsum` (plus a count table for NaN), and gathers the
window corners. It has no compensation and no infinity handling. NaN in the
input roughly doubles the ndfiltrs time, because every pixel then also reads a
count table. The radius laws and the NaN fraction are arbitrary benchmark
choices. The machine ran other CPU-heavy jobs during these timings (load
average about 8–11 on 14 cores), so treat the values as rough. The thread
scaling in particular is not a clean measurement. Reproduce with `python -m benchmarks.benchmark_variable`.
