# Percentiles, MAD, and sigma clipping

Use percentiles to map the local distribution, MAD to estimate local scatter
with less sensitivity to outliers, and sigma clipping to reject outliers before
calculating a neighborhood mean or median.

```python
import ndfiltrs as nf
import numpy as np

image = np.array([[1.0, 2.0, 100.0], [2.0, np.nan, 3.0], [1.0, 2.0, 4.0]])
lower_quartile = nf.nanpercentile_filter(image, 25, size=3)
scatter = nf.nanmad_filter(image, size=3)
clean_mean = nf.sigma_clipped_filter(image, size=3, spread="mad")
```

All functions use positive odd rectangular windows, the five standard boundary
modes, and aligned native-endian `float32` or `float64` inputs. Results are new
owned arrays with the input shape and dtype. A window size `(1, 5, 5)` filters
each spatial plane separately. Boundary repetitions and finite constant padding
count as individual samples. Set `cval=np.nan` with `mode="constant"` to omit
exterior samples in the NaN-aware or sigma-clipped APIs.

## Percentiles

`percentile_filter(array, percentile, size=3)` sorts each window and linearly
interpolates at rank `(n - 1) * percentile / 100`. Percentiles range from zero
to 100, inclusive. For `[1, 3, 9, 11]`, the 25th percentile is `2.5` and the
median is `6`. This is NumPy's `method="linear"` convention; it is not the
rank-selection convention of SciPy's neighborhood percentile filter.

`nanpercentile_filter` first omits NaNs, optionally omits infinities with
`ignore_inf=True`, then checks `min_count`. The retained sample count determines
the interpolation rank. Plain percentiles propagate any NaN in the window.

Exact endpoint ranks return the selected value, including infinity. Interpolating
a finite value with one infinite endpoint gives that infinity; interpolating
between negative and positive infinity gives NaN. Finite opposite-sign endpoints
are interpolated without first forming an overflowing difference. Signed zeros
use total ordering, with negative zero before positive zero.

## Median absolute deviation

`mad_filter` computes `scale * median(abs(x - median(x)))` for each window.
`scale=1.0` returns raw MAD in the same units as the input. Both medians average
the middle two samples when the retained count is even.

`nanmad_filter` applies omission and `min_count` before computing either median.
Undefined arithmetic, such as deviations from an infinite center, remains NaN;
it is not silently dropped during the second median. A finite center can have
some infinite deviations and still produce a finite MAD. Large deviations and
the final scale are combined without prematurely overflowing when the final
answer is representable.

MAD measures local scatter, including real spatial structure. It is not a
calibrated detector-noise estimate. Gaussian-consistent scaling, when appropriate,
is approximately `scale=1.482602218505602`.

## Sigma-clipped means and medians

`sigma_clipped_filter` always removes NaNs and both infinities before clipping.
Each rejection pass computes a center and spread from the retained samples,
then keeps values inside the inclusive interval
`[center - sigma_lower * spread, center + sigma_upper * spread]`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `sigma`, `sigma_lower`, `sigma_upper` | `3`, `None`, `None` | Nonnegative clipping widths; `None` takes the shared `sigma` |
| `center`, `spread` | `'median'`, `'std'` | Mean or median center; population standard deviation or Gaussian-consistent MAD |
| `maxiters` | `5` | Maximum rejection passes; `None` converges, `0` only omits nonfinite samples |
| `statistic` | `'mean'` | Return the retained mean, median, or mode (formula from `mode_estimator`) |
| `min_count` | `1` | Required retained count after clipping |

For `[1, 2, 100]`, `spread="mad"` rejects `100` and returns `1.5` with the
default clipping width. Zero spread retains only samples equal to the center.
Rejected values never re-enter a window, so convergence requires at most as
many changing passes as there are initial samples. Insufficient final support
returns NaN without an empty-window warning.

For clipping, `'mad'` means raw MAD divided by the standard normal distribution's
75th percentile (`0.6744897501960817`). The public `mad_filter` default remains
raw MAD. This distinction makes clipping widths comparable to standard deviations
for Gaussian data without silently changing the units of a raw MAD map.

## Mode estimates and biweight statistics

```python
# SExtractor-like: clip at median +/- 3 std until stable, then take the mode.
sky = nf.sigma_clipped_filter(image, size=31, maxiters=None, statistic="mode")
mode = nf.nanmode_filter(image, size=15, mode_estimator="mmm")
level = nf.nanbiweight_location_filter(image, size=15)
noise = nf.nanbiweight_scale_filter(image, size=15)
```

| `mode_estimator` | Mode estimate |
| --- | --- |
| `'sex'` (default) | `2.5 * median - 1.5 * mean`, or the median when `abs(mean - median) / std >= 0.3` |
| `'mmm'` | `3 * median - 2 * mean` |
| `'iraf'` | the mean when it is below the median, otherwise `'mmm'` |

`pearson` sets the factor in `pearson * median - (pearson - 1) * mean` (at least
`1`; SExtractor's `BACK_PEARSON`). The default is `2.5` for `'sex'` and `3` for the
others; SExtractor's documentation notes that `3.5` is often more accurate.

For `'sex'`, `std` is the population standard deviation and a zero `std` returns
the mean. This is `backguess` in SExtractor's `src/back.c` (with the default
`BACK_PEARSON = 2.5`); SExtractor also clips at median ± 3 std until the std
stops changing, which is `sigma_clipped_filter(maxiters=None)`. `'mmm'` and
`'iraf'` use the formulas in [astroimred](https://github.com/ysBach/astroimred)'s
`sky_fit`; neither is DAOPHOT's iterative MMM algorithm. For a SExtractor-like
background, clip first: `sigma_clipped_filter(..., maxiters=None, statistic="mode")`.

Biweight statistics follow `astropy.stats.biweight_location` (`c=6`) and
`biweight_scale` (`c=9`), centered on the window median with
`u = (x - median) / (c * MAD)`. Samples with `|u| >= 1` get zero weight. A zero
MAD returns the median (location) or `0` (scale). `modify_sample_size=True`
counts only samples with nonzero weight, as in Astropy.

Plain filters return NaN for windows containing NaN; `nan*` filters omit NaNs
and apply `min_count`. A retained infinity gives NaN, as for variance.

## Execution and limits

Native code copies input before releasing the GIL, uses bounded per-worker
window scratch space, and casts `float64` results to the input dtype once.
Thread count does not change a window's sample order or result. Percentiles and
MAD sort neighborhood samples; this implementation has no running-histogram
optimization. Large windows and repeated clipping can therefore be expensive.
Floating-point thresholds can differ at rounding-sensitive boundaries; exact
real-arithmetic equivalence is not promised.
