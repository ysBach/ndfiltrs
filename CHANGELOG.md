# Changelog

## 0.1.0 - Unreleased

First release of `ndfiltrs`: N-dimensional neighborhood filters for
NumPy, implemented in Rust.

### Performance

- Median filters offer exact per-window selection (`method="direct"`) and
  vectorized selection networks (`method="sortnet"`). Direct selection is the
  default. The faster choice depends on the input and window.
- Mean filters offer compensated direct sums, axis-by-axis filtering, and exact
  running sums (`method="boxcar"`). Boxcar rounds once to the output dtype and
  may differ in the last bits from the other methods. For example,
  `nanmean_filter(x, 63, method="boxcar")` skips NaNs without contaminating
  later windows.
- Weighted filters share compensated accumulation. Rectangular min/max filters
  use sliding extrema.
- [Benchmarks](docs/benchmarks.md) compare shipped methods with SciPy and
  Astro-SCRAPPY, including slower cases and differences in border handling.

### Common behavior

- Every filter accepts N-D `float32` and `float64` arrays and the
  SciPy-style boundary modes `reflect`, `constant`, `nearest`, `mirror`,
  and `wrap`.
- Window filters take a rectangular `size` or any `footprint` (disk, ring,
  cross, ...), shifted by `origin`, with SciPy's window geometry.
- `nan*` variants skip NaN samples and return NaN where fewer than
  `min_count` samples remain.
- Work runs in parallel Rust threads with the GIL released. Median, min/max,
  mean, var, and std results are bitwise identical for any thread count.

### Neighborhood statistics

- Rank: `median_filter`, `percentile_filter`, `min_filter`, `max_filter`,
  and their `nan*` variants, plus `rank_filter`. Example: the 25th percentile
  of `[1, 3, 9, 11]` is `2.5`.
  Rectangular min/max use sliding extrema rather than visiting every window element.
- Moments: `mean_filter`, `var_filter`, `std_filter` (with `ddof`), and
  their `nan*` variants. `mean_filter(method="separable")` averages one axis at a
  time for large boxcar windows.
- Robust: `mad_filter` and `nanmad_filter` (scaled median absolute
  deviation), and `sigma_clipped_filter` (iterative outlier rejection).
  Example: with `spread="mad"`, clipping `[1, 2, 100]` gives mean `1.5`.
- Mode: `mode_filter` and `nanmode_filter` use SExtractor, MMM, or IRAF rules
  with an adjustable `pearson` factor.
  Use `sigma_clipped_filter(maxiters=None, statistic="mode")` for a SExtractor-like background.
- Biweight location and scale filters match `astropy.stats`.

### Kernels, convolution, and morphology

- Weighted: `convolve` and `correlate` with any N-D kernel and `origin`
  shift. `nanconvolve` and `nancorrelate` divide by the weights of the
  samples that remain.
- Gaussian: `gaussian_filter` (smoothing and derivatives, optional
  `method="separable"`), `nangaussian_filter`, `gaussian_laplace`, and
  `gaussian_gradient_magnitude`. Use `axes=(-2, -1)` to filter each plane
  of a cube separately.
- Kernels: `ndfiltrs.kernels` builds N-D `box`, `tophat`, `ring`, `gaussian`,
  `moffat`, and `student_t` weights, with optional rotation and subpixel
  oversampling. `kernels.sample` samples any function or Astropy model.
  Astropy kernel objects also work directly as weights.
- Edge filters `sobel`, `prewitt`, and `laplace` match SciPy.
- Grey dilation, opening, closing, white and black top-hats, and morphological
  gradient and Laplace also match SciPy. Erosion is `min_filter`.

### Pixel repair

- `fixpix` fills masked pixels in N-D images. The default linear method
  matches [astroimred's `fixpix`](https://github.com/ysBach/astroimred/blob/fb70cae872406f907175e3b47d497ffe14b8c238/src/astroimred/imutil/pixels.py).
- `method="median"` uses the median of unmasked neighbors with any `footprint`.
- `method="mean"` uses their mean, optionally weighted by a `kernel`, like
  Astropy's `interpolate_replace_nans`. Passing `kernel` alone selects this method.
- Median and mean repair evaluate only masked pixels, in parallel.

### Packaging

- Reproducible, parity-checked SciPy/NumPy benchmarks report raw timings and
  operation-specific speedups or slowdowns.
- Independent numerical references cover boundary geometry, nonfinite samples,
  precision, and input preservation. CI also compares sampled kernels with Astropy.
- MIT license. Bundled license notices cover the Rust dependencies and
  `astroimred`.
- `cp310-abi3` wheels support Python 3.10+ on Linux (x86-64, ARM64),
  macOS (x86-64, ARM64), and Windows (x86-64). Linux requires glibc 2.28+.
  All platforms require NumPy 1.24+.
