# Convolution and Gaussian filters

Use `convolve` for weighted sums and `nanconvolve` for weighted smoothing with
missing samples. They have different normalization contracts even when the
input contains no NaNs.

```python
import ndfiltrs as nf
import numpy as np

line = np.array([1.0, 2.0, 3.0])
weights = np.ones(3)
weighted_sum = nf.convolve(line, weights)  # Center: 6.0
weighted_mean = nf.nanconvolve(line, weights)  # Center: 2.0

damaged = np.array([1.0, np.nan, 3.0])
smoothed = nf.nanconvolve(damaged, weights)  # Center: 2.0
```

All functions accept aligned, native-endian `float32` or `float64` input and
return a new C-contiguous array with the input shape and dtype. Arrays have
rank one or higher. The functions reject masked arrays. They accept
non-C-contiguous input through normalization.

Filters change every output position. Use
`fixpix` when only masked pixels should change.

## Choose the operation

| Function | Result | Weight restrictions |
| --- | --- | --- |
| `convolve`, `correlate` | Raw weighted sum | Finite real weights. Negative and zero values allowed |
| `nanconvolve`, `nancorrelate` | Weighted mean after omission | Finite nonnegative weights with at least one positive entry |
| `convolve_variance`, `nanconvolve_variance` | Either result above and its propagated variance | As `convolve` or `nanconvolve` |
| `matched_filter` | Template amplitude, its variance, and signal-to-noise ratio | Finite real weights with a nonzero entry |
| `gaussian_filter` | Gaussian smoothing or a Gaussian derivative | Generated from `sigma`, `order`, and support |
| `nangaussian_filter` | Gaussian weighted mean after omission | Smoothing only. No derivative order |
| `gaussian_laplace`, `gaussian_gradient_magnitude` | Gaussian Laplacian or gradient magnitude | Derivatives along selected axes |

## Kernel orientation and support

```python
nf.convolve(array, weights, mode="reflect", cval=0.0, origin=0)
nf.correlate(array, weights, mode="reflect", cval=0.0, origin=0)
nf.nanconvolve(
    array,
    weights,
    mode="reflect",
    cval=0.0,
    origin=0,
    ignore_inf=False,
    min_count=1,
)
```

`nancorrelate` accepts the same arguments as `nanconvolve`. Kernels must have
the same rank as the input, with a positive length along each axis. Odd and
even lengths are accepted. A shape such as `(1, 3, 3)` filters the final two
axes of a cube without mixing the first axis. These functions have no `axes`
argument.

For each kernel axis of length $m$, the anchor is
$a=\lfloor m/2\rfloor+\mathrm{origin}$. The anchor must lie between zero and
$m-1$. Scalar `origin` broadcasts. A tuple supplies one value per input axis.
For a one-dimensional kernel, the operations are

$$
\operatorname{convolve}(x,w)_i = \sum_j w_j x_{i+a-j},
\qquad
\operatorname{correlate}(x,w)_i = \sum_j w_j x_{i+j-a}.
$$

Boundary mapping applies to the sampled coordinates. For `origin=0`, an
even kernel of length four has anchor two.

Asymmetric kernels distinguish the operations. Consider `line = [1., 2., 3.]`
and `weights = [1., 0., 0.]`. The center result is `3.0` for convolution
and `1.0` for correlation.

`mode` accepts `reflect`, `constant`, `nearest`, `mirror`, and `wrap`, either
as a scalar or one entry per input axis. Their coordinate mappings match the
[neighborhood-statistic boundary modes](semantics.md#boundary-modes).
`cval` supplies every out-of-bounds sample selected by a constant boundary.

## Missing samples and normalization

Plain convolution and correlation propagate NaNs only through nonzero
kernel coefficients. A zero coefficient excludes its sample completely,
including NaNs and infinities. An all-zero raw kernel returns positive zero.
Signed weights change infinity signs. Contributions containing both positive
and negative infinity produce NaN.

NaN-aware convolution and correlation always omit NaNs. `ignore_inf=True`
also omits infinities. Otherwise, retained infinities participate in the
weighted result. These functions reject negative weights and all-zero kernels.

`min_count` must lie between one and the number of positive kernel entries.
Each output divides its retained weighted sum by its retained weight sum.
No retained weight, or fewer than `min_count` retained nonzero-weight
positions, produces NaN. There is no epsilon threshold for accepting a
denominator.

Boundary repetitions count separately toward `min_count`. Finite constant
padding contributes to both numerator and denominator. Set `mode="constant"`
and `cval=np.nan` to omit exterior positions instead of averaging in padding.
Finite hot/dead pixels remain valid unless the caller marks them with NaNs.

For example, weights `[1., 2., 1.]` around data `[1., NaN, 5.]` produce
`(1 + 5) / (1 + 1) = 3.0`. With `min_count=3`, the result is NaN instead.

Signed derivative kernels belong in `convolve` or `correlate`, because normalizing
their retained weights would change the derivative definition. For derivatives
of data with missing samples, use `nansavgol_filter`, which refits a local
polynomial to the retained samples (see [Savitzky-Golay filters](savgol.md)).

## Variance propagation and matched filtering

```python
value, variance = nf.convolve_variance(image, var, weights)
smoothed, smoothed_var = nf.nanconvolve_variance(image, var, psf)
snr, flux, flux_var = nf.matched_filter(image - background, var, psf)
```

The second argument holds one variance per pixel. Pass an inverse-variance
weight map with `noise_type="inverse_variance"`. It must have the shape of
the data and is cast to the data dtype. Each function computes all of its
outputs in one pass over the kernel taps.

### Assumptions

- Noise is independent from pixel to pixel. Covariance between input pixels,
  for example after resampling, is not modeled. The propagated variance then
  underestimates the true variance.
- The set of retained samples (the NaN mask) does not depend on the noise
  values.
- Weights are exact constants.

### Raw weighted sums

For one output, tap $j$ reads source pixel $p(j)$. Group the taps by pixel:

$$
y = \sum_j w_j x_{p(j)} = \sum_p c_p x_p,
\qquad
c_p = \sum_{j: p(j)=p} w_j .
$$

For independent $x_p$ with variance $v_p$,
$\operatorname{Var}(y)=\sum_p c_p^2 v_p$. Away from the edges each pixel is
read once, so this is $\sum_j w_j^2 v_{p(j)}$. The `reflect`, `nearest`,
`mirror`, and `wrap` modes can read one pixel through several taps near an
edge. Those taps are perfectly correlated, so their weights add before
squaring. For example, with `reflect`, weights `[3., 2., 1.]`, and variances
`[1., 10., 100.]`, the first output reads pixel 0 with weights 1 and 2. Its
variance is $(1+2)^2\cdot 1 + 3^2\cdot 10 = 99$, not $1+4+90$. A
constant-mode exterior sample is an exact constant, so it has zero variance.

`convolve_variance` returns `convolve` (or `correlate`, with `correlate=True`)
bitwise as its first output.

### Normalized smoothing

`nanconvolve_variance` divides by the retained weight sum $W=\sum_j w_j$. For
a fixed retained set, $W$ is a constant, so

$$
y = \frac{1}{W}\sum_p c_p x_p,
\qquad
\operatorname{Var}(y) = \frac{1}{W^2}\sum_p c_p^2 v_p .
$$

For example, weights `[1., 2., 1.]` around data `[1., NaN, 5.]` with
variances `[2., 7., 6.]` give $y=3$ and $\operatorname{Var}(y)=(2+6)/2^2=2$.
Where the noise omits no sample, the first output equals `nanconvolve` (or
`nancorrelate`) bitwise.

### Matched filter

Model the data around each position as $x_j = A w_j + n_j$, with template
$w$ and independent zero-mean noise $n_j$ of variance $v_j$. `matched_filter`
correlates the template, so it keeps the orientation of a PSF image. With
coefficients $a_j$ over the retained samples:

$$
N = \sum_j a_j x_j,
\qquad
\hat A = \frac{N}{S},\quad S = \sum_j w_j a_j,
\qquad
\operatorname{Var}(\hat A) = \frac{\operatorname{Var}(N)}{S^2},
\qquad
\mathrm{snr} = \frac{N}{\sqrt{\operatorname{Var}(N)}} .
$$

$\operatorname{Var}(N) = \sum_p \bigl(\sum_{j: p(j)=p} a_j\bigr)^2 v_p$ uses
the same grouping as above. $E[N] = A S$, so $\hat A$ is unbiased.

- `weighting="inverse_variance"` (default): $a_j = w_j / v_j$. This
  minimizes $\chi^2=\sum_j (x_j - A w_j)^2 / v_j$. $\hat A$ is then the
  minimum-variance linear unbiased estimate, with
  $\operatorname{Var}(\hat A) = 1/S = 1/\sum_j (w_j^2/v_j)$ away from repeated
  boundary pixels.
- `weighting="kernel"`: $a_j = w_j$, so $\hat A = \sum_j w_j x_j / \sum_j w_j^2$.
  Its `snr` equals $y/\sqrt{\operatorname{Var}(y)}$ for either function above:
  the normalization cancels.

With uniform variance both weightings give the same `snr` and amplitude,
except with `mode="constant"` and a finite `cval`. There, kernel weighting
keeps the exact padding samples, and inverse-variance weighting omits them.
Under pure noise, `snr` has unit variance, so a threshold on `snr` is in
units of the standard deviation of the filtered image. If the template sums
to one, $\hat A$ is the total flux of a point source with that PSF.
The template may have signed weights.

### Invalid and special values

| Case | `convolve_variance` | `nanconvolve_variance` | `matched_filter` |
| --- | --- | --- | --- |
| NaN data | All outputs NaN | Sample omitted | Sample omitted |
| NaN or negative noise | Variance NaN, value unchanged | Sample omitted | Sample omitted |
| Infinite variance (inverse variance `0`) | Variance infinite | Sample omitted | Sample omitted |
| Zero variance | Exact sample | Exact sample | Exact (`kernel`); omitted (`inverse_variance`) |
| Constant-mode exterior | Zero variance | Zero variance; omitted if `cval` is NaN | As zero variance; omitted if `cval` is NaN |
| Infinite data | Value infinite, finite variance | Same, unless `ignore_inf=True` | Same, unless `ignore_inf=True` |

An omitted sample leaves both the data sum and the variance sum, and it does
not count toward `min_count`. Under inverse-variance weighting, a
zero-variance sample would have infinite weight, so it is omitted. A NaN
estimate has NaN variance and `snr`. When the variance is zero, `snr` is
infinite (or NaN for a zero numerator).

### Arithmetic

Sums use compensated `float64` accumulation. Products or sums that overflow
or underflow are recomputed with binary scaling, so the variance of finite
inputs is finite whenever it fits in `float64`. Results are cast to the data
dtype once. Exact rounding is not promised. Interior outputs stream their
sums. Edge outputs store their taps and sort them by source pixel to group
repeats.

### Comparison with SExtractor detection filtering

This comparison uses the SExtractor source at commit
[`90296de`](https://github.com/astromatic/sextractor/tree/90296de1a23607f2501864d4d6c1f757755f0ee7)
and its [weighting documentation](https://sextractor.readthedocs.io/en/latest/Weighting.html).

What matches:

- SExtractor filters the detection image with the plain mask, without
  dividing by the variance. That corresponds to `weighting="kernel"`.
- Its `convolve()` in `src/filter.c` adds `mask[dy][dx]` times the pixel at
  `(y + dy - h/2, x + dx - w/2)` without flipping the mask. That is
  correlation, as in `matched_filter`.
- With a weight map, SExtractor also filters the variance map and
  thresholds each pixel against the filtered variance (`src/scan.c`).
  Weight maps become variances internally (`weight_to_var()` in
  `src/weight.c`).

What differs:

- SExtractor filters the variance map with the same mask $w$, not $w^2$. Its
  threshold is `DETECT_THRESH * sqrt(sum_j w_j v_j)`. The variance of the
  filtered image is $\sum_j w_j^2 v_j$. The SExtractor documentation states
  that `DETECT_THRESH` is in units of the background standard deviation, and
  `varnorm` (computed in `getconv()`) is not used anywhere else in `src/`. For
  uniform $v$ and a mask with unit sum, a SExtractor threshold $t$ equals an
  `snr` threshold $t/\sqrt{\sum_j w_j^2}$. With varying $v$, the two
  thresholds are not proportional.
- SExtractor has no inverse-variance weighting in detection filtering.
- With the default normalization, SExtractor divides the mask by
  $\sum_j |w_j|$ (by $\sqrt{\sum_j w_j^2}$ for zero-sum masks). `snr` does
  not depend on the template scale. The amplitude scales inversely with it.
- At image edges, SExtractor skips out-of-image taps without renormalizing.
  This equals `mode="constant", cval=0.0` with `weighting="kernel"`. It has
  no NaN handling. A zero weight becomes a variance of `BIG` (`1e30`), which
  raises the threshold of every neighbor that the filter reaches.
  ndfiltrs omits such samples instead.
- SExtractor accumulates in single precision.

## Gaussian smoothing and derivatives

```python
blur = nf.gaussian_filter(image, sigma=1.5, radius=5)
finite_blur = nf.nangaussian_filter(
    image, sigma=1.5, radius=5, min_count=2, method="separable"
)
dx = nf.gaussian_filter(image, sigma=1.5, order=(0, 1), radius=5)
laplace = nf.gaussian_laplace(image, sigma=1.5, radius=5)
gradient = nf.gaussian_gradient_magnitude(image, sigma=1.5, radius=5)
```

### Select axes and support

`sigma` is measured in pixels. Scalar values broadcast across selected axes.
Tuples provide one value per selected axis. `axes=None` selects every input
axis.

To smooth spatial axes of a cube, use `axes=(-2, -1)`. Unselected axes
have kernel length one. Axis-specific `sigma`, `order`, `radius`, and `mode`
entries follow the order of `axes`.

`radius` explicitly selects an odd support length of `2 * radius + 1` along
each selected axis. When `radius=None`, `truncate` controls the support in
units of `sigma`. Its configurable default is `4.0`. Explicit `radius`
controls support when provided. Larger support increases both kernel memory
and per-output work.

### Select the derivative order

`gaussian_filter` accepts nonnegative integer `order`, scalar or per selected
axis:

- Order zero smooths.
- Order one differentiates once.
- Order two differentiates twice.
- A tuple such as `(1, 1)` requests a mixed derivative.
- Higher orders use the corresponding analytic Gaussian derivative.

A zero `sigma` gives identity behavior only at order zero. Derivatives require
positive `sigma`. Generated coefficients must be representable as finite
`float64` values.

### Interpret normalization and derivative values

Order-zero kernels normalize sampled Gaussian weights to unit sum. Plain
Gaussian smoothing also divides by the computed weight sum during evaluation.
Thus, finite constant neighborhoods remain finite even near the dtype limit.
NaNs still propagate through nonzero coefficients. Only `nangaussian_filter`
omits them.

Derivative kernels sample analytic Gaussian derivatives with the same
order-zero normalization. Finite support and sampling mean their sums need
not be exactly zero. A constant input can therefore have a nonzero response
to an even-order derivative. No correction silently forces a zero sum.

Derivatives are in pixel coordinates.
Physical-coordinate derivatives require the corresponding pixel-scale factors.

`gaussian_laplace` sums second-derivative product kernels across the selected
axes before convolution. `gaussian_gradient_magnitude` combines first-derivative
component maps with a stable Euclidean norm. `nangaussian_filter` supports
`ignore_inf` and `min_count`, follows the normalized omission rules above,
and does not accept derivative orders.

## Separable smoothing

Set `method="separable"` in `gaussian_filter` or `nangaussian_filter` to smooth
one selected axis at a time. Kernel storage grows with the sum of axis lengths
instead of their product. The default `method="direct"` preserves dense-kernel
behavior. Derivative calculations require that method.

Separable smoothing carries each partial neighborhood's retained weight and
integer sample count into the next pass. Missing rows therefore receive their
proper weights. The filter applies `min_count` after all passes. Infinities, NaNs,
zero coefficients, and constant padding follow the same omission rules as
above.

Finite weighted means remain within their contributing range. This avoids
overflow for constant data near the dtype limit. The filter copies input once.
Native passes release the GIL. The filter casts the final result to the input dtype once.

Axis grouping changes floating-point rounding. Separable mode rejects positive
product weights below the smallest normal `float64` value.
This preserves which kernel positions count as support. Use `method="direct"`
for that case. This restriction follows the arithmetic range, not an array-size cutoff.

## Arithmetic and execution

Direct Gaussian operations construct dense N-D product kernels and use the
same native weighted backend as arbitrary convolution. Boundary handling applies
to that full kernel, including nonzero constant padding in derivative
operations. Separable smoothing is opt-in. FFT execution is not implemented.

Native evaluation uses compensated `float64` accumulation with a scaled
arithmetic fallback for finite overflow. Infinities supplied by the data
are handled separately from finite arithmetic overflow. Results cast to the
input dtype at the end. Gradient components remain `float64` until their
norm is computed.

Exact rounding is not promised, and scaling can underflow
very small contributions. Thread count does not change traversal order.

For $N$ output positions, kernel volume $V$, and array rank $d$, one direct
convolution takes $O(N V d)$ work. Dense Gaussian kernel storage grows as
the product of selected-axis support lengths. Each gradient component needs
another convolution. Kernel construction also grows with derivative order
and per-axis support lengths. Input snapshots and output arrays also consume
memory.

These methods make no speed claim against external specialized
implementations.

Separable smoothing takes $O(N \sum_a L_a)$ work for selected-axis support
lengths $L_a$, plus $O(N)$ working storage for means, retained masses, counts,
and invalid-value flags. It avoids allocating a dense product kernel.
