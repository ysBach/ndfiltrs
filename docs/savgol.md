# Savitzky-Golay filters

A Savitzky-Golay filter fits a low-degree polynomial to the samples in each
window by least squares. The output is the value of that polynomial at the
output position, or one of its derivatives. `savgol_filter` propagates NaNs.
`nansavgol_filter` fits only the samples that remain after NaNs, optional
infinities, and an optional mask are left out.

```python
import ndfiltrs as nf
import numpy as np

line = np.arange(9.0) ** 2  # 0, 1, 4, 9, ...
line[4] = np.nan
nf.savgol_filter(line, 5, 2)  # all NaN: every window here contains index 4
nf.nansavgol_filter(line, 5, 2)  # 0, 1, 4, ..., 64: the quadratic is exact
nf.nansavgol_filter(line, 5, 2, deriv=1)  # 0, 2, 4, ..., 16: the slope

image = np.random.default_rng(0).normal(size=(64, 64))
image[10, 20] = np.nan
smooth = nf.nansavgol_filter(image, 7, 2)  # quadratic surface per 7x7 window
dx = nf.nansavgol_filter(image, 7, 2, deriv=(0, 1), delta=0.5)  # d/dx
```

The input contract matches the other filters: aligned, native-endian
`float32` or `float64` arrays of rank one or higher. The result is a new
C-contiguous array with the input shape and dtype.

## What the fit does

For a window with half-widths $h_a$, the samples have integer offsets
$d = (d_0, d_1, \ldots)$ with $|d_a| \le h_a$. The filter finds the
polynomial $P$ that minimizes

$$
\sum_{d\ \text{retained}} w_d \bigl(x_d - P(d)\bigr)^2 ,
$$

where $x_d$ is the sample and $w_d$ is its weight (one by default). The
output is $P(0)$, or the derivative
$\partial^{k_0+k_1+\cdots} P / \partial x_0^{k_0} \partial x_1^{k_1} \cdots$
at the output position, divided by $\prod_a \mathrm{delta}_a^{k_a}$.

- `window_length`: odd window length per selected axis. A length of one
  leaves that axis unfiltered.
- `polyorder`: polynomial degree. Each degree must be less than its window
  length.
- `deriv`: derivative order per selected axis. A scalar broadcasts, so
  `deriv=1` on a 2-D image is the mixed derivative $\partial^2/\partial y\,\partial x$.
  Use `deriv=(0, 1)` for $\partial/\partial x$.
- `delta`: sample spacing per selected axis, in your physical unit.
- `axes`: axes to filter. The default is every axis. Tuple parameters follow
  the order of `axes`.

### Two polynomial bases

`basis="total"` (the default) uses all monomials $\prod_a d_a^{k_a}$ with
$\sum_a k_a \le p$. For degree 2 on an image, these are
$1, x, y, x^2, xy, y^2$: six terms.

`basis="per_axis"` limits each exponent separately: $k_a \le p_a$. For
degree 2 on each axis of an image, there are nine terms, including
$x^2 y^2$. On a complete rectangular window, this fit equals the
one-dimensional filter applied along one axis, then the next.

A total-degree fit needs fewer samples per window.

## Missing samples

A window with no left-out samples uses fixed coefficients. They are computed
once from the full window, so complete data costs one weighted sum per
output, as in a convolution.

A window with left-out samples solves its own least-squares problem with the
retained samples only. The solver works in `float64`: a Householder QR
factorization of the weighted design matrix, then a singular value
decomposition of the triangular factor. The output is the value or
derivative of the polynomial fitted to the retained samples.

A window gives NaN when:

- fewer than `min_count` positive-weight samples remain. The default is the
  number of polynomial terms, the fewest samples that can determine the fit.
- the retained samples do not determine the polynomial. For example, all
  retained samples of a 2-D window lie on one row, so the slope across rows
  is unknown.

The second test is numerical. With $n$ retained samples and $r$ terms, the
fit is undetermined when the singular values of the weighted design matrix
satisfy $\sigma_{\min} \le \sigma_{\max} \max(n, r)\,\varepsilon$, where
$\varepsilon$ is the `float64` machine epsilon. This is the default
tolerance of `numpy.linalg.matrix_rank`: a perturbation of relative size
$\varepsilon$ in the matrix can make it singular. The design matrix uses
offsets divided by $h_a$, so its entries lie in $[-1, 1]$. There is no other
conditioning cutoff. A window with an unusual set of retained samples can
give a large but finite result. Increase `min_count` to reject sparse
windows.

The filter never lowers the degree to make a fit possible. A lower degree
would change the estimator from pixel to pixel.

`ignore_inf=True` also leaves out infinite samples. Otherwise an infinite
sample stays in the fit. The result is then an infinity with the sign of
sample times coefficient. Infinities of both signs, or an infinite sample
with a zero coefficient, give NaN.
`mask=` leaves out samples marked `True`, in addition to NaNs.

### Why not renormalized derivative kernels

`nanconvolve` handles missing samples by dividing the weighted sum by the
sum of the retained weights. That is correct for a smoothing kernel with
positive weights, because the result stays a weighted mean. A derivative
kernel has signed weights that sum to zero. The sum of its retained weights
can be zero or have either sign, so dividing by it no longer computes a
derivative. For this reason the
Sobel, Prewitt, and Gaussian derivatives propagate NaNs (see
[Convolution and Gaussian filters](convolution.md)).

A masked least-squares fit has no such problem. It refits the polynomial to
the retained samples, then differentiates that polynomial. Where the data
are a polynomial of the fitted degree, the result is exact with or without
missing samples.

## Edges

`mode` sets the edge treatment per selected axis.

- `"interp"` (the default, as in SciPy): the window stays inside the array.
  Near an edge, the window moves inward, and the fit is evaluated at the
  output position instead of the window center. The window length must not
  exceed the axis length.
- `"reflect"`, `"constant"`, `"nearest"`, `"mirror"`, `"wrap"`: the array is
  extended as in the [neighborhood-statistic boundary modes](semantics.md#boundary-modes),
  and the centered window is fitted. The padded samples take part in the
  fit. With `nansavgol_filter`, `mode="constant"` and `cval=np.nan` leave
  the exterior out instead.

## Weights

`weights` is an optional nonnegative array with the window shape over the
selected axes. The fit then minimizes the weighted sum of squared residuals.
A zero weight removes that window position from every fit, so a circular
footprint is a 0/1 weight array. Scaling all weights by one constant does not
change the result. Weights apply to window positions, not to pixels: they
are the same for every output.

## Compatibility with SciPy

On complete 1-D data, `savgol_filter(x, window_length, polyorder, deriv,
delta, mode=mode, cval=cval)` equals `scipy.signal.savgol_filter` with the
same arguments, up to rounding. The differences are:

- SciPy filters along `axis=-1` only. Here the default filters every axis.
  Pass `axes=(-1,)` for the SciPy behavior on N-D input.
- SciPy accepts even window lengths. Here window lengths must be odd, so the
  fit is centered on a sample.
- SciPy returns zeros when `deriv > polyorder`. Here that is an error.
- SciPy rejects `mode="reflect"`. Here it is allowed.

## Arithmetic and cost

Coefficients, solves, and sums use `float64`. Sums are compensated, and the
result is cast to the input dtype once. Work runs in parallel Rust threads
with the GIL released.

For $N$ outputs and $V$ positive-weight window positions, complete windows
cost $O(V)$ each. Interp windows near an edge cost $O(V r)$ each for $r$
terms. A window with left-out samples costs $O(n r^2)$ for $n$ retained
samples. The filter evaluates every window directly; it has no separable
or FFT path.
