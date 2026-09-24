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
rank one or higher; masked arrays are rejected. Non-C-contiguous input is
accepted through normalization. Filters change every output position; use
`fixpix` when only masked pixels should change.

## Choose the operation

| Function | Result | Weight restrictions |
| --- | --- | --- |
| `convolve`, `correlate` | Raw weighted sum | Finite real weights; negative and zero values allowed |
| `nanconvolve`, `nancorrelate` | Weighted mean after omission | Finite nonnegative weights with at least one positive entry |
| `gaussian_filter` | Gaussian smoothing or a Gaussian derivative | Generated from `sigma`, `order`, and support |
| `nangaussian_filter` | Gaussian weighted mean after omission | Smoothing only; no derivative order |
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
axes of a cube without mixing the first axis; these functions have no `axes`
argument.

For each kernel axis of length $m$, the anchor is
$a=\lfloor m/2\rfloor+\mathrm{origin}$. The anchor must lie between zero and
$m-1$. Scalar `origin` broadcasts; a tuple supplies one value per input axis.
For a one-dimensional kernel, the operations are

$$
\operatorname{convolve}(x,w)_i = \sum_j w_j x_{i+a-j},
\qquad
\operatorname{correlate}(x,w)_i = \sum_j w_j x_{i+j-a}.
$$

Boundary mapping applies to the sampled coordinates. For `origin=0`, an
even kernel of length four has anchor two. Asymmetric kernels distinguish
the operations: with `line = [1., 2., 3.]` and `weights = [1., 0., 0.]`,
the center result is `3.0` for convolution and `1.0` for correlation.

`mode` accepts `reflect`, `constant`, `nearest`, `mirror`, and `wrap`, either
as a scalar or one entry per input axis. Their coordinate mappings match the
[neighborhood-statistic boundary modes](semantics.md#boundary-modes).
`cval` supplies every out-of-bounds sample selected by a constant boundary.

## Missing samples and normalization

Plain convolution and correlation propagate NaNs only through nonzero
kernel coefficients. A zero coefficient excludes its sample completely,
including NaNs and infinities. An all-zero raw kernel returns positive zero.
Signed weights change infinity signs; contributions containing both positive
and negative infinity produce NaN.

NaN-aware convolution and correlation always omit NaNs. `ignore_inf=True`
also omits infinities; otherwise retained infinities participate in the
weighted result. Negative weights and all-zero kernels are rejected.
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
`(1 + 5) / (1 + 1) = 3.0`; `min_count=3` instead produces NaN. Signed
derivative kernels belong in `convolve` or `correlate`, because normalizing
their retained weights would change the derivative definition.

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

`sigma` is measured in pixels. Scalar values broadcast across selected axes;
tuples provide one value per selected axis. `axes=None` selects every input
axis. To smooth spatial axes of a cube, use `axes=(-2, -1)`; unselected axes
have kernel length one. Axis-specific `sigma`, `order`, `radius`, and `mode`
entries follow the order of `axes`.

`radius` explicitly selects an odd support length of `2 * radius + 1` along
each selected axis. When `radius=None`, `truncate` controls the support in
units of `sigma`; its configurable default is `4.0`. Explicit `radius`
controls support when provided. Larger support increases both kernel memory
and per-output work.

`gaussian_filter` accepts nonnegative integer `order`, scalar or per selected
axis. Order zero smooths; order one differentiates once; order two
differentiates twice. A tuple such as `(1, 1)` requests a mixed derivative.
Higher orders use the corresponding analytic Gaussian derivative. A zero
`sigma` gives identity behavior only at order zero; derivatives require
positive `sigma`. Generated coefficients must be representable as finite
`float64` values.

Order-zero kernels normalize sampled Gaussian weights to unit sum. Plain
Gaussian smoothing also divides by the computed weight sum during evaluation,
so finite constant neighborhoods remain finite even near the dtype limit.
NaNs still propagate through nonzero coefficients; only `nangaussian_filter`
omits them.

Derivative kernels sample analytic Gaussian derivatives with the same
order-zero normalization. Finite support and sampling mean their sums need
not be exactly zero. A constant input can therefore have a nonzero response
to an even-order derivative. No
correction silently forces a zero sum. Derivatives are in pixel coordinates;
physical-coordinate derivatives require the corresponding pixel-scale factors.

`gaussian_laplace` sums second-derivative product kernels across the selected
axes before convolution. `gaussian_gradient_magnitude` combines first-derivative
component maps with a stable Euclidean norm. `nangaussian_filter` supports
`ignore_inf` and `min_count`, follows the normalized omission rules above,
and does not accept derivative orders.

## Separable smoothing

Set `method="separable"` in `gaussian_filter` or `nangaussian_filter` to smooth
one selected axis at a time. Kernel storage grows with the sum of axis lengths
instead of their product. The default `method="direct"` preserves dense-kernel
behavior; derivative calculations require that method.

Separable smoothing carries each partial neighborhood's retained weight and
integer sample count into the next pass. Missing rows therefore receive their
proper weights. `min_count` is applied after all passes; infinities, NaNs,
zero coefficients, and constant padding follow the same omission rules as
above. Finite weighted means remain within their contributing range, avoiding
overflow for constant data near the dtype limit. Input is copied once, native
passes release the GIL, and the final result is cast to the input dtype once.

Axis grouping changes floating-point rounding. To avoid changing which kernel
positions count as support, separable mode rejects positive product weights
below the smallest normal `float64` value; use `method="direct"` for that
case. This restriction follows the arithmetic range, not an array-size cutoff.

## Arithmetic and execution

Direct Gaussian operations construct dense N-D product kernels and use the
same native weighted backend as arbitrary convolution. Boundary handling applies
to that full kernel, including nonzero constant padding in derivative
operations. Separable smoothing is opt-in; FFT execution is not implemented.

Native evaluation uses compensated `float64` accumulation with a scaled
arithmetic fallback for finite overflow. Infinities supplied by the data
are handled separately from finite arithmetic overflow. Results cast to the
input dtype at the end; gradient components remain `float64` until their
norm is computed. Exact rounding is not promised, and scaling can underflow
very small contributions. Thread count does not change traversal order.

For $N$ output positions, kernel volume $V$, and array rank $d$, one direct
convolution takes $O(N V d)$ work. Dense Gaussian kernel storage grows as
the product of selected-axis support lengths; each gradient component needs
another convolution. Kernel construction also grows with derivative order
and per-axis support lengths. Input snapshots and output arrays also consume
memory. These methods make no speed claim against external specialized
implementations.

Separable smoothing takes $O(N \sum_a L_a)$ work for selected-axis support
lengths $L_a$, plus $O(N)$ working storage for means, retained masses, counts,
and invalid-value flags. It avoids allocating a dense product kernel.
