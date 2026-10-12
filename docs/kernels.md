# N-dimensional kernels

```python
import numpy as np
import ndfiltrs as nf
from ndfiltrs import kernels

image = np.ones((21, 21), dtype=np.float64)
smoothed = nf.convolve(image, kernels.moffat(alpha=1.5, beta=2.5, ndim=2, radius=6))
```

`ndfiltrs.kernels` builds finite, `float64`, C-contiguous weights using NumPy.
Pass a kernel to `convolve` for a weighted sum or to `nanconvolve` for a
nonnegative weighted mean. The factories are available as the
`ndfiltrs.kernels` submodule.

| Astropy kernel | Factory | Parameter mapping |
| --- | --- | --- |
| `Box2DKernel(width)` | `box(width, ndim=2)` | `width` may be even or odd |
| `Tophat2DKernel(radius)` | `tophat(radius, ndim=2)` | N-ball pixel fractions |
| `Ring2DKernel(radius_in, width)` | `ring(radius_in, radius_in + width, ndim=2)` | Both shell boundaries included |
| `Gaussian2DKernel(x, y, theta)` | `gaussian((y, x), rotation=R)` | `R` follows the array-axis example below |
| `Moffat2DKernel(gamma, alpha)` | `moffat(gamma, alpha, ndim=2, radius=...)` | Core width and exponent |

## Choose normalization and support

By default, `normalize=True` divides weights by their positive finite sum.
Set `normalize=False` to retain sampled profile amplitudes. `nanconvolve`
requires nonnegative weights with at least one positive entry. Use `convolve`
for signed kernels, such as derivatives.

For `tophat` and `ring`, each pixel weight is the fraction of subpixel-center
samples inside the requested region. `oversample=m` uses `m**ndim` samples at
offsets `(j + 0.5) / m - 0.5`. `oversample=1` samples pixel centers. Their
returned extent is the smallest odd support containing every nonzero sampled
weight.

Gaussian, Moffat, and Student-t use pixel-centered odd support. `box` keeps
the requested shape, which may be even.

`moffat` and `student_t` require `radius` because both have heavy tails.
For `gaussian`, default axis radii are `ceil(truncate * sqrt(Sigma_ii))`, with
`truncate=4.0` and covariance `Sigma = R @ diag(sigma**2) @ R.T`.
An explicit scalar or per-axis radius replaces those defaults.

## Set rotation in array-axis order

Rotation columns are principal axes in array-axis order (axis 0 first). For an
Astropy 2-D Gaussian with `x_stddev=1.3`, `y_stddev=0.7`, and `theta=0.37`
radians counterclockwise, use:

```python
import numpy as np

theta = 0.37
rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
weights = kernels.gaussian((0.7, 1.3), rotation=rotation)
```

The rotation check uses `numpy.allclose(R.T @ R, I)` to allow floating-point
round-off. Widths and radii are in pixels.

## Sample another profile

```python
import numpy as np
from astropy.modeling import models

sersic = models.Sersic2D(amplitude=1, r_eff=3, n=2, ellip=0.3, theta=0.5)
weights = kernels.sample(lambda y, x: sersic(x, y), radius=10, ndim=2, oversample=5)
exponential = kernels.sample(lambda y, x: np.exp(-np.hypot(y, x) / 2), radius=(8, 8))
```

`sample(function, radius)` calls `function(*offsets)` with one pixel-offset array
per axis (axis 0 first). It uses the result as weights. The `oversample` and
`normalize` options work as in the factories above.

Any Astropy functional model works this way, including Sersic, Airy, Voigt,
King, Lorentz, and Trapezoid. Astropy 2-D models take `(x, y)` = (axis 1, axis 0).
Use `lambda y, x: model(x, y)` to put the axes in the correct order.

During kernel construction, the callable receives broadcastable coordinate arrays
once per subpixel shift (`oversample**ndim` calls). Filtering does not call it for
each image pixel. Reuse the resulting weights across images.

Identical weights passed to the same filtering function have the same filtering
cost, regardless of how you build them.

A sampled dense kernel does not retain separability.
`gaussian_filter(method="separable")` can do less work than dense convolution
with sampled Gaussian weights.

Sampling approximates pixel integration. When accuracy requires it, increase
`oversample`. Check convergence.


### Recipes

| Profile | Call |
| --- | --- |
| Sersic, Airy, Voigt, King, ... | `sample(lambda y, x: model(x, y), radius=r, ndim=2)` with any Astropy 2-D model |
| Radial Lorentzian `(1 + r² / alpha²)⁻¹` | `moffat(alpha, 1, ndim=n, radius=r)`. Equal to `student_t(alpha, 1, ndim=1, radius=r)` only in 1-D. |
| Exponential disk | `sample(lambda *x: np.exp(-np.sqrt(sum(c * c for c in x)) / h), r, ndim=n)` |
| Ricker ("Mexican hat") | `sample(ricker, r, ndim=n, normalize=False)`. It sums to about zero. |
| Hann window (1-D) | `sample(lambda x: np.cos(np.pi * x / (2 * (r + 1))) ** 2, r, ndim=1)` |

To rotate or stretch any profile, transform the offsets inside the function:

```python
theta = np.deg2rad(30)


def elongated(y, x):
    u = np.cos(theta) * x + np.sin(theta) * y  # along the long axis
    v = -np.sin(theta) * x + np.cos(theta) * y
    return np.exp(-np.hypot(u / 4.0, v / 1.5))


weights = kernels.sample(elongated, radius=(12, 12), oversample=3)
```

