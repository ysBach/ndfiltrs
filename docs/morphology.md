# Edges and morphology

```python
import ndfiltrs as nf

edges = nf.sobel(image, axis=-1)  # d/d(axis -1), smoothed along other axes
curvature = nf.laplace(image)
background_removed = nf.white_tophat(image, footprint=nf.kernels.tophat(5, ndim=2))
```

## Derivative operators

`sobel`, `prewitt`, and `laplace` match `scipy.ndimage` and work in N-D. Sobel and
Prewitt use `[-1, 0, 1]` along `axis` and `[1, 2, 1]` or `[1, 1, 1]` along every
other axis; the result is positive where values increase with index. `laplace`
sums `[1, -2, 1]` over all axes. NaNs propagate. For NaN-aware or wider
derivatives, use `gaussian_filter(order=...)` or pass a kernel to `convolve`.

## Grey morphology

All operations use flat structuring elements given by `size` or `footprint` and
match SciPy, including `origin` and boundary modes.

| Operation | Definition |
| --- | --- |
| erosion | `min_filter` (no separate function) |
| `grey_dilation` | `max_filter` over the reflected footprint |
| `grey_opening` / `grey_closing` | erode then dilate / dilate then erode |
| `white_tophat` | input minus opening: compact bright features |
| `black_tophat` | closing minus input: compact dark features |
| `morphological_gradient` | dilation minus erosion |
| `morphological_laplace` | dilation plus erosion minus twice the input |

NaNs propagate through these operations; use `nanmin_filter` or `nanmax_filter`
directly to omit them. Inputs must be `float32` or `float64`.

"Top-hat" has two meanings. `white_tophat` and `black_tophat` are morphology.
Astropy's `Tophat2DKernel` is a disk-shaped smoothing kernel; here that is
`kernels.tophat`, used with `convolve` or as a `footprint`.
