# Background mesh settings

[`mesh_background`](https://ysbach.github.io/ndfiltrs/api/mesh_background.html#ndfiltrs.mesh_background)
estimates a smooth sky background and its RMS from a grid of
boxes ("meshes"). It replaces the BACK step of SExtractor, `sep.Background`,
and `photutils.background.Background2D`. It accepts any number of
dimensions, `float32` or `float64` data, NaNs, and a mask.

For a worked comparison on one star field, see the
[2-D sky background tutorial](background-tutorial.md).

```python
import ndfiltrs as nf
import numpy as np

rng = np.random.default_rng(0)
image = 100 + rng.normal(0, 5, (512, 512))
image[200:210, 300:310] += 1000  # a source
mask = np.zeros(image.shape, dtype=bool)
mask[0:40, 0:40] = True  # a bad region

result = nf.mesh_background(image, 64, mask=mask)
sky = image - result.background
snr = sky / result.rms
result.mesh_background.shape  # (8, 8)
```

The defaults follow SExtractor. [Reference settings](#reference-settings)
gives the settings for sep and photutils output.

## Pipeline

All statistics use `float64`. The output maps are cast once to the input
dtype. Results do not depend on the number of threads.

1. **Meshes.** Along an axis of length `n` with box size `b`, mesh `k` covers
   indices `k*b` to `min((k+1)*b, n) - 1`. The grid has `ceil(n / b)` meshes.
   The last mesh is partial when `b` does not divide `n`. A box larger than
   the axis gives one mesh.
2. **Samples.** A sample is used when it is finite and its `mask` entry is
   zero. NaN, +inf, and -inf are never used.
3. **Mesh statistics.** The samples of a mesh are sorted once. Each clipping
   pass keeps the samples inside `center - lower*spread` to
   `center + upper*spread`, bounds included. Rejected samples never return,
   so the survivors are always one contiguous run of the sorted samples.
   - `estimator="sextractor"` uses SExtractor's sequence on exact samples
     (SExtractor uses a histogram):
     1. Keep the samples within the mean ± 2 standard deviations.
     2. From all samples, keep those within ± 5 deviations of that subset's
        mean. This is SExtractor's histogram range.
     3. Clip at the median ± 3 standard deviations until no sample is
        rejected, at most 100 passes.
     4. The background is the SExtractor mode
        `2.5 median - 1.5 mean`, or the median when
        `|mean - median| >= 0.3 std`, or the mean when `std = 0`. The RMS is
        the population standard deviation of the survivors.
   - `estimator="sigma_clip"` clips once with `sigma`, `maxiters`, `center`,
     and `spread`, as `sigma_clipped_filter` and Astropy `SigmaClip` do. The
     background is `statistic` (`"mode"`, `"mean"`, or `"median"`) and the RMS
     is `rms` (`"std"` or `"mad"`) of the survivors.
   - Standard deviations are population values (`ddof = 0`). `"mad"` is the
     MAD divided by 0.6744897501960817, as `astropy.stats.mad_std`.
4. **Validity.** A mesh is valid when its count is positive and at least
   `min_valid_fraction` times its base. The count is the finite unmasked
   samples (`valid_count="finite"`) or the clipping survivors
   (`"retained"`). The base is the mesh's own pixel count (`valid_base="mesh"`)
   or the full box volume (`"box"`). A partial edge mesh with base `"box"`
   is valid only when it has enough pixels.
5. **Fill.** Each invalid mesh gets a value from valid meshes. Distances are
   Euclidean in mesh-index units.
   - `fill="nearest"` takes the mean of all valid meshes at the smallest
     distance.
   - `fill="idw"` weights the `idw_neighbors` nearest valid meshes by
     `1 / distance**idw_power`. All meshes at the same distance as the last
     selected one are also used, so the result does not depend on axis
     order.
   - With no valid mesh, all grids and maps are NaN.
6. **Median filter.** With `filter_size` larger than 1 on an axis, each mesh
   takes the median of its window in the filled grid. An even count uses the
   mean of the two middle values.
   - `filter_edge="shrink"` reduces the half width on each axis to the
     distance from the nearer edge, so the window stays centered.
   - `"truncate"` drops window positions outside the grid.
   - `filter_threshold=None` replaces every mesh. With
     `threshold_kind="deviation"`, a mesh is replaced when
     `|median - value| >= filter_threshold`. With `"above"`, a mesh is
     replaced when `value > filter_threshold`. The RMS grid is replaced at the
     same meshes, with the median of its own window.
7. **Global values.** `global_background` and `global_rms` are the medians of
   the final grids.
8. **Interpolation.** `interpolation="spline"` evaluates a natural cubic
   spline along each axis in turn (a tensor product). Node `k` is at the
   mesh center `(k + 0.5)*b - 0.5`, also for a partial edge mesh. Outside the
   first and last nodes, the end cubic continues. One mesh on an axis gives a
   constant along it. Two meshes give a straight line. A plane is reproduced
   exactly.

The Rust API exposes the same stages: `MeshBackground::new` validates the
geometry and a `MeshConfig` once per shape, `estimate` returns a `MeshGrid`,
and `interpolate_into` maps any mesh-shaped grid to full resolution.

## Reference settings

| Stage | SExtractor 2.29 / sep 1.4.1 | photutils 3.1.1 `Background2D` | ndfiltrs default |
| --- | --- | --- | --- |
| Box | `BACK_SIZE` 64 / `bw`, `bh` 64 | `box_size` (required) | `box_size` (required) |
| Statistics | histogram, see step 3 | 3σ median clip, `maxiters=10`, SExtractor mode, std | step 3 on exact samples |
| Validity | ≥ 50% finite of the mesh's pixels | ≥ 90% survivors of the full box | as SExtractor |
| Fill | mean of nearest valid meshes | IDW, 10 neighbors, power 1 | as SExtractor |
| Filter | 3×3, centered window shrinks at edges | 3×3, NaN-padded window | as SExtractor |
| Threshold | `BACK_FILTTHRESH` / `fthresh`, deviation | `filter_threshold`, level | as SExtractor |
| Interpolation | 2-D spline, see below | `scipy.ndimage.zoom`, order 3, `reflect`, clipped | natural cubic spline |

sep output with ndfiltrs:

```python
result = nf.mesh_background(
    image,
    (bh, bw),
    mask=mask > maskthresh,
    filter_size=(fh, fw),
    filter_threshold=fthresh,
    interpolation="sextractor",
)
```

photutils meshes with ndfiltrs (photutils `exclude_percentile=10`):

```python
result = nf.mesh_background(
    image,
    box_size,
    mask=mask,
    estimator="sigma_clip",
    maxiters=10,
    min_valid_fraction=0.9,
    valid_count="retained",
    valid_base="box",
    fill="idw",
    filter_edge="truncate",
)
```

## Differences from the references

### SExtractor and sep

- **Statistics.** SExtractor quantizes each mesh into a histogram with bin
  width `10 σ / nlevels`, where `nlevels` is about 0.95 times the pixel
  count, at most 4096. ndfiltrs uses the exact samples. Mesh values differ by
  a few bins. The parity test allows 4 bins. This is an estimate, not a
  bound: the mode moves by `2.5 + 1.5` bins when the median and mean each
  move by one bin, but the clipped sample sets can also differ.
- **Precision.** sep computes in `float32`. ndfiltrs computes in `float64`.
- **Invalid samples.** sep excludes pixels at or below `-1e30` and NaN from
  its first pass, but the histogram step casts NaN to an integer, which is
  undefined behavior in C. ndfiltrs excludes all nonfinite samples. Use
  `mask` for sep comparisons.
- **Filter size.** SExtractor uses `2*(size//2) + 1`, so an even size acts as
  the next odd size. ndfiltrs rejects even sizes.
- **Single-sample mesh statistic.** On a 3×8 image filled with 3, with
  1×1 meshes and filtering disabled, sep 1.4.1 returns a global background
  of 6 and a background map filled with 6. ndfiltrs computes the statistic
  from the sample and returns 3 in both interpolation modes.
- **Global RMS.** When the median RMS is not positive, SExtractor uses the
  median of the positive RMS values, or 1. ndfiltrs reports the median.
- **Interpolation.** The optional `interpolation="sextractor"` targets the
  2-D maps returned by sep 1.4.1. The following measurements use only its
  public `Background(..., fh=1, fw=1).back()` interface and input boxes
  filled with known constants.
  - **Axis offset.** With 8×8 boxes whose levels increase by one per box,
    pixel `[20, 20]` gives 2.0000 for a row ramp and 2.0625 for a column
    ramp. The difference corresponds to half a pixel.
  - **Single row of boxes.** For levels `[0, 3, 1, 8, 4, 2]` in boxes
    five pixels wide, sep gives 1.2 at column 4, the linear interpolation
    between the first two box centers. With five identical rows of boxes,
    sep gives 1.66060 there. A natural cubic spline gives 1.94113.
  - **One-pixel box width.** For the same six levels in one row of boxes
    one pixel wide, sep returns `[0, 3, 6, 9, 12, 15]`. It extrapolates
    the first segment across the rest of the image. The default natural
    spline passes through all six levels at their pixel centers.


### photutils

- **Interpolation.** photutils resizes the grid with
  `scipy.ndimage.zoom(order=3, mode="reflect", grid_mode=True)` and clips the
  map to the grid's range. ndfiltrs uses the natural spline at the same node
  positions. Maps differ most near the image edges (see below).
- **IDW ties.** photutils takes exactly 10 neighbors from a k-d tree. When
  several meshes tie at the 10th distance, the tree's traversal order
  decides, and that order can change with the query. ndfiltrs uses all tied
  meshes.
- **No `coverage_mask` and `fill_value`.** Set those pixels afterwards, for
  example `np.where(coverage, fill_value, result.background)`.
- **Output dtype.** photutils rounds meshes to the input dtype. ndfiltrs
  returns `float64` meshes.
- **Estimators.** Biweight location and scale and custom callables are not
  available.
- **Partial edge boxes.** photutils 3.x computes partial edge boxes from their
  own pixels and has no `edge_method` (earlier versions padded or cropped).
  With `valid_base="box"` and the default fraction, most partial boxes are
  invalid and filled.

### Cost

Mesh statistics sort the samples of each mesh. Both fill methods compare
every invalid mesh with every valid mesh, so the fill cost grows as their
product. This is negligible for typical grids (64×64 meshes) but grows
quickly for grids with many thousands of meshes.

## Interpolation references

The default `"spline"` mode uses a natural cubic spline through mesh
centers. Tests evaluate SciPy's `CubicSpline` with natural boundary
conditions along each axis. The optional 2-D mode targets
[`sep.Background.back()`](https://sep.readthedocs.io/en/stable/api/sep.Background.html).
Constant-valued boxes let the tests measure interpolation separately from
differences in mesh statistics. An isolated implementation used single-node
images, ramps, and impulse responses from the installed sep 1.4.1 binary.
Both modes run in Rust. The 2-D kernel shares horizontal weights across
image rows and uses per-worker scratch space.

Across 694 interpolation probes (16.1 million pixels), the maximum absolute
difference was 2.82 times the float32 spacing at the maximum absolute value
of the reference map. A separate stress case used two identical rows of
levels $1 + 2^{-23}[0,0,1,3,3,0,0,0,0]$ with boxes two pixels high and one
pixel wide. Extrapolation of the first cubic segment amplified rounding.
The last pixel of row 1 was 1.00001446 in ndfiltrs and 0.99999857 in sep
(a difference of 133.28 spacings). Exact float32 parity across all inputs
would require different arithmetic from the float64 used for mesh levels.

## Parity

Tests compare the default mode with a NumPy and SciPy reference
(`tests/_background_reference.py`) in 1-D, 2-D, and 3-D. Values near 100
agree within `1e-11` absolute, and counts and validity flags are equal. With
sep and photutils installed, `tests/test_background.py` also checks the
following cases. The table records the original pipeline measurements.

| Comparison | Measured maximum | Tolerance |
| --- | --- | --- |
| sep, mesh-constant images (fill, filter, interpolation) | 2.4e-7 relative | 4 `float32` ulps |
| sep, noise σ = 5, 32×31 meshes, mesh background / RMS | 1.5 / 0.4 bins (0.078 / 0.021) | 4 bins = 0.21 |
| photutils, mesh background and RMS, `float64` | 5.3e-15 relative | 1e-12 relative |
| photutils, `float32` input | 7.7e-8 relative | 2 `float32` epsilons |
| photutils, IDW-filled meshes without a tie at the 10th neighbor | float64 rounding | 1e-12 relative |

In the 512×512 benchmark image (σ = 5, 64×64 meshes), the full maps differ
from sep by at most 0.037 (background) and 0.0087 (RMS), and from photutils
by at most 0.81 at the edges and 0.10 inside the outer ring of nodes. See
[benchmarks](benchmarks.md#mesh-background) for timings.
