# ndfiltrs

`ndfiltrs` provides NumPy filters implemented in Rust. It reuses the
pure-Rust median-selection kernel from [reducers](https://github.com/ysBach/reducers),
without requiring its Python package at runtime.

Version 0.1.0 provides N-dimensional neighborhood statistics, arbitrary-kernel
convolution, Gaussian filters, and masked-pixel repair.
Inputs must be aligned, native-endian NumPy `float32` or `float64` arrays.
The filters reject unaligned arrays.

Read the [online documentation](https://ysbach.github.io/ndfiltrs/) for the
quickstart, API reference, numerical guides, and benchmarks.
It follows `main` and may include changes not yet released on PyPI.

## Source build

The first PyPI release is not yet available. See [RELEASING.md](RELEASING.md)
for the publishing procedure.

Install a Rust 1.84+ toolchain. Run these commands from the repository root.
They use the project's `.venv`:

```bash
uv sync --frozen --extra dev
uv run --no-sync maturin develop --release --locked
```

To use a shared environment, first check that it contains the build and runtime
dependencies. Activate that environment. Run
`uv run --active --no-sync maturin develop --release --locked`.
Use `--active --no-sync` for every later `uv run` command in that environment.
Never run `uv sync` or `uv pip sync` against a shared environment.

## Developer tests

The GIL tests need the private, test-only hooks. Build the instrumented
extension before running them:

```bash
uv run --no-sync maturin develop --release --locked --features test-hooks
uv run --no-sync pytest -q
```

Release and ordinary source builds omit `test-hooks`.
Their extensions do not contain the private test classes and functions.

Run these checks before submitting changes:

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
cargo fmt --all -- --check
cargo test --locked --no-default-features
cargo test --locked --features benchmark
cargo clippy --locked --features test-hooks -- -D warnings
```

Test numerical changes against independent expected results.
Preserve the versioned [gold image](tests/data/gold_v1/README.md).
Do not regenerate it to fit an implementation.
See [benchmarks/README.md](benchmarks/README.md) for benchmark procedures.

## Build the documentation

Install [Quarto](https://quarto.org/docs/get-started/).
The site build uses version 1.9.37. Create a separate docs environment:

```bash
uv venv .venv-docs --python 3.13
uv pip install --python .venv-docs/bin/python -r docs/requirements-quarto.txt
make docs-build
make docs-preview
```

Check the rendered menus with
`.venv-docs/bin/python -m pytest -q tests/test_docs_navigation.py`.

Edit the source files for each part of the site:

- Guides: `docs/*.md`.
- API descriptions: docstrings in `python/ndfiltrs/`.
- Navigation: `docs/quarto/_quarto.yml`.

The build generates the guide and API pages. Do not edit those generated files.
The site build does not execute examples.

The [docs workflow](.github/workflows/docs.yml) checks pull requests and
publishes pushes to `main`. In GitHub's Pages settings, select **GitHub
Actions** as the source. Allow the `github-pages` environment to deploy from
`main`.

## Use

```python
import ndfiltrs as nf
import numpy as np

image = np.array([[1.0, 9.0], [np.nan, 2.0]])
filtered = nf.nanmedian_filter(image, 3, mode="reflect")
averaged = nf.mean_filter(image, 3)  # NaNs propagate within each window
finite_average = nf.nanmean_filter(image, 3, ignore_inf=True, min_count=2)
local_min = nf.nanmin_filter(image, 3)
local_max = nf.nanmax_filter(image, 3)
local_std = nf.nanstd_filter(image, 3, ddof=1, min_count=2)
local_var = nf.nanvar_filter(image, 3, ddof=1, min_count=2)
```

An integer window size applies to every axis.
To use different sizes, give a tuple with one size per axis.
Boundary modes are `reflect`, `constant`, `nearest`, `mirror`, and
`wrap`. Any window shape works through `footprint`, and `origin` shifts it as in
SciPy:

```python
annulus = nf.kernels.ring(4, 6, ndim=2) != 0
sky = nf.nanmedian_filter(image, footprint=annulus)  # ring median
third = nf.rank_filter(image, 2, footprint=nf.kernels.tophat(1, ndim=2))
```

Window geometry matches SciPy. NaN and infinity policies are explicit. See
[filtering semantics](docs/semantics.md) for the exact contract.

To choose an exact selection network or running-sum mean:

```python
filtered = nf.nanmedian_filter(image, 3, method="sortnet")
smoothed = nf.nanmean_filter(image, 31, method="boxcar")
```

`sortnet` preserves direct median results, including arbitrary footprints.
`boxcar` accepts rectangular sizes and rounds the exact mean once to the output
dtype. Its last bits can differ from the direct compensated sum.
Both methods use more memory to reduce repeated work. Neither is faster for
every input. The default is `method="direct"`.
See [measured comparisons](docs/benchmarks.md).

Set `RAYON_NUM_THREADS` before the first Rayon-using call to configure
parallelism. `nf.get_num_threads()` reports the global pool's thread count.

## Convolution and Gaussian filters

```python
weights = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
weighted_sum = nf.convolve(image, weights)
smoothed = nf.nanconvolve(image, weights)  # Omits NaNs; divides by retained weights
gaussian = nf.nangaussian_filter(image, sigma=1.0, radius=3)
dx = nf.gaussian_filter(image, sigma=1.0, order=(0, 1), radius=3)
laplacian = nf.gaussian_laplace(image, sigma=1.0, radius=3)
```

`convolve` and `correlate` accept signed weights, odd or even kernels, and
`origin` offsets. Their NaN-aware variants always normalize nonnegative
weights: `nanconvolve` and `nancorrelate` are weighted smoothing operations.
Gaussian functions also include `gaussian_gradient_magnitude` and selected-axis
filtering. Smoothing accepts `method="separable"` to avoid a dense product
kernel. The default direct method also supports derivatives and underflowing
product weights. Axis grouping can change floating-point rounding.
See [convolution and Gaussian semantics](docs/convolution.md) for orientation,
support, derivative conventions, and numerical limits.

## Kernels

```python
from ndfiltrs import kernels

psf = kernels.moffat(alpha=1.5, beta=2.5, ndim=2, radius=8)
heavy = kernels.student_t(1.0, df=3, ndim=2, radius=8)
disk = kernels.tophat(3, ndim=2)
smoothed = nf.nanconvolve(image, psf)
```

`box, tophat, ring, gaussian, moffat, student_t` build normalized N-D weights with
optional rotation and subpixel oversampling. `kernels.sample` turns any function,
including any Astropy functional model, into weights. Astropy kernel objects
also work directly as weights. See [kernels](docs/kernels.md) for the Astropy mapping.

## Robust neighborhood statistics

```python
quartile = nf.nanpercentile_filter(image, 25, size=5)
scatter = nf.nanmad_filter(image, size=5)  # Raw median absolute deviation
clipped = nf.sigma_clipped_filter(image, size=5, spread="mad", maxiters=None)
sky = nf.sigma_clipped_filter(image, size=31, maxiters=None, statistic="mode")
level = nf.nanbiweight_location_filter(image, size=15)
```

Percentiles use linear interpolation. Sigma clipping omits all nonfinite values,
rejects outliers iteratively, and returns the retained mean, median, or mode.
`mode_filter` uses the SExtractor, MMM, or IRAF mode formula.
Biweight filters follow `astropy.stats`. See
[robust filtering semantics](docs/robust.md) for scaling, infinity rules,
convergence, and boundary support.

## Edges and morphology

```python
edges = nf.sobel(image, axis=0)
compact = nf.white_tophat(image, footprint=nf.kernels.tophat(5, ndim=2))
```

`sobel`, `prewitt`, `laplace`, and grey morphology (`grey_dilation`, opening,
closing, top-hats, gradient, Laplace) match SciPy. Erosion is `min_filter`.
See [edges and morphology](docs/morphology.md).

## Bad-pixel repair

```python
data = np.array([[1.0, 99.0, 5.0], [2.0, 3.0, 4.0]])
bad = np.array([[False, True, False], [False, False, False]])
repaired = nf.fixpix(data, bad)  # linear; returns a new array
median_repaired = nf.fixpix(data, bad, method="median", size=3)
kernel = nf.kernels.gaussian(1.0, ndim=2)
mean_repaired = nf.fixpix(data, bad, kernel=kernel)  # kernel implies method="mean"
# Allow returning data itself when no bad pixels are present:
result = nf.fixpix(data, bad, copy=False)
```

`fixpix` supports three repair methods:

- Linear interpolation fills whole masked runs along their shortest axis.
  It uses `priority` to break ties. This is the default method.
- Median replacement (`method="median"`) uses original unmasked neighbors.
- Mean replacement (`method="mean"`) also uses original unmasked neighbors, with optional
  `kernel` weights. It resembles Astropy's `interpolate_replace_nans` with
  an explicit mask.

All options, including `method`, are keyword-only. The default linear method
preserves the numerical array behavior of
[astroimred's `fixpix`](https://github.com/ysBach/astroimred/blob/fb70cae872406f907175e3b47d497ffe14b8c238/src/astroimred/imutil/pixels.py),
including crossing-run write order. It releases the GIL.
Median and mean repair compute only masked windows, in parallel.

The default always returns an independent copy.
`copy=False` permits input reuse only when the mask is absent or empty.
Actual repair never mutates the input.

`fixpix` returns an ndarray. The calling astronomy package must handle
FITS/CCDData parsing and header updates. See [repair semantics](docs/fixpix.md) for edges,
nonfinite anchors, dtype restrictions, and the comparison procedure.

## Reusable reference image

[Gold image v1](tests/data/gold_v1/README.md) supplies a fixed 256 x 320 scene.
It includes noiseless and uncorrupted noisy truth layers, sparse and stress masks,
and analytic controls. Tests check file integrity and compare current methods with
independent references. [Preview](docs/images/gold-v1.png).

```bash
uv run --no-sync pytest -q tests/test_gold_image.py
uv run --no-sync python scripts/evaluate_gold_image.py --output /tmp/gold-report.json
```

## Local scatter and mean

`std_filter` and `var_filter` use `ddof=0` by default.
Each window divides its centered sum of squares by `retained_count - ddof`.
NaN-aware versions omit
NaNs, optionally omit infinities, and enforce `min_count` after omission.
Insufficient counts or retained infinities yield NaN. Standard deviation is
computed without first squaring its physical scale, so it can remain finite
when variance overflows. These maps measure local scatter, including image
structure. They are not calibrated detector uncertainty maps.

By default, mean filters use compensated float64 sums and rescale values if
the sum overflows. They cast once to the input dtype.
They evaluate each neighborhood independently, so work scales with window volume.
For large boxcars, `method="separable"` averages one axis at a time.
Its rounding differs in the last bits.

`nanmean_filter` omits NaNs and optionally infinities.
Finite hot or dead pixels still contribute.
It is a full-image filter, not a masked-pixel repair method.

## Release status

Performance depends on the operation, window, array layout, and thread count.
`ndfiltrs` does not promise to outperform SciPy or NumPy on every workload.
The [benchmarks](docs/benchmarks.md) separate measured comparisons
from untested claims. No native IRAF `fixpix` benchmark is available.
Comparisons with [astroimred](https://github.com/ysBach/astroimred)'s `fixpix`
do not establish IRAF performance.

Release tooling targets GitHub and PyPI. Manual release-workflow runs build and
test without uploading to a registry. Matching version tags publish only after
the required CI and wheel jobs pass. See [RELEASING.md](RELEASING.md) for the
first-project setup, rehearsal, tagging commands, and the shared publishing
convention for these packages.

The build pins [reducers 0.4.0](https://crates.io/crates/reducers/0.4.0) without a local path
override. Publishing requires a PyPI trusted publisher and a successful hosted
rehearsal. Consult GitHub Actions for current Linux/macOS/Windows build results.
The Rust API remains private, so crates.io publication is disabled.
See [source attribution](docs/semantics.md#source-attribution) for provenance.

## License

`ndfiltrs` uses the [MIT License](LICENSE).
The `fixpix` behavioral source is [astroimred](https://github.com/ysBach/astroimred).
The package includes its [BSD notice](python/ndfiltrs/licenses/astroimred-BSD-3-Clause.txt).
The Rust [reducers 0.4.0](https://crates.io/crates/reducers/0.4.0) dependency
declares MIT and BSD-3-Clause licenses.
The package includes [both notices](python/ndfiltrs/licenses/).
