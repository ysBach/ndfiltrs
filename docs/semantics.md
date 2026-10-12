# Neighborhood-statistic filtering semantics

This page covers median, mean, min, max, variance, and standard-deviation
filters and their NaN-aware variants. See [convolution and Gaussian filters](convolution.md)
for arbitrary kernels, shifted origins, and Gaussian selected-axis support.

## Check input requirements

These statistic filters accept aligned, native-endian NumPy `float32` or `float64` arrays
with rank one or higher. They reject unaligned arrays. Scalar `size` and
scalar mode values broadcast to every axis. Tuples must match the array rank.
Supported modes are `reflect`, `constant`, `nearest`, `mirror`, and `wrap`,
matching SciPy coordinate mapping.

The filters convert inputs with `np.asarray`. They reject masked arrays before
conversion. Size entries accept the integer-index protocol, including NumPy
integers, but reject booleans. Per-axis sizes and modes accept lists and tuples
with exactly one entry per dimension. Mode names are case-sensitive.
The filters do not support grid aliases.

`cval` must be a non-boolean real scalar. The filters validate it even when
unused, then cast it to the input dtype.
`min_count` must be a non-boolean integer from one through the number of window elements.
`ignore_inf` must be a Python or NumPy boolean.

Windows can exceed axis lengths. A size of one preserves that axis's footprint.
The invalid-value policy still applies. Empty axes return a validated empty
array with the original shape and dtype.

## Window shape: `size`, `footprint`, and `origin`

```python
disk = nf.kernels.tophat(3, ndim=2) != 0  # 7 x 7 disk
local_median = nf.nanmedian_filter(image, footprint=disk)
annulus = nf.kernels.ring(4, 6, ndim=2) != 0
sky = nf.nanmedian_filter(image, footprint=annulus)  # IRAF-style ring median
```

- `size` gives a rectangular window, `3` per axis by default. Even sizes work.
- `footprint` gives any window shape. Nonzero entries select samples. Pass `size` or
  `footprint`, not both.
- `origin` shifts the window per axis, as in SciPy. Element `k` of an axis of
  length `n` reads `center + k - (n // 2 + origin)`.
  The allowed range is `-(n // 2) <= origin <= (n - 1) // 2`. Positive values move the window toward
  lower indices.
- `min_count` counts window elements, so its upper bound is the number of
  nonzero footprint entries. Boundary repeats count separately.

Window geometry matches SciPy exactly. However, `ndfiltrs` is not a drop-in SciPy
replacement:

- NaN policies differ.
- For the median, an even count gives the midpoint of the two middle values. SciPy returns the upper one.
- These filters have no `axes` argument.

## Boundary modes

For input `a b c d`, the two symmetric extensions are:

```text
reflect: d c b a | a b c d | d c b a
mirror:  d c b   | a b c d | c b a
```

`reflect` repeats the edge sample. `mirror` does not. `nearest` repeats the
closest edge, `wrap` repeats the opposite edge, and `constant` inserts `cval`.

## Invalid values

`median_filter` propagates a canonical NaN if a neighborhood contains NaN and
retains infinities. `nanmedian_filter` omits NaN, optionally omits infinities,
and returns canonical NaN below `min_count`.

Even retained counts use an overflow-safe arithmetic midpoint.
Signed zeros use total floating ordering.
Boundary-created repetitions and `constant`-mode `cval` entries count as
separate footprint positions and follow the same NaN/infinity policy.

Total ordering puts `-0.0` before `+0.0`. Even retained counts compute their
midpoint in `float64`, casting once for `float32` output. Opposite finite
signs use `(a + b) / 2`. Same finite signs use `a + (b - a) / 2`.

Equal infinities retain that infinity. Opposite infinities produce canonical
NaN. A finite value paired with infinity produces that infinity. The midpoint
of `-0.0` and `+0.0` is `+0.0`.

Generated NaNs have bit patterns `0x7fc00000` for `float32` and
`0x7ff8000000000000` for `float64`. Results are bitwise deterministic across
Rayon thread counts.

```python
import ndfiltrs as nf
import numpy as np

line = nf.median_filter(np.array([1.0, 9.0, 2.0]), 3)
image = nf.nanmedian_filter(np.ones((8, 8)), (3, 5), mode="mirror")
volume = nf.median_filter(np.ones((4, 8, 8), np.float32), (1, 3, 3))
```

## Arithmetic means

`mean_filter` computes an unweighted arithmetic mean of each neighborhood (a
boxcar filter). `nanmean_filter` omits NaNs and optionally infinities, and
returns canonical NaN below `min_count`, without an empty-window warning.
Window positions created by boundary mapping count separately. A finite
`cval` participates in both the sum and the retained count.

Plain means propagate NaNs only within neighborhoods containing them.
Retained infinities of one sign yield that infinity. Opposing infinities
produce canonical NaN. The filters track infinity signs separately.
Overflow of a finite subtotal therefore cannot change the result in a window containing infinity.

### Direct means and rounding

With `method="direct"`, both dtypes use compensated (Neumaier) float64 summation
in C-order window traversal. The filters divide by the retained count, then
cast once to the input dtype.

Only if the finite subtotal overflows, the filters accumulate the window again
after power-of-two rescaling based on its retained count. This bounds intermediate
magnitudes. The final rescaled mean stays within the retained sample range.
Rescaling can underflow subnormal contributions. Compensation reduces
cancellation error but does not guarantee a correctly rounded exact mean.

Exact zero sums produce positive zero, including a window of negative zeros.
Results are bitwise deterministic across Rayon thread counts.

Ordinary means accumulate samples during traversal without a temporary window
buffer. Overflow recovery gathers and rescales that window only. Weighted
filters use the same streaming approach, replaying samples when products or
subtotals need binary scaling.

Interior windows use precomputed flat offsets.
Boundary windows still apply the selected mapping on each axis. Direct
traversal preserves tap order and does not use a rolling sum.

Each window is independent, so nonfinite samples never contaminate unrelated
windows through a running sum. This default `method="direct"` takes `O(N * V * d)`
work for `N` pixels, window volume `V`, and rank `d`.

### Choose separable means

`method="separable"` averages one axis at a time. It uses the same NaN, infinity,
`min_count`, and boundary rules. Its cost is proportional to the sum of the window
lengths instead of their product. See [measured comparisons](benchmarks.md)
for speedups and slowdowns.

Separable means require an odd `size`, no `footprint`, and `origin=0`.
Their rounding differs from the direct method in the last bits. Neither method is a drop-in for
[SciPy uniform_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html),
whose NaN behavior is undefined.

### Mask bad pixels before filtering

`nanmean_filter` does not accept a bad-pixel mask. Finite hot/dead samples
remain valid. Masking those samples requires a separately prepared NaN input.
The filter changes all pixels. `fixpix` changes only masked pixels (linear,
median, or mean repair).

## Minimum and maximum

`min_filter` and `max_filter` share the plain mean/median signatures.
`nanmin_filter` and `nanmax_filter` additionally share `ignore_inf=False` and
`min_count=1` with the other NaN-aware filters. Each returns an actual retained
sample, with no arithmetic rounding. NaNs propagate in plain filters and are
omitted in NaN-aware filters. Retained infinities participate in ordering.

Total ordering places negative zero before positive zero, so minimum selects
negative zero and maximum positive zero when both are present. Empty retained
windows and windows below `min_count` return canonical NaN.

Rectangular windows (`size`) use an exact separable sliding-extremum algorithm
(van Herk / Gil-Werman). It uses about three comparisons per sample per axis
for any window length. Results match bit for bit those from visiting every window
element. Geometry setup and boundary padding still depend on the support.
See [measured comparisons](benchmarks.md) for timings.
`footprint` windows visit every element.

## Variance and standard deviation

`var_filter` and `std_filter` add keyword-only `ddof=0` to the plain filter
signature. `nanvar_filter` and `nanstd_filter` also accept `ignore_inf=False`
and `min_count=1`.

`ddof` must be a nonnegative integer. It accepts the integer-index protocol
but rejects booleans and real-valued floats. Values beyond the native unsigned
index range raise `OverflowError`. Window volume does not limit `ddof`.
When retained count `n <= ddof`, the result is canonical NaN without warnings.
NaN-aware filters check `min_count` independently after omission.

Variance is the sum of squared deviations from the retained-sample mean,
divided by `n - ddof`. Standard deviation is its square root. `ddof=0` gives
population variance, and `ddof=1` gives the usual sample-variance correction.
Boundary repetitions count as distinct samples.

Plain filters propagate NaNs.
Any retained infinity yields canonical NaN for variance and standard deviation,
including a constant infinite window. Use `ignore_inf=True` to omit infinities.
Constant finite windows yield positive zero when the count is sufficient.

Both operations use float64 arithmetic and cast once to the input dtype.
They subtract a retained reference value before scaling deviations, preserving
small differences on a large background. If subtraction overflows, the filters
halve operands first. Compensated sums compute the mean offset and centered squared
deviations in normalized coordinates.

Standard deviation restores the scale after the square root, independently of variance.
It can remain representable when variance overflows to infinity or underflows to zero. Very small
normalized contributions can still underflow, and exact rounding is not
promised. No data-size or workload heuristic selects an algorithm.

These are local-scatter maps. Astronomical sources, gradients, and detector
noise all contribute. Converting scatter to measurement uncertainty requires
additional modeling. The functions return a map with the input shape and
filter all output positions. They are not masked-only repair operations.

Variance and standard deviation visit each window independently, with
`O(N * V * d)` work. The rectangular min/max and optional separable mean paths
described above use different algorithms. No universal speed advantage over
SciPy is asserted.

## Layout, memory, and threads

Non-C-contiguous accepted input is normalized once. The native extension then
takes an owned snapshot before releasing the GIL, so every call copies the
input. Output is new, owned, writable, aligned, and C-contiguous.

Concurrent mutation by Python code after the snapshot cannot affect the result. Mutation
during a non-contiguous normalization copy, or from native code during
validation or snapshot creation, is unsupported.

Direct window filters need the following storage:

- The input snapshot and output.
- `O(window_volume * ndim)` shared offsets.
- `O(T * window_volume + T * ndim)` worker state for `T` workers.

Separable paths also allocate intermediate arrays.
Non-C-contiguous input adds the Python normalization copy.

### Set the thread count before calling Rayon

Rayon uses its process-global thread pool. Set `RAYON_NUM_THREADS` before the
first Rayon-using call in the process. Another Rayon consumer can initialize
the pool first.

`nf.get_num_threads()` reports the global pool size and can itself initialize
the pool. Set the environment variable before calling it too.

## Implementation notes

Python validates inputs. PyO3 snapshots them before detaching from the GIL.
Rust owns boundary mapping, N-D traversal, invalid-value filtering, support
counts, and one reusable scratch buffer per Rayon task. `src/reduction.rs`
selects the statistic. Median calls the pure-Rust
[reducers::reducers_1d::median_valid_in_place](https://docs.rs/reducers/0.4.0/reducers/reducers_1d/fn.median_valid_in_place.html).
The implementation does not use reducers' Python bindings or axis-level
parallel dispatch. The Python reducers distribution
is not needed at runtime.

Local arithmetic supplements the reducers selection primitive:

- `src/mean.rs` supplies compensated sums and finite-overflow recovery.
- `src/dispersion.rs` avoids unscaled squared intermediates.
- Local extrema scans preserve signed-zero ordering without nested parallelism.

The arithmetic contracts above describe their limits.

Rust checks shape, stride, boundary-period, window-volume, and allocation
arithmetic. Recoverable allocation failures become Python memory errors.
Allocator aborts remain an external limitation. See [benchmarks](benchmarks.md)
for measured performance and comparison conditions.

### Choose an explicit exact method

Use `median_filter(..., method="sortnet")` or
`nanmedian_filter(..., method="sortnet")` for comparator-network selection.
Use `mean_filter(..., method="boxcar")` or
`nanmean_filter(..., method="boxcar")` for exact rolling sums.
The default is `method="direct"`. There is no automatic cutoff.

#### Median networks

The implementation generates an odd-even median network for any tap count,
including arbitrary footprints. It removes unused comparisons and symbolic
padding, then runs comparisons across adjacent outputs. NaN policies,
signed-zero ordering, and even-count midpoint rules match direct selection.
Mapped borders use direct selection.

Each tap becomes a reversible integer key in floating-point total order,
encoded once. Selected keys decode without changing their bits.

#### Rolling boxcar means

The rolling boxcar supports rectangular N-D windows, even sizes, shifted
origins, and all boundary modes. Each axis adds entering contributions and
removes leaving ones using exact integers. Finite means are rounded once
to the output dtype, nearest with ties to even.

This can differ in the last bits from the compensated direct mean.
It is not a bitwise replacement. Exact zero is positive.
A negative nonzero mean underflowing to zero stays negative.

NaNs and infinities have separate exact counts. Thus, an invalid sample stops
affecting the result when it leaves the neighborhood.

#### Memory and work

These methods trade memory for less repeated work. A network worker needs
`tap_count × interior_row_width` temporary values. Its generated comparison
schedule also grows with support.

Rolling means hold two exact-sum arrays and two sample-count arrays.
On a 64-bit target these require approximately
`2 × input.size × (8 × L + 32)` bytes, excluding input/output and line scratch.

`L` is the number of 64-bit words needed for the input exponent span and window
count. Its upper bound is 34 for binary64. This is a proven representation bound, not a
performance cutoff. Wide exponent ranges can therefore be expensive.

Each boxcar axis also caches its entering/leaving boundary indices, using
memory proportional to that axis length rather than the number of lines.

Rolling work after each line's first window is independent of window width.
Initial windows still visit each axis support, so supports much larger than
the image are not constant-cost. See [benchmarks](benchmarks.md).

## Source attribution

`fixpix` follows astroimred's
[pixel module at fb70cae](https://github.com/ysBach/astroimred/blob/fb70cae872406f907175e3b47d497ffe14b8c238/src/astroimred/imutil/pixels.py).
Its [BSD 3-Clause notice](../python/ndfiltrs/licenses/astroimred-BSD-3-Clause.txt)
is bundled with `ndfiltrs`. The repair implementation is native Rust.
Astroimred is only an optional comparison dependency.

The bundled [reducers MIT notice](../python/ndfiltrs/licenses/reducers-MIT.txt) comes from
[revision 7625507](https://github.com/ysBach/reducers/commit/7625507ac1ff4101c3f97348c0a79a7d07e6ab5e) and the
[authoritative LICENSE](https://raw.githubusercontent.com/ysbach/reducers/7625507ac1ff4101c3f97348c0a79a7d07e6ab5e/LICENSE).

The [Rust dependency notices](../python/ndfiltrs/licenses/RUST-DEPENDENCIES.txt)
bundle the normal/build dependency inventory,
including all target branches and proc macros, excluding dev-only edges.
The generator preserves source texts, versions, and hashes. Checks detect drift
and missing expected license texts. Artifact checks compare bundled bytes and check
wheel RECORD integrity. `.gitattributes` prevents Windows newline conversion.

This is a source inventory, not a comprehensive legal review of linked code,
system/toolchain components, or separately installed Python dependencies.
The inventory includes [reducers 0.4.0](https://crates.io/crates/reducers/0.4.0)
and its MIT and BSD 3-Clause
notices. See [publishing](../RELEASING.md) for release checks.

## Robust neighborhood statistics

[Percentiles, MAD, and sigma clipping](robust.md) use the same rectangular
windows and boundary conventions. Percentiles interpolate sorted samples.
MAD measures deviations from the window median. Sigma clipping always omits
nonfinite samples and applies its support threshold after rejection.
