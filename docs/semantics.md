# Neighborhood-statistic filtering semantics

This page covers median, mean, min, max, variance, and standard-deviation
filters and their NaN-aware variants. See [convolution and Gaussian filters](convolution.md)
for arbitrary kernels, shifted origins, and Gaussian selected-axis support.

These statistic filters accept aligned, native-endian NumPy `float32` or `float64` arrays
with rank one or higher. Unaligned arrays are rejected. Scalar `size` and
scalar mode values broadcast to every axis; tuples must match the array rank.
Supported modes are `reflect`, `constant`, `nearest`, `mirror`, and `wrap`,
matching SciPy coordinate mapping.

Inputs are converted with `np.asarray`; masked arrays are rejected before
conversion. Size entries accept the integer-index protocol, including NumPy
integers, and reject booleans. Lists and tuples are accepted for per-axis
sizes and modes, with exactly one entry per dimension. Mode names are
case-sensitive; grid aliases are not supported. `cval` must be a non-boolean
real scalar and is validated even when unused, then cast to the input dtype.
`min_count` must be a non-boolean integer from one through the number of window elements;
`ignore_inf` must be a Python or NumPy boolean.

Windows can exceed axis lengths. A size of one preserves that axis's footprint;
the invalid-value policy still applies. Empty axes return a validated empty
array with the original shape and dtype.

## Window shape: `size`, `footprint`, and `origin`

```python
disk = nf.kernels.tophat(3, ndim=2) != 0  # 7 x 7 disk
local_median = nf.nanmedian_filter(image, footprint=disk)
annulus = nf.kernels.ring(4, 6, ndim=2) != 0
sky = nf.nanmedian_filter(image, footprint=annulus)  # IRAF-style ring median
```

- `size` gives a rectangular window, `3` per axis by default. Even sizes work.
- `footprint` gives any window shape; nonzero entries are used. Pass `size` or
  `footprint`, not both.
- `origin` shifts the window per axis, as in SciPy: element `k` of an axis of
  length `n` reads `center + k - (n // 2 + origin)`, with
  `-(n // 2) <= origin <= (n - 1) // 2`. Positive values move the window toward
  lower indices.
- `min_count` counts window elements, so its upper bound is the number of
  nonzero footprint entries. Boundary repeats count separately.

Window geometry matches SciPy exactly. `ndfiltrs` is still not a drop-in SciPy
replacement: NaN policies differ, the median of an even count is the midpoint of
the two middle values (SciPy returns the upper one), and there is no `axes`
argument.

## Boundary modes

For input `a b c d`, the two symmetric extensions are:

```text
reflect: d c b a | a b c d | d c b a
mirror:  d c b   | a b c d | c b a
```

`reflect` repeats the edge sample; `mirror` does not. `nearest` repeats the
closest edge, `wrap` repeats the opposite edge, and `constant` inserts `cval`.

## Invalid values

`median_filter` propagates a canonical NaN if a neighborhood contains NaN and
retains infinities. `nanmedian_filter` omits NaN, optionally omits infinities,
and returns canonical NaN below `min_count`. Even retained counts use an
overflow-safe arithmetic midpoint. Signed zeros use total floating ordering.
Boundary-created repetitions and `constant`-mode `cval` entries count as
separate footprint positions and follow the same NaN/infinity policy.

Total ordering puts `-0.0` before `+0.0`. Even retained counts compute their
midpoint in `float64`, casting once for `float32` output. Opposite finite
signs use `(a + b) / 2`; same finite signs use `a + (b - a) / 2`.
Equal infinities retain that infinity; opposite infinities produce canonical
NaN; a finite value paired with infinity produces that infinity. The midpoint
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
Retained infinities of one sign yield that infinity; opposing infinities
produce canonical NaN. Infinity signs are tracked separately, so overflow
of a finite subtotal cannot change the result in a window containing infinity.

Both dtypes use compensated (Neumaier) float64 summation in C-order window
traversal, divide by the retained count, then cast once to the input dtype.
Only if the finite subtotal overflows, the window is accumulated again after
power-of-two rescaling based on its retained count. This bounds intermediate
magnitudes; the final rescaled mean is bounded by the retained sample range.
Rescaling can underflow subnormal contributions. Compensation reduces
cancellation error but does not guarantee a correctly rounded exact mean.
Exact zero sums produce positive zero, including a window of negative zeros.
Results are bitwise deterministic across Rayon thread counts.

Ordinary means accumulate samples during traversal without a temporary window
buffer. Overflow recovery gathers and rescales that window only. Weighted
filters use the same streaming approach, replaying samples when products or
subtotals need binary scaling. Interior windows use precomputed flat offsets;
boundary windows still apply the selected mapping on each axis. These changes
preserve tap order and arithmetic rather than using a rolling sum.

Each window is independent, so nonfinite samples never contaminate unrelated
windows through a running sum. This default `method="direct"` takes `O(N * V * d)`
work for `N` pixels, window volume `V`, and rank `d`.

`method="separable"` averages one axis at a time with the same NaN, infinity,
`min_count`, and boundary rules, at a cost proportional to the sum of the window
lengths instead of their product. See [measured comparisons](benchmarks.md)
for speedups and slowdowns. It needs an odd `size`,
no `footprint`, and `origin=0`, and its rounding differs from the direct method in
the last bits. Neither method is a drop-in for
[SciPy uniform_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html),
whose NaN behavior is undefined.

`nanmean_filter` does not accept a bad-pixel mask. Finite hot/dead samples
remain valid; masking those samples requires a separately prepared NaN input.
The filter changes all pixels. `fixpix` changes only masked pixels (linear,
median, or mean repair).

## Minimum and maximum

`min_filter` and `max_filter` share the plain mean/median signatures.
`nanmin_filter` and `nanmax_filter` additionally share `ignore_inf=False` and
`min_count=1` with the other NaN-aware filters. Each returns an actual retained
sample, with no arithmetic rounding. NaNs propagate in plain filters and are
omitted in NaN-aware filters; retained infinities participate in ordering.
Total ordering places negative zero before positive zero, so minimum selects
negative zero and maximum positive zero when both are present. Empty retained
windows and windows below `min_count` return canonical NaN.

Rectangular windows (`size`) use an exact separable sliding-extremum algorithm
(van Herk / Gil-Werman): about three comparisons per sample per axis for any
window length, with results identical bit for bit to visiting every window
element. Geometry setup and boundary padding still depend on the support.
See [measured comparisons](benchmarks.md) for timings.
`footprint` windows visit every element.

## Variance and standard deviation

`var_filter` and `std_filter` add keyword-only `ddof=0` to the plain filter
signature. `nanvar_filter` and `nanstd_filter` also accept `ignore_inf=False`
and `min_count=1`. `ddof` is a nonnegative integer (the integer-index protocol
is accepted; booleans and real-valued floats are rejected). Values beyond the
native unsigned index range raise `OverflowError`. It is not capped by window
volume: when retained count `n <= ddof`, the result is canonical NaN without
warnings. NaN-aware `min_count` is checked independently after omission.

Variance is the sum of squared deviations from the retained-sample mean,
divided by `n - ddof`; standard deviation is its square root. `ddof=0` gives
population variance, and `ddof=1` gives the usual sample-variance correction.
Boundary repetitions count as distinct samples. Plain filters propagate NaNs;
any retained infinity yields canonical NaN for variance and standard deviation,
including a constant infinite window. Use `ignore_inf=True` to omit infinities.
Constant finite windows yield positive zero when the count is sufficient.

Both operations use float64 arithmetic and cast once to the input dtype.
They subtract a retained reference value before scaling deviations, preserving
small differences on a large background. If subtraction overflows, operands
are halved first. Compensated sums compute the mean offset and centered squared
deviations in normalized coordinates. Standard deviation restores the scale
after the square root, independently of variance; it can remain representable
when variance overflows to infinity or underflows to zero. Very small
normalized contributions can still underflow, and exact rounding is not
promised. No data-size or workload heuristic selects an algorithm.

These are local-scatter maps. Astronomical sources, gradients, and detector
noise all contribute; converting scatter to measurement uncertainty requires
additional modeling. The functions return a map with the input shape and
filter all output positions; they are not masked-only repair operations.

Variance and standard deviation visit each window independently, with
`O(N * V * d)` work. The rectangular min/max and optional separable mean paths
described above use different algorithms. No universal speed advantage over
SciPy is asserted.

## Layout, memory, and threads

Non-C-contiguous accepted input is normalized once. The native extension then
takes an owned snapshot before releasing the GIL, so every call copies the
input. Output is new, owned, writable, aligned, and C-contiguous. Concurrent
mutation by Python code after the snapshot cannot affect the result. Mutation
during a non-contiguous normalization copy, or from native code during
validation or snapshot creation, is unsupported. Direct window filters need the
input snapshot, output, `O(window_volume * ndim)` shared offsets, and
`O(T * window_volume + T * ndim)` worker state for `T` workers. Separable paths
also allocate intermediate arrays.
Non-C-contiguous input adds the Python normalization copy.

Rayon uses its process-global thread pool. Set `RAYON_NUM_THREADS` before the
first Rayon-using call in the process; another Rayon consumer can initialize
the pool first.
`nf.get_num_threads()` reports the global pool size and can itself initialize
the pool, so set the environment variable before calling it too.

## Implementation notes

Python validates inputs; PyO3 snapshots them before detaching from the GIL.
Rust owns boundary mapping, N-D traversal, invalid-value filtering, support
counts, and one reusable scratch buffer per Rayon task. `src/reduction.rs`
selects the statistic. Median calls the pure-Rust
`reducers::reducers_1d::median_valid_in_place`; reducers' Python bindings and
axis-level parallel dispatch are not used. The Python reducers distribution
is not needed at runtime.

Local arithmetic supplements the reducers selection primitive:
`src/mean.rs` supplies compensated sums and finite-overflow recovery,
`src/dispersion.rs` avoids unscaled squared intermediates, and local extrema
scans preserve signed-zero ordering without nested parallelism. The arithmetic
contracts above describe their limits. Sharing these improvements upstream
requires preserving those contracts.

Rust checks shape, stride, boundary-period, window-volume, and allocation
arithmetic. Recoverable allocation failures become Python memory errors;
allocator aborts remain an external limitation. The specialized 2-D candidate
is confined to the `benchmark` Cargo feature; measurements do not justify
production dispatch. See [benchmarks](benchmarks.md).

### Explicit exact methods

Use `median_filter(..., method="sortnet")` or
`nanmedian_filter(..., method="sortnet")` for comparator-network selection.
Use `mean_filter(..., method="boxcar")` or
`nanmean_filter(..., method="boxcar")` for exact rolling sums.
Existing defaults remain `method="direct"`; there is no automatic cutoff.

- The odd-even median network is generated for any tap count, including arbitrary
  footprints. It removes unused comparisons and symbolic padding, then runs
  comparisons across adjacent outputs. NaN policies, signed-zero ordering,
  and even-count midpoint rules match direct selection. Mapped borders use
  direct selection. Each tap is encoded once into a reversible integer key in
  floating-point total order; selected keys decode without changing their bits.
- The rolling boxcar supports rectangular N-D windows, even sizes, shifted
  origins, and all boundary modes. Each axis adds entering contributions and
  removes leaving ones using exact integers. Finite means are rounded once
  to the output dtype, nearest with ties to even. This can differ in the last
  bits from the compensated direct mean; it is not a bitwise replacement.
  Exact zero is positive; a negative nonzero mean underflowing to zero stays
  negative. NaNs and infinities have separate exact counts, so an invalid
  sample stops affecting the result when it leaves the neighborhood.

These methods trade memory for less repeated work. A network worker needs
`tap_count × interior_row_width` temporary values; its generated comparison
schedule also grows with support. Rolling means hold two arrays of exact sums
and sample counts. On a 64-bit target these require approximately
`2 × input.size × (8 × L + 32)` bytes, excluding input/output and line scratch.
`L` is the number of 64-bit words needed for the input exponent span and window
count: up to 34 for binary64. It is a proven representation bound, not a
performance cutoff. Wide exponent ranges can therefore be expensive.
Each boxcar axis also caches its entering/leaving boundary indices, using
memory proportional to that axis length rather than the number of lines.

Rolling work after each line's first window is independent of window width.
Initial windows still visit each axis support, so supports much larger than
the image are not constant-cost. See [benchmarks](benchmarks.md).

## Source attribution

`fixpix` follows `astroimred/src/astroimred/imutil/pixels.py`, inspected at
commit `1ba91cb60e95806da9acfe14dbe3af5f1ce7c9d3` (pixel-module revision
`fb70cae872406f907175e3b47d497ffe14b8c238`). Its BSD 3-Clause notice is retained
in `python/ndfiltrs/licenses/astroimred-BSD-3-Clause.txt`. NF implements the
Rust repair itself; astroimred is only an optional comparison dependency.

Published reducers 0.3.2 omits `LICENSE` from its crate. The bundled
`reducers-MIT.txt` comes from its recorded revision
`7625507ac1ff4101c3f97348c0a79a7d07e6ab5e` and
[authoritative LICENSE](https://raw.githubusercontent.com/ysbach/reducers/7625507ac1ff4101c3f97348c0a79a7d07e6ab5e/LICENSE).
The downloaded crate's SHA-256 is
`07d04b1fe8f55b5694092e0afe10dcb9d66aa1a0b7579e092c23a8f2c431e205`;
the notice's SHA-256 is
`a2672db0608261148bbb525fd3869ab0a42b16d24343b1079c646bc67dd5d4da`.

`RUST-DEPENDENCIES.txt` bundles the normal/build dependency notice inventory,
including all target branches and proc macros, excluding dev-only edges.
The generator preserves source texts, versions, and hashes; checks detect drift
and missing expected license texts. Artifact checks verify bundled bytes and
wheel RECORD integrity; `.gitattributes` prevents Windows newline conversion.
This is a source inventory, not a comprehensive legal review of linked code,
system/toolchain components, or separately installed Python dependencies.
The initial inventory uses registry reducers 0.3.2; local overrides must be
resolved before release. See [publishing](../RELEASING.md).

## Robust neighborhood statistics

[Percentiles, MAD, and sigma clipping](robust.md) use the same rectangular
windows and boundary conventions. Percentiles interpolate sorted samples;
MAD measures deviations from the window median. Sigma clipping always omits
nonfinite samples and applies its support threshold after rejection.
