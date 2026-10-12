# Masked-pixel repair

```text
fixpix(array, mask=None, *, method=None, priority=None,
       size=None, footprint=None, kernel=None, origin=None, mode=None,
       cval=None, ignore_inf=None, min_count=None, copy=True)
```

The signature above shows keyword-only arguments. `fixpix` accepts dense,
aligned, native-endian `float32` or `float64` arrays of rank one or higher.
By default it returns a new, owned, writable, C-contiguous ndarray of the same
shape and dtype. Input and mask are never mutated. All options after `mask`,
including `method`, are keyword-only.

## Copies and empty masks

`copy=True` preserves the independent-output contract, including empty arrays,
`mask=None`, and masks without bad pixels. The no-repair path checks the mask
before normalizing image layout, allocating native state, or performing repair.

`copy=False` permits returning `np.asarray(array)` directly when the mask is
absent or contains no bad pixels. For a plain ndarray this is the same object,
including its strides, writeability, and shared storage. Modifying that result
can therefore modify the input. Array-like conversion may itself allocate.
When any bad pixels are present, both settings return a fresh repaired array;
`copy=False` does not enable in-place repair.

```python
empty_mask = np.zeros(image.shape, dtype=bool)
unchanged = nf.fixpix(image, empty_mask, copy=False)
assert unchanged is image  # for a plain NumPy ndarray
```

No-op calls still validate dtype, rank, mask shape, method, and every option.
Checking an empty dense mask requires examining the mask; a copy is not needed
with `copy=False`. Timing depends on shape, mask layout/dtype, and the machine.

The mask must have exactly the input shape. Boolean and numeric masks are
accepted; nonzero values mean bad. Masked arrays, object/string masks, and
broadcasting are rejected. NaNs and infinities in the data are not implicitly
masked. `priority` applies only to linear repair and must be a complete permutation of the nonnegative axis
numbers; integer-index objects are accepted and booleans are rejected. Its
default is descending axis order, so the horizontal axis wins 2-D ties.

## Linear interpolation contract

For every masked pixel, find its contiguous masked run along every axis in
the original mask. Select the shortest run; ties use `priority`. The entire
selected run is interpolated between the immediately adjacent unmasked
anchors. At an edge, use the single anchor at both ends. A selected run that
spans its whole axis has no anchors and performs no write; there is no fallback
to a longer run on another axis.

Crossing runs follow the final result of repeated C-order whole-run writes in
[astroimred.fixpix](https://github.com/ysBach/astroimred/blob/fb70cae872406f907175e3b47d497ffe14b8c238/src/astroimred/imutil/pixels.py#L114).
This is deliberately different from independently filling
each bad pixel only along that pixel's chosen axis. Unmasked pixels are never
written. A fully masked image is unchanged.

For a run of length `L`, position `k` from one through `L` receives
`((b - a) / (L + 1)) * k + a`, calculated in `float64` and cast once to the
input dtype. This preserves the source's arithmetic rather than changing
rounding or overflow behavior. In particular, nonfinite anchors can create
NaNs, including an infinite sole edge anchor because `inf - inf` is NaN.
Opposite extreme finite anchors can overflow the subtraction. Mark invalid
anchors in the input mask when they should not participate in interpolation.

## Median repair

```python
repaired = nf.fixpix(image, bad, method="median", size=5, min_count=3)
```

The example's window and support count are caller choices. Only masked pixels
receive estimates. Each window (`size` or `footprint`, shifted by `origin`) uses the **original**
unmasked samples. Originally masked samples and NaNs are always omitted;
repaired values never feed later windows. Unmasked pixels are bitwise unchanged.

Median-only options use `None` to request these defaults:

| Option | Median default | Meaning |
|---|---|---|
| `size` | `3` | Positive scalar or one size per axis |
| `footprint` | none | Any window shape; nonzero entries are used (not with `size`) |
| `origin` | `0` | Window shift per axis, as in `median_filter` |
| `mode` | `"reflect"` | Boundary mode, scalar or one per axis |
| `cval` | NaN | Value outside `"constant"` boundaries |
| `ignore_inf` | `False` | Whether to omit infinities as well as NaNs |
| `min_count` | `1` | Minimum retained footprint samples |

The five modes and midpoint arithmetic match `nanmedian_filter`. Mapped
boundary repetitions count separately toward `min_count`; constant samples
follow the same NaN/infinity policy. Use `mode="constant"` with the default
NaN `cval` to omit exterior samples. Insufficient support yields canonical NaN.
A completely masked image yields NaNs unless usable constant boundary samples
supply enough support. There is no automatic window expansion or linear fallback.

Any non-`None` option that does not apply to the chosen method raises
`ValueError`. Explicit `None` means unspecified. Unknown methods, including
`"cubic"`, are rejected.

## Mean repair

```python
repaired = nf.fixpix(image, bad, method="mean", size=5)  # equal weights
kernel = nf.kernels.gaussian(1.5, ndim=2)
smooth = nf.fixpix(image, bad, kernel=kernel)  # weighted; implies method="mean"
```

Each masked pixel becomes the mean of the **original** unmasked samples around
it, weighted by `kernel` if given (equal weights over `size` or `footprint`
otherwise; pass one of the three). Kernel element `k` weights the neighbor at
offset `k - center`, the same window geometry and `origin` as median repair. The
result equals `nancorrelate` of the image with masked pixels set to NaN; for a
symmetric kernel it is Astropy's `interpolate_replace_nans` with an explicit mask.
Weights must be finite and nonnegative. `mode`, `cval` (default NaN),
`ignore_inf`, and `min_count` behave as for median repair.

Unmasked pixels are unchanged. Unmasked NaNs are omitted from the averages but
stay NaN; pass `mask | np.isnan(image)` to repair them too. Only masked pixels
are evaluated, so the cost scales with the number of bad pixels.

## Work and memory

The mask is checked for any bad pixels before normalization. For nonempty
masks, native position extraction visits machine-word chunks and skips words
whose bytes are all zero, preserving every nonzero byte's C-order position.
This is a general bitwise optimization; no density or workload threshold is used.
NumPy creates a C-order image snapshot directly, including from strided input,
without a separate zero-fill or contiguous-image intermediate.

Linear repair retains its run selection and crossing-write order and computes
serially. Median and mean repair mark original bad positions as NaN in the
private snapshot, compute all `K` estimates before writing them back, and reuse
the existing window traversal and arithmetic. They visit `K` windows rather than
all `N` image positions, split across Rayon threads; results do not depend on the
thread count. Beyond the output, storage is `O(K + window_volume + ndim)`;
noncontiguous or nonboolean masks can additionally require normalization
storage. All methods release the GIL.

## Relationship to astroimred

For the default linear method and supported array inputs, compare with
`air.fixpix(array, mask, priority=priority, update_header=False,
verbose=False).data`. `ndfiltrs` does not accept FITS paths or manage CCDData
units, masks, uncertainties, or headers. Integer, complex, non-native-endian,
and scalar inputs are outside this API's scope; it never silently casts image
data to a different dtype. Priority validation is stricter than the current
`air.fixpix` implementation for malformed permutations.

The native code snapshots input data and masked positions before releasing the
GIL. Python mutation or reallocation after that snapshot cannot alter the
running result. Mutation by native code during validation or copies, or during
a NumPy normalization copy, is unsupported. Linear repair writes are serial to
preserve crossing-run order.

## Verification

Tests compare with an independent SciPy-based run reference, literal edge and
nonfinite expectations, and the real `air.fixpix` when installed. Small masks
are enumerated exhaustively; seeded N-D cases exercise priorities and layouts.
Private test hooks check both GIL progress and source reallocation after the
snapshot. Benchmarks must compare equivalent numerical results before timing
and include public-call copies and validation. Speedup is workload-dependent;
there is no fixed speedup guarantee. Native IRAF `fixpix` has not been timed.
Comparisons with `astroimred` are not measurements of IRAF itself.

## Implementation notes

Linear repair copies directly into private owned output and records original
masked offsets. Sorting those offsets by axis line and position groups runs.
Each bad pixel selects its shortest run; priority breaks ties. Anchors are
outside the original mask, so repeating a run produces identical values.
Retaining its last C-order trigger and applying runs in trigger order preserves
crossing writes while eliminating redundant interpolation. Writes stay serial
after GIL detachment.

For `N` pixels, `B` masked pixels, and rank `d`, the work is
`O(N + d B log B)` with `O(B + d)` native scratch beyond output. Retained runs
have at most `d B` interpolation writes. Strided arrays are copied directly
into output; mask normalization may need another copy. No workload threshold
changes the algorithm. Median repair evaluates only original masked windows
and scatters estimates after all windows finish, so repairs never feed each
other.

## Sparse repair measurements

`benchmarks/benchmark_fixpix_sparse.py` compares linear and median repair,
no-op aliasing, and full-image median-filter composition. It checks numerical
agreement, ownership, and source preservation before timing public calls in
shuffled order. Linear interpolation and median repair are different estimators;
their timings describe cost, not interchangeable scientific results. The
full-image median comparison uses `ndfiltrs`, not SciPy.

Measured on 2026-09-06 using release wheels built with Rust 1.84.0, macOS
26.6 ARM64, CPython 3.13.11, and NumPy 2.5.2. Medians of seven rounds,
five calls per timing sample, after numerical checks and warmup. For
512 x 512 float64 images and a 3 x 3 median window, times rounded to two
significant figures are:

| Bad pixels | Linear, copy (µs) | Linear, copy=False (µs) | Median, copy (µs) |
|---:|---:|---:|---:|
| 0 | 30 | 5.2 | 32 |
| 1 | 40 | 39 | 45 |
| 26 | 36 | 36 | 38 |
| 262 | 43 | 49 | 57 |

For zero bad pixels, `copy=False` returns the original ndarray; all other
results in the table are independent arrays. At 0.1% bad pixels (262), median
repair took 57 µs versus 14 ms for full-image median-filter composition.
This comparison measures the benefit of filtering only masked positions on
this sparse input, not general SciPy performance. The report also covers
float32 and 2048 x 2048 images. Results vary with image, window, layout, cache
state, and hardware; runtime and peak memory in a scientific pipeline need
to be measured for that pipeline.

Raw measurement reports remain local. Use the
[benchmark commands](../benchmarks/README.md) to generate timings and source hashes.
