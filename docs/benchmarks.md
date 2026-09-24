# Benchmark evidence

Benchmark JSON reports are local measurement artifacts and are not distributed
with the source. The tables below summarize recorded runs; use the
[benchmark commands](../benchmarks/README.md) to produce reports containing
raw samples, validation errors, source hashes, and your environment details.

## Sortnet and boxcar

Public calls now accept `median_filter(..., method="sortnet")` and
`mean_filter(..., method="boxcar")`, including their NaN-omitting variants.
Defaults remain unchanged. These methods are explicit choices, not automatic
dispatch based on image or window size.

CPU sampling of the release extension identified the median comparator loop
as the dominant cost; Python was a small fraction of samples. Generated ARM64
instructions already used SIMD, but repeatedly transformed floating-point
bits for total ordering. The updated network encodes each tap once, compares
integer keys, and decodes only the selected outputs. An odd-even merge
topology replaces the bitonic topology for every support, with unused outputs
pruned by the same dependency rule. There are no hard-coded small networks.

The boxcar profile identified exact-state processing and rounding, not Python,
as the main work. Input values now encode directly into their proven compact
integer span. Entering/leaving boundary indices are computed once per axis
and reused across lines. Rounding remains exact. Fusing final-axis rounding
into the rolling loop regressed timings and was not retained.

The final local one-thread and four-thread reports
time complete public calls, including validation, snapshots, and allocation.
Each uses the same 30-case matrix and seven rotating repetitions. The paired
baseline is the original native-candidate package immediately before the
public methods and optimizations in this section, not the original direct
implementation. No tests, builds, or profilers ran during these final timings.

| 256×256 input, one thread | Previous candidate (ms) | Public method (ms) | SciPy (ms) | Astro-SCRAPPY (ms) |
| --- | ---: | ---: | ---: | ---: |
| Median, float32, 3×3 | 0.685 | 0.585 | 5.153 | 0.042 |
| Median, float32, 5×5 | 2.330 | 1.654 | 16.190 | 11.603 |
| Median, float32, 7×7 | 5.678 | 4.206 | 28.327 | 24.801 |
| Median, float64, 31×31 | 429.506 | 289.483 | 472.791 | — |
| Boxcar, float64, 15×15 | 3.968 | 3.182 | 0.331 | — |
| Boxcar, float64, 31×31 | 4.254 | 3.543 | 0.386 | — |
| Boxcar, float64, 63×63 | 4.504 | 3.647 | 0.365 | — |

These are improvements, **not a universal speed win**. At one thread, sortnet
beats SciPy in 15/18 cases and Astro-SCRAPPY in the measured 5×5 and 7×7 cases;
Astro-SCRAPPY remains about 14× faster at 3×3. Boxcar improves over the prior
exact candidate in all 12 cases but loses to SciPy in all 12. Its 63×63
float64 case is about 10× slower than SciPy and about 8× faster than the prior
separable method (29.099 ms).

Regressions remain visible: the 1-D float64 size-31 sortnet case is about 5%
slower than its previous network, and 31×31 float64 sortnet remains slower
than direct selection (151.937 ms). Do not infer an automatic crossover from
these measurements. Astro-SCRAPPY copies borders, whereas ndfiltrs computes
them; matching interiors are checked bitwise. SciPy's floating-point rolling
mean does not provide the exact-sum rounding contract of `boxcar`.

### Earlier native-candidate measurements

The following reports predate public methods and the profiling-led changes
above. They describe the original native candidates, not current timings.

`median_network` generates and prunes a comparator network for any tap count,
then vectorizes comparisons across adjacent outputs. It applies the useful
idea from Astro-SCRAPPY without a hardcoded 3×3 implementation or an
approximate row/column median. It preserves full-image results, including
reflected borders, signed zeros, and NaN policies.

`mean_rolling` reuses exact integer sums between overlapping windows and axis
passes. It rounds the final mean once to the requested dtype. Separate counts
keep NaNs and infinities local to their neighborhoods. This avoids the
cancellation drift of a floating-point subtract/add recurrence, but costs
more memory and arithmetic than SciPy's floating-point boxcar.

The earlier local one-thread and four-thread reports compare
30 cases with seven calls per runner, rotating timing order. Candidate calls
include the same Python argument validation as public calls, snapshots, and
allocation. Both dtypes and 1-D/2-D/3-D are included. The paired prior package
is a snapshot immediately before these two candidates, not the earlier
wheel used in the older table below.

Measured on macOS 26.6 ARM64, Python 3.13.11, NumPy 2.5.2, SciPy 1.18.1,
and Astro-SCRAPPY 1.3.0, with a release extension built using Rust 1.96.0:

| 256×256 input, one thread | Prior ndfiltrs (ms) | Native candidate (ms) | SciPy (ms) | Astro-SCRAPPY (ms) |
| --- | ---: | ---: | ---: | ---: |
| Exact median, float32, 3×3 | 3.714 | 0.621 | 4.665 | 0.035 |
| Exact median, float32, 5×5 | 8.070 | 2.238 | 15.238 | 11.329 |
| Exact median, float32, 7×7 | 11.094 | 5.449 | 27.132 | 24.165 |
| Exact median, float64, 31×31 | 143.940 | 413.567 | 442.709 | — |
| Boxcar, float64, 15×15 | 6.822 separable | 3.906 | 0.322 | — |
| Boxcar, float64, 31×31 | 14.051 separable | 3.998 | 0.334 | — |
| Boxcar, float64, 63×63 | 28.679 separable | 4.362 | 0.351 | — |

The 3×3 candidate improves on the prior implementation by about **6×**, but
Astro-SCRAPPY remains about **18× faster** in that case. The 63×63 rolling
boxcar is about **6.6× faster** than the prior separable path and **121× faster**
than the prior direct path (528.942 ms); SciPy remains about **12× faster**.
These are full-call measurements, not identical border work for Astro-SCRAPPY
or identical summation accuracy for SciPy.

Regressions prevent a universal replacement: the 31×31 float64 network is
about 2.9× slower than direct selection. A 1-D float64 boxcar of width 31 on
16,384 samples takes 0.713 ms with rolling sums versus 0.384 ms for the prior
direct path. No size cutoff or automatic choice is inferred from this matrix.

Median parity is checked bitwise over complete outputs. Astro-SCRAPPY is
checked only on the matching float32 interior because it copies its borders.
Rolling means are checked against exact rational references in tests; these
finite-input benchmarks only check agreement within a 64-epsilon input-scale
absolute tolerance. The raw reports retain errors and every sample, including
regressions. See [memory and rounding contracts](semantics.md#explicit-exact-methods)
and [reproduction commands](../benchmarks/README.md#sortnet-and-boxcar-methods).

## Matched SciPy and NumPy calls

`ndfiltrs` is not faster on every operation. The optimized release build was
measured on 2026-10-12 with macOS 26.6 ARM64, Python 3.13.11, NumPy 2.5.2,
and SciPy 1.18.1. The original pre-optimization wheel runs in the **same
process and NumPy environment** as the new extension.

| Operation, 256×256 float64 | Window | Before (ms) | After (ms) | SciPy (ms) |
| --- | --- | ---: | ---: | ---: |
| Mean, direct | 3×3 | 1.085 | 0.433 | 0.502 |
| Mean, direct / separable | 15×15 | 16.416 / 3.383 | 9.226 / 2.905 | 0.520 |
| Gaussian, direct / separable | 15×15 | 30.629 / 3.447 | 15.882 / 3.019 | 0.896 |
| Signed dense convolution | 15×15 | 27.239 | 18.654 | 17.459 |
| Median | 15×15 | 18.960 | 17.266 | 187.633 |

All ndfiltrs columns use four threads. These are medians of seven calls
after a parity check and warmup, with alternating timing order. Public-call
validation, snapshots, output allocation, and NumPy padding are included.
Gaussian calls use `sigma=1` and the same explicit support; convolution uses
the same signed weights. All cases use finite, contiguous inputs and `reflect`
boundaries. Median and extrema outputs agree exactly. Arithmetic comparisons
use `rtol=atol=64 * finfo(dtype).eps * max(abs(input))` on these unit-scale inputs;
the reports record maximum absolute errors.

The full 72-case matrix covers float32/float64, 1-D/2-D/3-D, and two window
widths per shape. Including the additional separable paths, **96 before/after
outputs match bit-for-bit at each thread setting**. This is measured parity,
not proof for every possible input; independent tests also cover cancellation,
overflow, subnormals, missing support, infinities, and boundary geometry.

The local optimized one-thread and four-thread reports recorded both extension
hashes, environment details, errors, and raw times.
Across the twelve cases for each operation, median before/after speedups were
1.45× for direct means, 1.81× for direct Gaussians, and 1.49× for convolution
at four threads; at one thread they were 1.56×, 1.50×, and 1.24×.
Separable mean/Gaussian median gains were about 1.06–1.08×. Some individual
separable cases were slower; no universal speedup is claimed. Median/extrema
timing differences are small or variable and are not claimed as new speedups.

Choosing the faster measured ndfiltrs path per case beats SciPy in 18/72
four-thread cases and 8/72 one-thread cases. This is a retrospective comparison,
not an automatic dispatch policy. Older unpaired runs used a different NumPy
version, so use the paired measurements for speedup claims.
NumPy sliding-window reductions are also measured; they are not necessarily
the fastest algorithms available using NumPy. The mean baseline accumulates
in float64, including for float32 input.

These cases do not measure NaN omission, strided arrays, every shape, or IRAF.
Do not extrapolate median speedups to boxcars or Gaussian smoothing. SciPy's
boxcar algorithm avoids summing every window again; `ndfiltrs`' direct method sums
each window or each axis support. Dense convolution also loses the work saving
available to a separable Gaussian implementation.

Reproduction commands are in the [benchmark guide](../benchmarks/README.md).

Masked-pixel repair has a separate [comparison with astroimred and a SciPy
reference](fixpix.md#original-linear-measurements-before-the-sparse-path-update),
plus [updated sparse and median measurements](fixpix.md#sparse-path-and-median-measurements).
Raw samples distinguish independent copies from no-op input reuse.

## Astro-SCRAPPY: exact versus approximate median

Astro-SCRAPPY 1.3.0's native median filters accept contiguous 2-D float32 arrays.
The benchmark checks exact interiors against NumPy's full-window median,
bit-for-bit, before timing. Full calls are timed: Astro-SCRAPPY **copies border
pixels unchanged**, whereas ndfiltrs filters them. These are therefore not
identical full-image operations.

| 256×256 float32, exact median | ndfiltrs, one thread (ms) | ndfiltrs, four threads (ms) | Astro-SCRAPPY, requested one thread (ms) |
| --- | ---: | ---: | ---: |
| 3×3 | 5.933 | 2.176 | 0.055 |
| 5×5 | 13.390 | 4.223 | 18.943 |
| 7×7 | 18.425 | 5.106 | 40.091 |

Astro-SCRAPPY is dramatically faster for 3×3; ndfiltrs wins the measured exact
5×5 and 7×7 cases, even with one thread. Its
[C implementation](https://github.com/astropy/astroscrappy/blob/v1.3.0/astroscrappy/utils/medutils.c)
uses fixed comparator networks for 3×3/5×5 and quickselect for 7×7. Its
separable option instead takes successive row and column medians: that is
**not** the exact median of a rectangular neighborhood.

For the 256×256 noise input, the approximation differed at 43.0%, 70.9%, and
80.9% of valid interior pixels for 3×3, 5×5, and 7×7 respectively. The reports
retain those fast timings and errors separately; they are not exact speedups.

The local one-thread and four-thread reports recorded
seven raw samples, hashes, versions, and the 128×128 cases as well.
Actual Astro-SCRAPPY OpenMP team size is unknown. The installed macOS extension
has no dynamic OpenMP dependency or imported OpenMP symbols; setting
`OMP_NUM_THREADS=4` alone does not demonstrate four-thread execution.

### What changed, and what remains

The transferable improvement is cheaper traversal and less temporary storage.
Convolution now computes interior offsets once; direct means and weighted
passes accumulate while reading taps. Exceptional arithmetic replays the
original scaled fallback. No approximate median, hardcoded window-size
dispatch, or precision-reducing arithmetic was added.

The [explicit methods above](#sortnet-and-boxcar) address selection
vectorization and exact removable sums. They improve specific workloads, not
every case, and remain separate from the public defaults. A naïve floating-point
subtract/add recurrence would retain cancellation error after an extreme
sample leaves the window; the rolling candidate instead keeps exact sums.

## Generic N-D versus specialized 2-D (historical)

The fresh Criterion run used macOS 26.6 arm64 and the release profile. Its
full named matrix passed the embedded bit-parity preflight before timing. The
matrix covers both dtypes, all five boundary modes, plain and omission
policies, multiple shapes and windows, and NaN/infinity omission inputs.

| Case | Generic N-D serial | Specialized 2-D candidate |
|---|---:|---:|
| f32 32×24, 3×5, constant/mirror, omit | 78.527 µs [78.353, 78.755] | 65.448 µs [64.595, 66.524] |
| f64 256×256, 5×5, reflect | 9.1969 ms [9.1616, 9.2398] | 7.0792 ms [7.0213, 7.1473] |
| f64 18×11, 7×3, nearest/wrap, omit | 30.343 µs [29.896, 31.024] | 24.069 µs [24.020, 24.119] |
| f64 9×13, 3×1, mirror/wrap, omit, ignore-inf | 2.8860 µs [2.8749, 2.9020] | 2.3999 µs [2.3894, 2.4132] |

For the representative f64 256×256 case, the candidate was about 23% faster:
`(9.1969 - 7.0792) / 9.1969`. This is evidence that generic coordinate
traversal can matter for small 2-D windows, but it is insufficient to justify a
production 2-D fast path. The candidate is compiled only with the `benchmark`
Cargo feature and is excluded from wheels.

The matrix verifies bit parity across both dtypes, all boundary modes,
anisotropic/singleton windows, and invalid-value policies before timing.

## Public Python call

The fresh public benchmark used macOS 26.6 arm64, Python 3.13.11, four Rayon
threads, a 256×256 input, a 5×5 window, and seven measured repetitions after
one warmup:

| Case | Median | Range |
|---|---:|---:|
| float32 plain | 6.374083 ms | 5.464625–7.793750 ms |
| float32 NaN-aware | 8.065417 ms | 7.609500–8.522333 ms |
| float64 plain | 6.831709 ms | 6.196500–7.227959 ms |
| float64 NaN-aware | 7.181916 ms | 5.573167–7.758250 ms |

These timings include the owned native input snapshot added before GIL
detachment. A fresh-process high-water RSS sample for one float64 512×512,
7×7 call was 39,387,136 bytes. It includes the interpreter, imports, input,
output, and native libraries, so it is not an isolated algorithmic-memory
measure.

## Limits

These are single-machine measurements. The Criterion candidate remains
benchmark-only, and the results do not establish a production fast path or
untested platform performance.
