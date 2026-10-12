# Benchmarks

`sortnet` is faster than SciPy for most of the measured median filters, but
Astro-SCRAPPY is much faster for 3×3 medians. Exact `boxcar` means are slower
than SciPy's floating-point uniform filter. Neither method is a universal
speed win.

## Median filters

Times below are median wall times in milliseconds, rounded to two significant
figures. Inputs are finite 256×256 arrays, with one ndfiltrs worker and
`mode="reflect"`. Each measurement includes the complete Python call:
validation, input copy, allocation, and filtering.

| Input and window | ndfiltrs `direct` (ms) | ndfiltrs `sortnet` (ms) | SciPy (ms) | Astro-SCRAPPY (ms) |
| --- | ---: | ---: | ---: | ---: |
| float32, 3×3 | 4.1 | 0.58 | 5.2 | 0.042 |
| float32, 5×5 | 8.8 | 1.7 | 16 | 12 |
| float32, 7×7 | 12 | 4.2 | 28 | 25 |
| float64, 31×31 | 150 | 290 | 470 | — |

`sortnet` compares groups of samples in parallel using an exact selection
network. With one worker, it beats SciPy in 15 of the 18 median cases measured
across 1-D, 2-D, and 3-D inputs. It is not always the best ndfiltrs method.
Direct selection is faster for the 31×31 float64 case above.
The default is `method="direct"`.
ndfiltrs does not choose a method based on these timings.

Astro-SCRAPPY is about 14× faster than `sortnet` for the 3×3 case.
Its exact median functions accept float32, so the table has no float64
comparison. It also copies border pixels, whereas ndfiltrs filters them.
The benchmark checks bitwise equality on the matching interior.
The full-image operations therefore differ.

These exact-median comparisons exclude Astro-SCRAPPY's approximate row/column median.

## Boxcar means

The same run measured rectangular mean filters on finite float64 arrays:

| Window | ndfiltrs `direct` (ms) | ndfiltrs `separable` (ms) | ndfiltrs `boxcar` (ms) | SciPy (ms) |
| --- | ---: | ---: | ---: | ---: |
| 15×15 | 16 | 6.9 | 3.2 | 0.33 |
| 31×31 | 91 | 14 | 3.5 | 0.39 |
| 63×63 | 540 | 29 | 3.6 | 0.37 |

`boxcar` reuses sums as the window moves, so large windows cost much less than
repeatedly visiting every sample. It is about 8× faster than `separable` in
the 63×63 case, but about 10× slower than SciPy. SciPy is faster in all 12
measured boxcar cases.

### Why is boxcar slower than SciPy?

Both algorithms use running sums. Neither uses an FFT. Their arithmetic and
memory costs differ:

- [SciPy's uniform filter](https://github.com/scipy/scipy/blob/v1.18.1/scipy/ndimage/src/ni_filters.c#L311)
  keeps a floating-point sum along each line. Moving the window takes one
  subtraction, one addition, and a division for the output.
- [ndfiltrs boxcar](../src/boxcar.rs) stores each sum exactly in multiple
  integer words. Window updates propagate carries and maintain separate
  counts for finite values, NaNs, and positive and negative infinity.
- Intermediate state occupies two full-image sum buffers and two count buffers.
  After processing all axes, ndfiltrs divides each exact sum and rounds once
  to the output dtype.

That extra work prevents cancellation drift and keeps invalid values local
to their windows. The last bits can differ from both SciPy's running sum and
ndfiltrs' compensated `direct` mean. The boxcar tests check results against
exact rational arithmetic, not just agreement with another library.

The code does more arithmetic and moves more data than SciPy. The timings
do not isolate how much each adds to runtime.
They do not prove that the current gap is unavoidable.
The implementation still needs work to compete with
SciPy. With four ndfiltrs workers, the 63×63 case takes 1.2 ms, compared with
0.37 ms for SciPy in that run.

## Measurement setup

The measurements date from 12 October 2026. The environment used macOS 26.6
ARM64, Python 3.13.11, NumPy 2.5.2, SciPy 1.18.1, and Astro-SCRAPPY 1.3.0.
The ndfiltrs extension used a release build with Rust 1.96.0.

The full matrix has 30 cases covering float32 and float64 in 1-D, 2-D, and
3-D. Each case includes seven repetitions.
The benchmark rotates timing order between implementations.
No builds, tests, or profilers ran during timing.
`RAYON_NUM_THREADS` controls ndfiltrs workers. OpenMP thread counts were
requested for Astro-SCRAPPY, but its actual team size is unknown.

The benchmark checks finite-input means within numerical tolerances before timing.
Separate tests cover exact rounding, NaNs, infinities, and extreme values.
These timings do not establish performance for every layout, boundary mode,
or invalid-value policy. No matched IRAF speed measurement is available.

See [pixel-repair timings](fixpix.md) for `fixpix`, and the
[benchmark instructions](../benchmarks/README.md) to reproduce the runs.
Generated JSON reports stay local. They contain raw samples, validation
errors, source hashes, and environment details.
