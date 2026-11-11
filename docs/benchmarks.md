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

## Large-window medians

`histogram` was faster than `direct` and SciPy at every measured window with
one worker. It was faster than `sortnet` from 7×7 upward with one worker, and
from 7×7 (float64) or 11×11 (float32) with 14 workers. These timings are provisional: sibling builds and tests
loaded the machine during the run.

Times are median wall times in milliseconds of complete Python calls on
finite inputs with `mode="reflect"`, rounded to two significant figures.
"—" means the method was skipped: its predicted call time exceeded 30 s.

4096×4096, one worker:

| Window | `histogram` float32 | `histogram` float64 | `direct` float32 | `sortnet` float32 | SciPy float32 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3×3 | 770 | 810 | 1,000 | 120 | 1,300 |
| 7×7 | 1,000 | 970 | 3,100 | 1,100 | 8,800 |
| 11×11 | 1,200 | 1,300 | 5,500 | 3,300 | 20,000 |
| 21×21 | 1,900 | 1,700 | 14,000 | 15,000 | — |
| 31×31 | 2,900 | 2,200 | — | — | — |
| 51×51 | 4,000 | 3,800 | — | — | — |
| 101×101 | 5,400 | 4,700 | — | — | — |
| 201×201 | 8,700 | 8,400 | — | — | — |

4096×4096, 14 workers (SciPy uses one thread):

| Window | `histogram` float32 | `histogram` float64 | `direct` float32 | `sortnet` float32 | SciPy float32 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3×3 | 210 | 140 | 170 | 19 | 1,300 |
| 7×7 | 210 | 150 | 410 | 120 | 8,100 |
| 11×11 | 260 | 170 | 700 | 400 | 18,000 |
| 21×21 | 340 | 250 | 2,000 | 2,200 | — |
| 31×31 | 450 | 300 | 4,000 | 7,100 | — |
| 51×51 | 750 | 450 | 15,000 | 43,000 | — |
| 101×101 | 1,100 | 840 | — | — | — |
| 201×201 | 1,600 | 1,500 | — | — | — |

1024×1024, one worker:

| Window | `histogram` float32 | `histogram` float64 | `direct` float32 | `sortnet` float32 | SciPy float32 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3×3 | 49 | 52 | 65 | 7.1 | 81 |
| 7×7 | 59 | 62 | 190 | 60 | 500 |
| 11×11 | 73 | 73 | 340 | 200 | 1,100 |
| 31×31 | 180 | 140 | 1,900 | 3,000 | 7,600 |
| 51×51 | 240 | 220* | 5,000 | 12,000 | 22,000 |
| 101×101 | 360 | 280 | 21,000 | — | — |
| 201×201 | 560 | 520 | — | — | — |

\* The matrix run gave 390 ms for this cell. Nine later calls at load
average 3.9 gave a median of 220 ms. Nine calls at 3×3 in that recheck gave
47 ms for `histogram` and 63 ms for `direct` (float32), as in the table.

Smallest window from which `histogram` was faster at every larger measured
window:

| Comparison | 1 worker | 14 workers |
| --- | --- | --- |
| vs `direct` | 3×3 in all four cases | 3×3, except 5×5 for 4096² float32 |
| vs `sortnet` | 7×7 in all four cases | 7×7 for float64, 11×11 for float32 |
| vs SciPy | 3×3 in all four cases | 3×3 in all four cases |

With one worker on 1024² float32, `histogram` and `sortnet` took 59 and
60 ms at 7×7. That difference is smaller than the run-to-run variation under
this load. ndfiltrs does not choose a method from these timings. The default
stays `method="direct"`.

From 21×21 to 201×201 the `histogram` time grew by 4.5× (float32, one
worker), while the window area grew by 92×. For a `k × k` window each output
updates `2k` histogram entries, plus a share of one sort per block (see
[filtering semantics](semantics.md)). With 14 workers, 4096² took 5.5×
(float32) and 5.4× (float64) less time than one worker at 201×201.
The load average was 9 to 49 during the 14-worker run, so this
scaling is a lower bound for an idle machine, not a measurement of one.

Before timing, every case checked that `direct` and `sortnet` equal
`histogram` bitwise, and that SciPy equals it on odd windows. Where `direct`
was skipped, a 256×256 input checked `histogram` against `direct` at the
same window. All checks passed. The run used 12 window sizes from 3 to 201,
both dtypes, and three timed calls per method. Calls longer than 10 s kept
one sample. It used the software versions in
[measurement setup](#measurement-setup), measured on 12 November 2026. Two
conditions differ from that setup. Each method's calls ran back to back,
without rotating the order between methods. Other builds and tests ran
during timing: the one-minute load average was 3.8 to 22 during the
one-worker run.

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
