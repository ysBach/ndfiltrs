# Benchmarks

Git ignores generated JSON reports, which remain local. Keep them outside
the source distribution. The commands below generate fresh reports.

## Compare sortnet and boxcar methods

`benchmark_algorithms` evaluates `median_filter(method="sortnet")` and
`mean_filter(method="boxcar")`. The default method is `direct`.
Each timed public call includes normal argument validation, the input
snapshot, allocation, and native execution.

```bash
uv run --no-sync maturin develop --release --locked
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 uv run --no-sync python -m benchmarks.benchmark_algorithms --repeats 7 --output /tmp/algorithms-threads1.json
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=4 uv run --no-sync python -m benchmarks.benchmark_algorithms --repeats 7 --output /tmp/algorithms-threads4.json
```

Run these commands sequentially, with other CPU-heavy work stopped.
`--quick --preflight-only` checks a smaller matrix without timing.
Dependencies are the project environment plus SciPy and Astro-SCRAPPY.

The 30-case matrix covers float32/float64 and 1-D/2-D/3-D. It includes 3/5/7
small medians, 15/31 large medians, and boxcar widths up to 127. Median results
must agree bitwise over the complete image. The script checks Astro-SCRAPPY
only on the matching interior. Boxcar timings use a finite-input tolerance check.
Separate tests use exact rational sums and direct target-dtype rounding.

Reports record raw samples, rotating call order, errors, thread settings, and
source/extension hashes. Reports retain large-window regressions.

The network trades temporary tap buffers for vectorized comparisons. Rolling
means trade memory for exact removable sums, with no intermediate rounding.
Neither method uses a hardcoded window-size dispatch or approximate median.
See [results and limits](../docs/benchmarks.md).

### Sample CPU use

1. Run `python -m benchmarks.profile_filter --method sortnet --size 3 --dtype float32 --seconds 30` under the same environment.
2. On macOS, use its logged PID with `sample PID 8 1 -file /tmp/filter.sample.txt`.

Use `--method boxcar --size 63` to sample exact means. Do not collect comparative
timings while sampling, compiling, or running tests.

## Compare Astro-SCRAPPY exact and approximate medians

Install only into the project environment if needed:

```bash
UV_CACHE_DIR=/tmp/ndfiltrs-uv-cache uv pip install --python .venv/bin/python astroscrappy
```

Check parity without collecting timings, including during build coordination:

```bash
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 UV_CACHE_DIR=/tmp/ndfiltrs-uv-cache uv run --no-sync python -m benchmarks.benchmark_astroscrappy --quick --preflight-only --output /tmp/astroscrappy-parity.json
```

Run these commands after the release build is ready, with other CPU-heavy work stopped:

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE RAYON_NUM_THREADS=1 UV_CACHE_DIR=/tmp/ndfiltrs-uv-cache uv run --no-sync python -m benchmarks.benchmark_astroscrappy --output /tmp/astroscrappy-threads1.json
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE RAYON_NUM_THREADS=4 UV_CACHE_DIR=/tmp/ndfiltrs-uv-cache uv run --no-sync python -m benchmarks.benchmark_astroscrappy --output /tmp/astroscrappy-threads4.json
```

The bounded default covers 128 and 256 square images with 3/5/7 windows and
three raw samples per runner. `--quick` uses one 32 square image.

Inputs are finite C-contiguous **float32 only**.
Astro-SCRAPPY's native wrappers do not provide a matching float64 contract.
Timings do not hide an input downcast.

Exact outputs must match NumPy's true median bitwise on the valid interior.
Astro-SCRAPPY copies its radius-wide border, while ndfiltrs filters it using
reflect boundaries. Timings include full calls, so border work differs.
This is not full-image equivalence or an isolated interior-kernel measurement.

`approximate_records_not_equivalent` reports Astro-SCRAPPY's row-then-column
separable median separately, with disagreement counts, maximum error, and
RMSE against the true median. The script checks it against an independent
two-pass definition. It never accepts this method as an equivalent exact median
or includes it in an exact-speedup ratio. The script introduces no accuracy thresholds.

Reports include package versions, extension/source hashes, actual Rayon pool
size, and requested OpenMP environment. Actual OpenMP team size/build support
is unknown. An environment variable alone does not establish parallel execution.
Use fresh processes for each thread setting. The reviewed implementation is
[Astro-SCRAPPY 1.3.0 C](https://github.com/astropy/astroscrappy/blob/v1.3.0/astroscrappy/utils/medutils.c)
and its [Cython wrappers](https://github.com/astropy/astroscrappy/blob/v1.3.0/astroscrappy/utils/median_utils.pyx).

## Compare matched SciPy and NumPy filters

Run the finite-input comparison with a release build, fixed thread counts,
and no concurrent CPU-heavy jobs:

```bash
RAYON_NUM_THREADS=1 uv run --no-sync python -m benchmarks.benchmark_scipy --output /tmp/scipy-threads1.json
RAYON_NUM_THREADS=4 uv run --no-sync python -m benchmarks.benchmark_scipy --output /tmp/scipy-threads4.json
```

The script checks matching boundaries, shapes, dtypes, and numerical outputs
before timing. It records seven raw samples per implementation by default
and alternates call order. It covers 72 finite contiguous-array cases across
1-D/2-D/3-D and both floating dtypes. Timings include NumPy sliding-window padding.
This does not establish performance for NaN omission, other layouts, or IRAF.
See the [measured results](../docs/benchmarks.md).

Add `--baseline /path/to/previous/ndfiltrs` to compare a trusted prior installed
package in the same interpreter and NumPy environment. The script requires
bit-for-bit before/after agreement for each direct and separable path before
timing. It records both versions' extension hashes. Keep the original wheel
outside the source checkout so rebuilding the current extension cannot replace it.

The NumPy mean baseline accumulates in float64 for both input dtypes. It is not
NumPy's default float32 accumulator or an optimized rolling-sum implementation.

## Measure masked-pixel repair

Benchmark masked-pixel repair separately:

```bash
uv run --no-sync python -m benchmarks.benchmark_fixpix --repeats 7
```

This requires an environment already containing `astroimred`, NumPy, SciPy,
and the built `ndfiltrs` extension. Add `--astroimred-src ../astroimred/src`
to compare a source checkout. The script checks parity, records raw timings
and source hashes, and includes sparse overhead cases. Reports label the
independent SciPy reference separately. See [repair results](../docs/fixpix.md).

You can also measure sparse linear and median methods without astroimred:

```bash
uv run --no-sync python -m benchmarks.benchmark_fixpix_sparse --repeats 7 --loops 5
```

Use `--baseline /path/to/previous/ndfiltrs` to compare an installed prior release
(the directory containing its `__init__.py` and extension). Cases cover zero,
one, 0.01%, and 0.1% masked pixels on two image sizes and both floating dtypes.
The `copy=False` no-op result aliases input. Report it separately from copied
output. The smaller image includes a full-image `nanmedian_filter` composition
as an equivalent median baseline. Reports record source hashes and raw samples.

## Measure median filters and memory

Run these commands from the repository root to prepare the project's `.venv`
and measure median filters:

```bash
uv sync --frozen --extra dev --extra bench
uv run --no-sync maturin develop --release --locked
uv run --no-sync python -m benchmarks.benchmark_filter
uv run --no-sync python -m benchmarks.memory_worker
cargo bench --locked --features benchmark --bench filter2d_candidate
```

For an already provisioned shared environment, follow the
[shared-environment instructions](../README.md#source-build) and use
`uv run --active --no-sync` without synchronization.

The Rust candidate is benchmark-only and cannot enter Python wheels because
the maturin feature list does not enable `benchmark`. Check byte parity tests
before accepting timings. Public timings include Python dispatch, the owned
native input snapshot, and native output construction. `ru_maxrss` is a process
high-water estimate, not an isolated algorithmic allocation measurement.

## Evaluate the frozen scientific scene

Use `scripts/evaluate_gold_image.py --output /tmp/gold-report.json` in the same
Python environment as the built package. It evaluates both repair methods on
[gold v1](../tests/data/gold_v1/README.md). It reports resolved coverage and
conditional bias/RMSE against the noisy and noiseless truth layers.

Timing and error metrics are diagnostics, not production thresholds.
The public-call timings exclude image generation and format conversion.

The generated gold-v1 report records both truth-layer errors and unresolved
counts. It is not a pass/fail reference for future algorithms.
