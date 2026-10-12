# Gold image v1

`scene.npz` is the canonical numeric fixture, not a screenshot. Its 256 x 320
main image is a synthetic astronomical scene, with 14 arrays and a SHA-256
manifest.

Keep v1 stable. Create v2 for changes to scientific content.
Do not regenerate this file during tests.

The original data use the repository's MIT license.
They contain no downloaded observation or personal data.

## Contents

| Arrays | Purpose |
|---|---|
| `background`, `model` | Linear background and noiseless signal with six Gaussian point sources, an extended source, a sharp step, and sinusoidal texture |
| `noise_sigma`, `noise`, `clean` | Signal-dependent Gaussian noise and a fixed uncorrupted noisy realization |
| `observed_sparse`, `mask_sparse`, `defect_kind_sparse` | 81 bad pixels (0.0989%), including source centers, edges, and the step |
| `observed_stress`, `mask_stress`, `defect_kind_stress` | 618 bad pixels (0.7544%), including a full column, crossing runs, a block, and border clusters |
| `control_flat`, `control_ramp`, `control_mask` | Small exactly representable controls with one interior bad-pixel location |

The masks are boolean (`True` is bad). Defect codes are 0=valid, 1=hot
(`clean+500`), 2=dead (`-100`), 3=NaN, 4=+infinity, 5=-infinity. All invalid
pixels in the observed images are marked. Unmasked values equal `clean`
bit-for-bit. `model`, `clean`, and `noise` are finite little-endian float64.
Cast inputs explicitly to float32 for dtype comparisons.

Coordinates are zero-based `(y, x)`, with y increasing downward.
The model samples pixel centers. Gaussian amplitudes are peak sample units,
not integrated source fluxes. The noise model is an illustrative Gaussian
approximation, not a calibrated detector model.
`manifest.json` records all parameters and seeds.

## Use

From the repository root, with NumPy and a built ndfiltrs in the chosen environment:

```python
import numpy as np
import ndfiltrs as nf

with np.load("tests/data/gold_v1/scene.npz", allow_pickle=False) as scene:
    image = scene["observed_sparse"]
    bad = scene["mask_sparse"]
    clean = scene["clean"]

repaired = nf.fixpix(
    image, bad, method="median", size=3, mode="constant", min_count=3, ignore_inf=True
)
```

`model` is the noiseless truth and `clean` is the original noisy realization.
Neither is an exact expected result for interpolation or filtering. Especially
near source peaks, steps, and large missing regions, legitimate repair methods
have bias or insufficient support.

The regression tests use independent algorithmic references and exact flat/ramp
controls. References include exact rational means and variances, high-precision
standard deviations, and total-order extrema. Tests do not require perfect
reconstruction.

```bash
uv run --no-sync pytest -q tests/test_gold_image.py tests/test_statistics.py
uv run --no-sync python scripts/evaluate_gold_image.py --output /tmp/gold-report.json
```

For the shared environment, activate it first. Add `--active` to the commands.
Never synchronize that environment.
The report gives resolved/unresolved counts, bias and RMSE against both truth
layers, plus public-call timings. Error metrics are conditional on resolved
pixels: compare coverage as well. These are diagnostics, without empirical
pass/fail thresholds. A saved report is evidence for that build, not an expected
output that future implementations must reproduce.

## Regeneration and preview

```bash
uv run --no-sync python scripts/generate_gold_image.py --output /tmp/gold-rebuild
uv run --no-sync python scripts/preview_gold_image.py
```

Generation needs only NumPy. Preview additionally needs Pillow. The PNG is a
contact sheet with a common asinh display stretch. The NPZ retains all original
values. [`docs/images/gold-v1.png`](../../../docs/images/gold-v1.png) is only for
inspection, not a numerical test input.

The generator refuses to overwrite existing scene files. A fixed PCG64 seed
makes intent reproducible, but floating transcendental operations and normal
sampling can vary with platform or NumPy version. The stored fixture is the
cross-platform authority. `manifest.json` records per-array and archive hashes,
generator identity, and the generation environment. Reconstruction comparisons
must not silently regenerate expected arrays from ndfiltrs itself.

This fixture is 2-D and deliberately synthetic. Existing small-array tests
remain responsible for other dimensions, layouts, dtype extremes, arithmetic
contracts, and boundary modes. Representative real observations are still
needed before making scientific performance claims.
