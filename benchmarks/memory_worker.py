"""Fresh-process sampled peak-RSS benchmark."""

from __future__ import annotations

import json
import platform
import resource

import ndfiltrs as nf
import numpy as np


def main() -> None:
    """Run one filter and report the process high-water RSS estimate."""
    values = np.arange(512 * 512, dtype=np.float64).reshape(512, 512)
    nf.median_filter(values, 7)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    units = "bytes" if platform.system() == "Darwin" else "KiB"
    print(json.dumps({"ru_maxrss": usage.ru_maxrss, "units": units}))


if __name__ == "__main__":
    main()
