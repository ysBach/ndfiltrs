"""Shared deterministic benchmark inputs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
from numpy.typing import NDArray


def make_input(
    shape: tuple[int, ...], dtype: type[np.float32] | type[np.float64]
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Create one deterministic benchmark array."""
    random = np.random.default_rng(319)
    return random.normal(size=shape).astype(dtype)


def load_baseline(path: Path) -> ModuleType:
    """Load a prior installed package under a separate module name.

    Parameters
    ----------
    path : pathlib.Path
        Trusted package directory containing `__init__.py` and the extension.

    Returns
    -------
    types.ModuleType
        Prior package, using the current interpreter and NumPy installation.

    Raises
    ------
    ImportError
        The directory does not contain an importable package.
    """
    source = path.resolve() / "__init__.py"
    spec = importlib.util.spec_from_file_location("_ndfiltrs_baseline", source)
    if not source.is_file() or spec is None or spec.loader is None:
        raise ImportError("baseline must be an installed ndfiltrs package directory")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
