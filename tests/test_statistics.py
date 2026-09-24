"""Independent N-D extrema, variance, and standard-deviation contracts."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import cast

import ndfiltrs as nf
import numpy as np
import pytest

from tests._reference import MODES
from tests._statistics_reference import (
    FloatArray,
    Statistic,
    statistics_filter_reference,
)

STATISTICS = ("min", "max", "var", "std")
FUNCTIONS = tuple(
    f"{prefix}{kind}_filter" for prefix in ("", "nan") for kind in STATISTICS
)
MOMENTS = ("var_filter", "std_filter", "nanvar_filter", "nanstd_filter")
NAN_FUNCTIONS = tuple(name for name in FUNCTIONS if name.startswith("nan"))
DTYPES = (np.float32, np.float64)
GOLD = Path(__file__).parent / "data" / "gold_v1" / "scene.npz"


@pytest.fixture(params=FUNCTIONS)
def function(request: pytest.FixtureRequest) -> Callable[..., FloatArray]:
    """Resolve public functions at test execution, without skipping missing APIs."""
    return getattr(nf, request.param)


def _assert_statistics(
    actual: FloatArray, expected: FloatArray, statistic: Statistic, volume: int
) -> None:
    """Check classification, canonical NaNs, extrema bits, and rounding bounds.

    Parameters
    ----------
    actual, expected : numpy.ndarray
        Native output and the independent reference.
    statistic : {'min', 'max', 'var', 'std'}
        Statistic being checked.
    volume : int
        Maximum number of retained positions in one window.
    """
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    unsigned = np.uint32 if actual.dtype == np.float32 else np.uint64
    canonical = 0x7FC00000 if actual.dtype == np.float32 else 0x7FF8000000000000
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(actual[np.isnan(actual)].view(unsigned), canonical)
    if statistic in ("min", "max"):
        np.testing.assert_array_equal(actual.view(unsigned), expected.view(unsigned))
        return
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    assert not np.signbit(actual[~np.isnan(actual)]).any()
    # gamma_k = k*u/(1-k*u) is the standard accumulated rounding envelope.
    # Budget two centered passes, each with six rounded operations per sample
    # (subtraction/squaring and compensated accumulation), plus eight scalar
    # operations for normalization/scaling/root. This accepts stable centered
    # arithmetic, not cancellation from subtracting raw second moments.
    # Two output roundings account for the independently rounded reference
    # and actual result. There is no absolute floor: exact zeros stay zero.
    operations = 12 * volume + 8
    unit64 = np.finfo(np.float64).eps / 2
    gamma = operations * unit64 / (1 - operations * unit64)
    rtol = gamma + np.finfo(actual.dtype).eps
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0)


def test_reference_has_literal_variance_and_signed_zero_answers() -> None:
    """Anchor the oracle to hand-derived answers, including out-of-range variance."""
    for kind, answer in (("min", -1.0), ("max", 1.0), ("var", 1.0), ("std", 1.0)):
        result = statistics_filter_reference(
            np.array([-1.0, 0.0, 1.0]), (3,), ("wrap",), statistic=kind, ddof=1
        )
        np.testing.assert_array_equal(result, answer)
    for kind, negative in (("min", True), ("max", False)):
        result = statistics_filter_reference(
            np.array([-0.0, 0.0, -0.0]), (3,), ("wrap",), statistic=kind
        )
        np.testing.assert_array_equal(np.signbit(result), negative)
    for scale in (np.ldexp(1.0, 600), np.ldexp(1.0, -600)):
        result = statistics_filter_reference(
            np.array([-scale, 0, scale]), (3,), ("wrap",), statistic="std", ddof=1
        )
        np.testing.assert_array_equal(result, scale)


def test_public_signature_defaults_and_keyword_only_options(
    function: Callable[..., FloatArray],
) -> None:
    signature = inspect.signature(function)
    name = function.__name__
    names = ["array", "size", "footprint", "origin", "mode", "cval"]
    if "var" in name or "std" in name:
        names.append("ddof")
        assert signature.parameters["ddof"].default == 0
    if name.startswith("nan"):
        names.extend(("ignore_inf", "min_count"))
        assert signature.parameters["ignore_inf"].default is False
        assert signature.parameters["min_count"].default == 1
    assert set(signature.parameters) == set(names)
    assert signature.parameters["size"].default is None
    assert signature.parameters["mode"].default == "reflect"
    assert signature.parameters["cval"].default == 0.0
    for name in names[2:]:
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(4,), (3, 4), (2, 1, 3), (2, 1, 2, 3)])
@pytest.mark.parametrize("mode", MODES)
def test_all_statistics_match_independent_nd_windows(
    function: Callable[..., FloatArray],
    dtype: type[np.floating],
    shape: tuple[int, ...],
    mode: str,
) -> None:
    # Dyadic samples keep input conversion exact; irregular order makes min,
    # max, variance, and each boundary distinguishable on these small arrays.
    array = ((np.arange(np.prod(shape)).reshape(shape) * 7) % 23 - 11).astype(dtype) / 4
    size = (5,) + (3,) * (len(shape) - 1)
    statistic = cast(
        Statistic, function.__name__.removeprefix("nan").removesuffix("_filter")
    )
    options = {"ddof": 1} if statistic in ("var", "std") else {}
    expected = statistics_filter_reference(
        array, size, (mode,) * array.ndim, statistic=statistic, cval=1.25, **options
    )
    actual = function(array, size, mode=mode, cval=1.25, **options)
    _assert_statistics(actual, expected, statistic, int(np.prod(size)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_mixed_modes_axis_identity_and_numpy_integer_options(
    function: Callable[..., FloatArray],
    dtype: type[np.floating],
) -> None:
    array = np.arange(12, dtype=dtype).reshape(2, 1, 2, 3)
    size = (3, 1, 5, 3)
    mode = ("constant", "nearest", "mirror", "wrap")
    statistic = cast(
        Statistic, function.__name__.removeprefix("nan").removesuffix("_filter")
    )
    options = {"ddof": np.int64(1)} if statistic in ("var", "std") else {}
    if function.__name__.startswith("nan"):
        options.update(ignore_inf=np.bool_(True), min_count=np.int64(2))
    actual = function(
        array, list(map(np.int64, size)), mode=list(mode), cval=-2, **options
    )
    expected = statistics_filter_reference(
        array,
        size,
        mode,
        statistic=statistic,
        cval=-2,
        ddof=1 if statistic in ("var", "std") else 0,
    )
    _assert_statistics(actual, expected, statistic, int(np.prod(size)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_layout_ownership_and_identity_windows(
    function: Callable[..., FloatArray],
    dtype: type[np.floating],
) -> None:
    source = np.arange(24, dtype=dtype).reshape(4, 6)
    statistic = cast(
        Statistic, function.__name__.removeprefix("nan").removesuffix("_filter")
    )
    for array in (
        source,
        source.T,
        source[::-1, ::2],
        np.asfortranarray(source),
        np.broadcast_to(source[:1, :], source.shape),
    ):
        array.flags.writeable = False
        original = array.copy()
        for size in (1, 3):
            actual = function(array, size)
            expected = statistics_filter_reference(
                array, (size, size), ("reflect", "reflect"), statistic=statistic
            )
            _assert_statistics(actual, expected, statistic, size * size)
            assert type(actual) is np.ndarray
            assert actual.flags.owndata and actual.flags.c_contiguous
            assert actual.flags.writeable and actual.flags.aligned
            assert not np.shares_memory(actual, array)
            np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0, 3), (2, 1, 3, 0)])
def test_empty_axes_return_owned_validated_arrays(
    function: Callable[..., FloatArray],
    shape: tuple[int, ...],
) -> None:
    array = np.empty(shape, dtype=np.float32)
    actual = function(array, 5, mode="mirror")
    assert actual.shape == shape and actual.dtype == array.dtype
    assert actual is not array and actual.flags.owndata
    assert actual.flags.c_contiguous and actual.flags.aligned and actual.flags.writeable
    with pytest.raises(ValueError):
        function(array, 0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["min", "max"])
def test_extrema_signed_zero_ties_and_infinities(
    dtype: type[np.floating],
    kind: Statistic,
) -> None:
    for prefix in ("", "nan"):
        function = getattr(nf, f"{prefix}{kind}_filter")
        for values in (
            [-0.0, 0.0, -0.0],
            [0.0, -0.0, 0.0],
            [-0.0] * 3,
            [0.0] * 3,
            [-np.inf, 2, np.inf],
        ):
            array = np.array(values, dtype=dtype)
            expected = statistics_filter_reference(
                array, (3,), ("wrap",), statistic=kind
            )
            _assert_statistics(function(array, 3, mode="wrap"), expected, kind, 3)
        # Padding zeros have the same tie order as real input samples.
        array = np.array([0.0 if kind == "min" else -0.0], dtype=dtype)
        cval = -0.0 if kind == "min" else 0.0
        actual = function(array, mode="constant", cval=cval)
        np.testing.assert_array_equal(np.signbit(actual), kind == "min")
        array = np.array([-0.0, 0.0], dtype=dtype)
        np.testing.assert_array_equal(np.signbit(function(array, 1)), np.signbit(array))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", NAN_FUNCTIONS)
@pytest.mark.parametrize("ignore_inf", [False, True])
def test_nan_policies_counts_and_nonfinite_padding(
    dtype: type[np.floating],
    name: str,
    ignore_inf: bool,
) -> None:
    function = getattr(nf, name)
    statistic = cast(Statistic, name.removeprefix("nan").removesuffix("_filter"))
    array = np.array([[np.nan, 1, np.inf], [-np.inf, 3, 7]], dtype=dtype)
    size, mode = (3, 5), ("constant", "mirror")
    for cval in (np.nan, np.inf, -np.inf, 2.0):
        for min_count in (1, 5, 15):
            options = dict(cval=cval, ignore_inf=ignore_inf, min_count=min_count)
            if statistic in ("var", "std"):
                options["ddof"] = 1
            expected = statistics_filter_reference(
                array, size, mode, statistic=statistic, omit_nan=True, **options
            )
            with warnings.catch_warnings(), np.errstate(all="raise"):
                warnings.simplefilter("error")
                actual = function(array, size, mode=mode, **options)
            _assert_statistics(actual, expected, statistic, 15)


@pytest.mark.parametrize("dtype", DTYPES)
def test_plain_nonfinite_effects_are_local_and_canonical(
    function: Callable[..., FloatArray],
    dtype: type[np.floating],
) -> None:
    statistic = cast(
        Statistic, function.__name__.removeprefix("nan").removesuffix("_filter")
    )
    omit_nan = function.__name__.startswith("nan")
    for value in (np.nan, np.inf, -np.inf):
        array = np.arange(11, dtype=dtype)
        array[5] = value
        for mode, cval in (("reflect", 0.0), ("constant", value)):
            expected = statistics_filter_reference(
                array, (3,), (mode,), statistic=statistic, omit_nan=omit_nan, cval=cval
            )
            actual = function(array, mode=mode, cval=cval)
            _assert_statistics(actual, expected, statistic, 3)
            # Interior windows on both sides must retain their finite result.
            assert np.isfinite(actual[[2, 8]]).all()


@pytest.mark.parametrize("name", NAN_FUNCTIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_empty_support_is_silent_and_boundary_repetitions_count(
    name: str,
    dtype: type[np.floating],
) -> None:
    function = getattr(nf, name)
    statistic = cast(Statistic, name.removeprefix("nan").removesuffix("_filter"))
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        for invalid in (np.nan, np.inf, -np.inf):
            actual = function(np.full(3, invalid, dtype=dtype), ignore_inf=True)
            _assert_statistics(actual, np.full(3, np.nan, dtype=dtype), statistic, 3)
        array = np.array([np.nan], dtype=dtype)
        for min_count in (2, 3):
            actual = function(array, mode="constant", cval=7, min_count=min_count)
            answer = (
                (0 if statistic in ("var", "std") else 7) if min_count == 2 else np.nan
            )
            _assert_statistics(actual, np.array([answer], dtype=dtype), statistic, 3)
        array = np.array([7], dtype=dtype)
        actual = function(array, min_count=3)
        answer = 0 if statistic in ("var", "std") else 7
        _assert_statistics(actual, np.array([answer], dtype=dtype), statistic, 3)


@pytest.mark.parametrize("name", MOMENTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_ddof_uses_retained_count_and_is_independent_of_min_count(
    name: str,
    dtype: type[np.floating],
) -> None:
    function = getattr(nf, name)
    statistic = cast(Statistic, name.removeprefix("nan").removesuffix("_filter"))
    omit_nan = name.startswith("nan")
    array = np.array([1, 3, np.nan] if omit_nan else [1, 2, 3], dtype=dtype)
    for ddof in (0, 1, 2, 3, 4, np.iinfo(np.intp).max):
        expected = statistics_filter_reference(
            array, (3,), ("wrap",), statistic=statistic, omit_nan=omit_nan, ddof=ddof
        )
        with warnings.catch_warnings(), np.errstate(all="raise"):
            warnings.simplefilter("error")
            actual = function(array, mode="wrap", ddof=ddof)
        _assert_statistics(actual, expected, statistic, 3)
    if omit_nan:
        assert np.isnan(function(array, mode="wrap", ddof=0, min_count=3)).all()
    # Repetition counts as distinct samples: n=3 at a singleton reflect edge.
    np.testing.assert_array_equal(function(np.array([9], dtype=dtype), ddof=2), 0)
    assert np.isnan(function(np.array([9], dtype=dtype), ddof=3)).all()


@pytest.mark.parametrize("name", MOMENTS)
@pytest.mark.parametrize(
    "ddof,error",
    [
        (-1, ValueError),
        (True, TypeError),
        (np.bool_(False), TypeError),
        (1.0, TypeError),
        (np.float64(1), TypeError),
        ("1", TypeError),
        (None, TypeError),
    ],
)
def test_invalid_ddof_rejected_even_on_empty_input(
    name: str,
    ddof: object,
    error: type[Exception],
) -> None:
    for array in (np.ones(3), np.empty(0)):
        with pytest.raises(error):
            getattr(nf, name)(array, ddof=ddof)


@pytest.mark.parametrize("name", MOMENTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_large_offset_small_scatter_and_constant_maximum(
    name: str,
    dtype: type[np.floating],
) -> None:
    function = getattr(nf, name)
    statistic = cast(Statistic, name.removeprefix("nan").removesuffix("_filter"))
    # Power-of-two offsets and ULP multiples retain the deviations in the
    # input dtype; rational centering cannot lose scatter to a large offset.
    offset = dtype(np.ldexp(1.0, np.finfo(dtype).nmant))
    for sign in (1, -1):
        array = sign * offset + np.array([-2, -1, 0, 1, 2], dtype=dtype)
        for ddof in (0, 1):
            expected = statistics_filter_reference(
                array, (5,), ("wrap",), statistic=statistic, ddof=ddof
            )
            _assert_statistics(
                function(array, 5, mode="wrap", ddof=ddof), expected, statistic, 5
            )
    for value in (np.finfo(dtype).max, -np.finfo(dtype).max):
        actual = function(np.full(5, value, dtype=dtype), 5, ddof=1)
        np.testing.assert_array_equal(actual, np.zeros(5, dtype=dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("prefix", ["", "nan"])
def test_std_survives_variance_overflow_and_underflow(
    dtype: type[np.floating],
    prefix: str,
) -> None:
    # +/-MAX at n=3, ddof=0 has finite std = MAX*sqrt(2/3), but
    # variance overflows even binary64 for the float64 input.
    maximum = np.finfo(dtype).max
    # The power of two is chosen from the dtype's exponent range: its square
    # lies below the smallest subnormal, while the value itself is normal.
    exponent = np.finfo(dtype).minexp
    small = dtype(np.ldexp(1.0, exponent))
    for array, ddof, overflow in (
        (np.array([-maximum, 0, maximum], dtype=dtype), 0, True),
        (np.array([-small, 0, small], dtype=dtype), 1, False),
    ):
        for statistic in ("var", "std"):
            expected = statistics_filter_reference(
                array, (3,), ("wrap",), statistic=statistic, ddof=ddof
            )
            with warnings.catch_warnings(), np.errstate(all="raise"):
                warnings.simplefilter("error")
                actual = getattr(nf, f"{prefix}{statistic}_filter")(
                    array, mode="wrap", ddof=ddof
                )
            _assert_statistics(actual, expected, statistic, 3)
            if statistic == "std":
                assert np.isfinite(actual).all() and (actual > 0).all()
            elif overflow:
                assert np.isposinf(actual).all()
            else:
                np.testing.assert_array_equal(actual, 0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["var", "std"])
def test_extreme_scatter_after_omission_and_subnormal_standard_deviation(
    dtype: type[np.floating],
    kind: Statistic,
) -> None:
    function = getattr(nf, f"nan{kind}_filter")
    maximum = np.finfo(dtype).max
    for invalid in (np.nan, np.inf, -np.inf):
        array = np.array([-maximum, 0, maximum, invalid, invalid], dtype=dtype)
        options = dict(ignore_inf=True, min_count=3, ddof=0)
        expected = statistics_filter_reference(
            array, (5,), ("wrap",), statistic=kind, omit_nan=True, **options
        )
        _assert_statistics(
            function(array, 5, mode="wrap", **options), expected, kind, 5
        )
        assert np.isnan(
            function(array, 5, mode="wrap", ignore_inf=True, min_count=4)
        ).all()
    # n=3, ddof=1 makes std([-t,0,t]) exactly t, even for the least subnormal.
    tiny = np.nextafter(dtype(0), dtype(1))
    array = np.array([-tiny, 0, tiny], dtype=dtype)
    expected = statistics_filter_reference(
        array, (3,), ("wrap",), statistic=kind, ddof=1
    )
    _assert_statistics(function(array, mode="wrap", ddof=1), expected, kind, 3)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"size": 0}, ValueError),
        ({"size": (3, 0)}, ValueError),
        ({"size": True}, TypeError),
        ({"size": (3,)}, ValueError),
        ({"mode": "bad"}, ValueError),
        ({"mode": ("wrap",)}, ValueError),
        ({"cval": True}, TypeError),
    ],
)
def test_shared_geometry_validation(
    function: Callable[..., FloatArray],
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        function(np.ones((2, 3)), **kwargs)


def test_shared_array_validation(function: Callable[..., FloatArray]) -> None:
    for array in (
        np.ones(3, dtype=int),
        np.ones(3, dtype=np.float16),
        np.ones(3, dtype=np.complex128),
        np.ma.array([1.0], mask=[False]),
    ):
        with pytest.raises(TypeError):
            function(array)
    with pytest.raises(ValueError):
        function(np.array(1.0))
    array = np.ndarray(3, dtype=np.float64, buffer=bytearray(25), offset=1)
    with pytest.raises(ValueError, match="aligned"):
        function(array)
    array = np.ones(3, dtype=">f8" if np.little_endian else "<f8")
    with pytest.raises(TypeError, match="native-endian"):
        function(array)


@pytest.mark.parametrize("name", NAN_FUNCTIONS)
@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"ignore_inf": 1}, TypeError),
        ({"min_count": True}, TypeError),
        ({"min_count": 1.0}, TypeError),
        ({"min_count": 0}, ValueError),
        ({"min_count": 4}, ValueError),
    ],
)
def test_nan_options_validate_on_empty_input(
    name: str,
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        getattr(nf, name)(np.empty(0), **kwargs)


@pytest.fixture(scope="module")
def gold() -> dict[str, FloatArray]:
    """Load only frozen numeric arrays used by these independent checks."""
    with np.load(GOLD, allow_pickle=False) as archive:
        return {
            name: archive[name]
            for name in (
                "clean",
                "observed_sparse",
                "observed_stress",
                "control_flat",
                "control_ramp",
            )
        }


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layer", ["clean", "observed_sparse", "observed_stress"])
def test_gold_patches_match_independent_statistics(
    function: Callable[..., FloatArray],
    dtype: type[np.floating],
    layer: str,
    gold: dict[str, FloatArray],
) -> None:
    statistic = cast(
        Statistic, function.__name__.removeprefix("nan").removesuffix("_filter")
    )
    omit_nan = function.__name__.startswith("nan")
    # Source core, step, defect block, and image corner. The selected patches
    # are separate inputs; their edges deliberately use the requested mapping.
    for rows, cols in (
        (slice(46, 51), slice(55, 60)),
        (slice(119, 124), slice(157, 162)),
        (slice(176, 181), slice(41, 46)),
        (slice(0, 5), slice(0, 5)),
    ):
        array = gold[layer][rows, cols].astype(dtype)
        original = array.copy()
        for ignore_inf in (False, True) if omit_nan else (False,):
            options = dict(ignore_inf=ignore_inf, min_count=3) if omit_nan else {}
            if statistic in ("var", "std"):
                options["ddof"] = 1
            expected = statistics_filter_reference(
                array,
                (3, 3),
                ("reflect", "reflect"),
                statistic=statistic,
                omit_nan=omit_nan,
                **options,
            )
            actual = function(array, **options)
            _assert_statistics(actual, expected, statistic, 9)
            np.testing.assert_array_equal(array.view(np.uint8), original.view(np.uint8))


@pytest.mark.parametrize("dtype", DTYPES)
def test_gold_controls_have_analytic_extrema_and_scatter(
    function: Callable[..., FloatArray],
    dtype: type[np.floating],
    gold: dict[str, FloatArray],
) -> None:
    statistic = cast(
        Statistic, function.__name__.removeprefix("nan").removesuffix("_filter")
    )
    flat = gold["control_flat"].astype(dtype)
    actual = function(flat)
    np.testing.assert_array_equal(actual, 16 if statistic in ("min", "max") else 0)
    ramp = gold["control_ramp"].astype(dtype)
    # The frozen ramp is 16+2*x+4*y. A 3x3 window has extrema center +/-6
    # and sum of squared centered deviations 9*(2^2+4^2)*(2/3) = 120.
    if statistic in ("min", "max"):
        expected = ramp[1:-1, 1:-1] + (-6 if statistic == "min" else 6)
        _assert_statistics(function(ramp)[1:-1, 1:-1], expected, statistic, 9)
    else:
        for ddof in (0, 1):
            variance = 120 / (9 - ddof)
            answer = variance if statistic == "var" else np.sqrt(variance)
            expected = np.full(ramp[1:-1, 1:-1].shape, answer, dtype=dtype)
            _assert_statistics(
                function(ramp, ddof=ddof)[1:-1, 1:-1], expected, statistic, 9
            )
