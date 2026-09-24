"""Generated parity tests against the independent N-D reference."""

import ndfiltrs as nf
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tests._reference import MODES, median_filter_reference


@settings(max_examples=80, deadline=None)
@given(
    data=st.data(),
    rank=st.integers(min_value=1, max_value=3),
    dtype=st.sampled_from([np.float32, np.float64]),
    omit_nan=st.booleans(),
    ignore_inf=st.booleans(),
)
def test_small_nd_filters_match_independent_reference(
    data: st.DataObject,
    rank: int,
    dtype: type[np.float32] | type[np.float64],
    omit_nan: bool,
    ignore_inf: bool,
) -> None:
    shape = data.draw(
        st.tuples(*(st.integers(min_value=1, max_value=4) for _ in range(rank)))
    )
    size = data.draw(st.tuples(*(st.sampled_from([1, 3, 5]) for _ in range(rank))))
    mode = data.draw(st.tuples(*(st.sampled_from(MODES) for _ in range(rank))))
    values = data.draw(
        st.lists(
            st.sampled_from([-np.inf, -3.5, -0.0, 0.0, 2.25, np.inf, np.nan]),
            min_size=int(np.prod(shape)),
            max_size=int(np.prod(shape)),
        )
    )
    array = np.asarray(values, dtype=dtype).reshape(shape)
    volume = int(np.prod(size))
    min_count = data.draw(st.integers(min_value=1, max_value=volume))
    cval = data.draw(st.sampled_from([-np.inf, -1.25, -0.0, 0.0, 4.0, np.inf, np.nan]))
    expected = median_filter_reference(
        array,
        size,
        mode,
        cval=float(dtype(cval)),
        omit_nan=omit_nan,
        ignore_inf=ignore_inf if omit_nan else False,
        min_count=min_count if omit_nan else 1,
    )
    if omit_nan:
        actual = nf.nanmedian_filter(
            array,
            size,
            mode=mode,
            cval=cval,
            ignore_inf=ignore_inf,
            min_count=min_count,
        )
    else:
        actual = nf.median_filter(array, size, mode=mode, cval=cval)
    np.testing.assert_array_equal(
        actual.view(f"u{array.dtype.itemsize}"),
        expected.view(f"u{array.dtype.itemsize}"),
    )
