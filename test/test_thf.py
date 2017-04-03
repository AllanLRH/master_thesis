import thf
import numpy as np
import pandas as pd
import pytest


def test_thf_isArrRowSetEqual():
    arrA = np.array([['B', 'E', 'B'],
                     ['A', 'C', 'D', 'B'],
                     ['B', 'A'],
                     ['E', 'C'],
                     ['D', 'E', 'C', 'A']])

    arrApermuted1 = arrA.copy()
    arrApermuted1[[0, 3]] = arrApermuted1[[3, 0]]

    arrApermuted2 = arrA.copy()
    arrApermuted2[[0, 2]] = arrApermuted2[[2, 0]]

    tmp = arrA.copy()
    tmp[[0, 1]] = tmp[[1, 0]]
    dfApermuted3 = pd.DataFrame((pd.Series(row) for row in tmp))

    arrB = np.array([['B', 'E', 'B'],
                     ['A', 'C', 'D', 'B'],
                     ['Z', 'A', 'R'],
                     ['Z', 'B', 'R'],
                     ['B', 'A']])

    tmp[[2, 4]] = tmp[[4, 2]]
    dfApermuted4 = pd.DataFrame((pd.Series(row) for row in tmp))

    dfB = pd.DataFrame((pd.Series(row) for row in arrB))

    tmp = dfB.values.copy()
    tmp[[3, 4]] = tmp[[4, 3]]
    dfBpermuted1 = pd.DataFrame((pd.Series(row) for row in tmp))

    assert thf.isArrRowSetEqual(arrA        , arrApermuted1) is True  # noqa
    assert thf.isArrRowSetEqual(arrA        , arrApermuted2) is True  # noqa
    assert thf.isArrRowSetEqual(arrA        , arrApermuted1, arrApermuted2) is True  # noqa
    assert thf.isArrRowSetEqual(dfB         , dfBpermuted1) is True  # noqa
    assert thf.isArrRowSetEqual(dfApermuted3, dfApermuted4) is True
    assert thf.isArrRowSetEqual(dfApermuted3, dfB) is False
    assert thf.isArrRowSetEqual(arrA, arrB) is False
    with pytest.raises(ValueError, message="Expecting ValueError on diffent input types"):
        thf.isArrRowSetEqual(arrA           , dfApermuted3)  # noqa
    with pytest.raises(ValueError, message="Expecting ValueError on diffent input types"):
        thf.isArrRowSetEqual(arrA           , arrApermuted1, dfApermuted3)  # noqa


if __name__ == '__main__':
    test_thf_isArrRowSetEqual()
