import pandas as pd
import itertools

# ****************************************************************************
# *                      File with test helper functions                     *
# ****************************************************************************


def isArrRowSetEqual(*arrs):
    """Compares ndarrays or dataframes contents, treating each of them as a set, where
    the rows becomes sets.

    Args:
        *arrs: Arrays to be tested for equality. May be ndarray or dataframes

    Returns:
        bool: True if all are "set-set"-equal, False otherwise.

    Raises:
        ValueError: If there's a mixed input of ndarrays and dataframes.
    """
    setDiff = lambda s0, s1: s0.difference(s1)

    if len({type(arr) for arr in arrs}) > 1:
        raise ValueError("All arrays must be if the same type.")

    if isinstance(arrs[0], pd.DataFrame):
        arrTup = tuple(arr.values for arr in arrs)
    else:
        arrTup = arrs

    setLst = [{frozenset(row) for row in arr} for arr in arrTup]
    for el in itertools.product(setLst, repeat=2):
        if setDiff(*el) != set():
            return False
    return True
