import numpy as np
import pandas as pd
from sklearn import decomposition
import multiprocessing


def nanEqual(a, b):
    """Returs elementvise true for equal elements in inputs a and b, treading
       NaN's as equal.
       Inputs must both have the 'shape'-attribute and be of equal shape.

    Args:
        a (array): First array
        b (array): Second array

    Returns:
        boolean array

    Raises:
        ValueError: If any of the inputs doesn't have the 'shape' attribute.
        ValueError: The inputs aren't of equal shape.
    """
    if not (hasattr(a, 'shape') or hasattr(b, 'shape')):
        raise ValueError("Inputs a nad b must both have the shape-attribute (like numpy arrays)")
    if a.shape != b.shape:
        raise ValueError("Inputs shapes must be identical. a.shape = {} and b.shape is {}"
                         .format(a.shape, b.shape))
    return np.all( (a == b) | (np.isnan(a) & np.isnan(b)) )  # noqa


def timedelta2unit(timedelta, unit):
    """Convert a timedelta or iterable with Timedeltas to
       a numeric value matching a given unit.

    Args:
        timedelta (pd.Timedelta): Timedelta to convert. Can be an itetable of Timdeltas.
        unit (str): Unit to convert timedeltas to. Valid arguments are {'s', 'h', 'd', 'y'}.

    Returns:
        np.double: Numpy array of timedelta values matching given unit.

    Raises:
        ValueError: If input(s) lack(s) the 'total_seconds' method.
    """

    def _inner(time, unit):
        """Convert a pd.Timedelta ('time') to a numeric value matching 'unit'self.

        Args:
            time (pd.Timedelta): Timedelta to convert.
            unit (str): String indicating unit, same as outer function.

        Returns:
            double: time converted to a double matching given 'unit'
        """
        if not hasattr(time, 'total_seconds'):
            raise ValueError(('The input (or the elements of the iterable)' +
                              'must have the attribute "total_seconds".'))
        ts = time.total_seconds()
        if unit == 's':
            return ts
        if unit == 'h':
            return ts / 3600
        if unit == 'd':
            return ts / (24 * 3600)
        if unit == 'y':
            return ts / (365 * 24 * 3600)

    if hasattr(timedelta, '__iter__'):
        return np.fromiter((_inner(td, unit.lower()) for td in timedelta), np.double)
    return _inner(timedelta, unit.lower())


def standardizeData(data, getStdMean=False):
    """
    Normalize the data by substracting the mean from each feature,
    and dividing by the standard deviation.
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    # set values of where there's no data to 1.0, since we're dividing with the std
    std[mean == 0] = 1.0
    normData = (data - mean)/std
    if getStdMean:
        return (normData, std, mean)
    return normData


def pcaFit(toPca, standardizeData=True, **kwargs):
    """Standardize data, create a PCA object and fit the data.

    Args:
        toPca (np.array): Data to perform PCA analysis on.
        standardizeData (bool, optional): Standardize data if True, leave as be othewise.
        **kwargs (dict, optional): Additional keyword arguments to PCA.

    Returns:
        PCA: Fitted PCA instance.
    """
    if standardizeData:
        toPca, std, mean = standardizeData(toPca, getStdMean=True)
    else:
        mean, std = np.NaN, np.NaN
    pca = decomposition.PCA(**kwargs)
    pca.fit(toPca)
    pca.norm_std = std
    pca.norm_mean = mean
    return pca


def mapAsync(func, funcargLst, n=None):
    """Given a function and an iterable with tuple-packed arguments, evalueate the
    the function in parallel using multiple processors.

    Args:
        func (function): Function to evalueate.
        funcargLst (list): List with tuples containing arguemnts for func.
        n (int, optional): Number of processors to use. Default is 16 or number of
                           processors - 1 (if there isn't 16 processors avaiable).

    Returns:
        list: List with result.
    """
    if n is None:
        n = 16 if 16 < multiprocessing.cpu_count() else multiprocessing.cpu_count() - 1
    with multiprocessing.Pool(n) as pool:
        call = pool.starmap_async(func, funcargLst)  # parse args correctly
        res = call.get()
        pool.close()
        pool.join()
        return res


def lstDct2dct(lst):
    """Merge a list of dicts into a single dict.
    Raises warning if there's overlapping key values.

    Args:
        lst (list): List with dictionaries.

    Returns:
        dict: Merged dictionary.
    """
    retDct = dict()
    for dct in lst:
        keyIntersection = set(dct.keys()).intersection(set(retDct.keys()))
        if keyIntersection:
            Warning('Keys are being overwritten: {}'.format(keyIntersection))
        retDct.update(dct)
    return retDct


def randomSample(itr, n):
    """Draw n samples from iterable, where each element can only be drawn once.

    Args:
        itr (iterable): Iterable to draw from.
        n (int): number of elements to draw.

    Returns:
        list: n elements from itr.

    Raises:
        ValueError: If the requested number of draws isn't smaller then the length if itr.
    """
    if len(itr) <= n:
        raise ValueError("'itr' must be longer that n")
    idx = np.arange(len(itr))
    np.random.shuffle(idx)
    ret = [itr[i] for i in idx[:n]]
    return ret


def getFirstDayInTimeseries(ts):
    """Given a Pandas time series, get the Timestamp for the start of the day for the
    earliest entry in the time series.

    Args:
        ts (Series[Tiemstamp]): Pandas Series with Timestamps (datetime might also work).

    Returns:
        Timestamp: Pandas Timestamp matching the data for the first entry in ts.
    """
    t0 = ts.min()
    t0d = t0.date()
    return pd.Timestamp(t0d)
