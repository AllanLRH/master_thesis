import numpy as np


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
        raise ValueError("Inputs shapes must be identical. a.shape = {} and b.shape is {}".format(a.shape, b.shape))
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


def standardizeData(data):
    """
    Normalize the data by substracting the mean from each feature,
    and dividing by the standard deviation.
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    # set values of where there's no data to 1.0, since we're dividing with the std
    std[mean == 0] = 1.0
    normData = (data - mean)/std
    return normData
