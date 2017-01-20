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
