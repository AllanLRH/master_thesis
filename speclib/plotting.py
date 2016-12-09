#!/usr/bin/env python
# -*- coding: utf8 -*-
from collections import Iterable


def looseAxesLimits(ax, loosen=0.05):
    """The default Matplotlib plot have very tight axes.
       This function loosenens the axes.

    Args:
        ax (axis): The axes-handle for the relevant axis.
        loosen (float or tuple, default = 0.05): The fractional amount to adjust the plot.
                   - If a float is passed, the plot distance will be modified by the same
                   value in all directions.
                   - If a tuple with 2 floats is passed, the first float will be applid in
                   the horizontal direction, and the second float will be applied in the
                   vertical direction.
                   - If a tuple with 4 floats is passed, the floats will be used when
                   adjusting the horizontal-left, horozontal-right, vertical-left and
                   vertical-right directions.

    Raises:
        ValueError: If loosen is not a float, 2-element tuple (2-tuple) or 4-tuple.
        ValueError: If contents in a loosen-tuple is not all floats.
    """
    # Get current min and max value for the axes, calculate the vertical and horizontal ranges
    axminX, axmaxX = ax.get_xlim()
    axminY, axmaxY = ax.get_ylim()
    axrngX = axmaxX - axminX
    axrngY = axmaxY - axminY

    # Check for input errors in loosen-argument
    if not isinstance(loosen, float) and len(loosen) not in (2, 4):
        raise ValueError("loosen-argument must be float, 2-tuple, or 4-tuple.")
    if isinstance(loosen, Iterable) and not all([isinstance(el, float) for el in loosen]):
        raise ValueError("The contents of loosen must all be floats.")

    # Parse the loosen-argument
    if isinstance(loosen, float):
        LX = RX = LY = UY = loosen
    elif len(loosen) == 2:
        LX = RX = loosen[0]
        LY = UY = loosen[1]
    else:
        LX, RX, LY, UY = tuple(loosen)

    # Calculate new axis-limits
    newXLinm = ((axminX - LX*axrngX), (axmaxX + RX*axrngX))
    newYLinm = ((axminY - LY*axrngY), (axmaxY + UY*axrngY))
    # Apply new limits
    ax.set_xlim(newXLinm)
    ax.set_ylim(newYLinm)


