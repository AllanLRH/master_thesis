#!/usr/bin/env python
# -*- coding: utf8 -*-


def looseAxesLimits(ax, loosen=0.05):
    """
    The default Matplotlib plot have very tight axes.
    This function loosenens the axes.
    Takes an axes-object as input.
    """
    axminX, axmaxX = ax.get_xlim()
    axminY, axmaxY = ax.get_ylim()
    axrngX = axmaxX - axminX
    axrngY = axmaxY - axminY

    if isinstance(loosen, float):
        LX = RX = LY = UY = loosen
    elif len(loosen) == 2:
        LX = RX = loosen[0]
        LY = UY = loosen[1]
    else:
        LX, RX, LY, UY = tuple(loosen)

    newXLinm = ((axminX - LX*axrngX), (axmaxX + RX*axrngX))
    newYLinm = ((axminY - LY*axrngY), (axmaxY + UY*axrngY))
    ax.set_xlim(newXLinm)
    ax.set_ylim(newYLinm)
