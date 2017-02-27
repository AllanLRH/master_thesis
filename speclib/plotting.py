#!/usr/bin/env ipython
# -*- coding: utf8 -*-
from collections import Iterable
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx


def looseAxesLimits(ax, loosen=0.05):
    """The default Matplotlib plot have very tight axes.
       This function loosenens the axes.

    Args:
        ax (axis): The axes-handle for the relevant axis.
        loosen (float/int or tuple, default = 0.05): The fractional amount to adjust the plot.
                   - If a float is passed, the plot distance will be modified by the same
                   value in all directions.
                   - If a tuple with 2 floats is passed, the first float will be applid in
                   the horizontal direction, and the second float will be applied in the
                   vertical direction.
                   - If a tuple with 4 floats is passed, the floats will be used when
                   adjusting the left, right, bottom and top directions.

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
    if not (isinstance(loosen, float) or isinstance(loosen, int)) and len(loosen) not in (2, 4):
        raise ValueError("loosen-argument must be float/int, 2-tuple, or 4-tuple.")
    if isinstance(loosen, Iterable) and not all([(isinstance(el, float) or isinstance(el, int)) for el in loosen]):
        raise ValueError("The contents of loosen must all be floats or ints.")

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


def barSBS(ax, *args, offset=0.04, extraGroupSpace=None):
    """Plots dataseries as a bar chart, with the series bars next to each other.
    The passed order is preserved.

    Args:
        ax (axes): Axes to plot on.
        *args (dict): Data series contained in a dict. 'y' and 'label' are required keys,
                      referenceing to data and label respectively. 'x' is x-values and
                      is optional.
        offset (float, optional): Space between individual bars, default 0.04.
        extraGroupSpace (float, optional): Extra space between groups, default is 0 for
                                           less than 4 groups.
    """
    dataDicts = args
    if extraGroupSpace is None:
        extraGroupSpace = 0.0 if len(dataDicts) <= 3 else 0.015 * len(dataDicts)
    getNextColor = lambda: next(ax._get_lines.prop_cycler)['color']
    width = 1/(1.35*len(dataDicts)) - offset/2 - extraGroupSpace/2
    for i, dct in enumerate(dataDicts):
        if 'x' in dct:
            x = dct['x']
        else:
            x = np.arange(len(dct['y']))
        ax.bar(x+extraGroupSpace+i*(width + offset), dct['y'], width, color=getNextColor(),
               label=dct['label'])
    xtickLabels = ["%d" % el for el in ax.get_xticks()]
    ax.set_xticks(ax.get_xticks() + 0.35)
    ax.set_xticklabels(xtickLabels)
    ax.legend()


def countsOnBarPlot(ax):
    # attach some text labels
    # From: http://matplotlib.org/examples/api/barchart_demo.html
    # Don't include the last Rectangle, which is the canvas (or something like that)
    rectangles = [el for el in ax.get_children() if isinstance(el, mpl.patches.Rectangle)][:-1]
    for rect in rectangles:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')


def plotNeatoGraph(g, plotSettings=None, labels=None, fig_ax=None):
    """Plot a NetworkX graph, optionally add labels, and modify plot settings.

    Args:
        g (NetworkX graph): The graph to plot.
        plotSettings (dict, optional): Update plottting preference, options and defaults is:
                                        * 'node_color':   'steelblue',
                                        * 'edge_color':   'slategray',
                                        * 'figsize':      (16, 9),
                                        * 'font_color':   'mediumaquamarine',
                                        * 'font_size':    15,
                                        * 'font_weight':  'bold'
        labels (dict, optional): A dict with {'node': 'label'}.
        fig_ax (tuple, optional): Tuple containing (fig, ax) (Matplotlib figure and Axis).
        figsize (tuple, optional): Matplotlib figure size,

    Returns:
        (fig, ax): figure and axis
    """
    ps = {'node_color': 'steelblue',
          'edge_color':  'slategray',
          'figsize': (16, 9),
          'font_color': 'mediumaquamarine',  # for labelled nodes
          'font_size': 15,  # for labelled nodes
          'font_weight': 'bold'}  # for labelled nodes
    if plotSettings is not None:
        ps.update(plotSettings)
    fig, ax = plt.subplots(figsize=ps['figsize'])
    pos = nx.drawing.nx_agraph.graphviz_layout(g, prog='neato')
    nx.draw_networkx_nodes(g, pos, node_size=70, node_color=ps['node_color'], ax=ax)
    nx.draw_networkx_edges(g, pos, edge_color=ps['edge_color'])
    if labels:
        nx.draw_networkx_labels(g, pos, labels=labels, font_color=ps['font_color'],
                                font_size=ps['font_size'], font_weight=ps['font_weight'])
    ax.set_axis_bgcolor('white')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return (fig, ax)


def nxQuickDraw(G, **kwargs):
    """Quickly draw a networkx graph.

    Args:
        G (nx.Graph-like): Graph to draw, can be Graph, DiGramh, MultiGraph or DiMutiGraph.
        **kwargs: Additional keyword arguments to nx.draw.
    """
    kwDct = dict(with_labels=True,  # Use some sane defaults
                 node_color='lightblue',
                 edge_color='lightgray',
                 node_size=150)
    # Make a copy so that the defaults of the function is not changed whenever it's
    # called with arguments other that the graph
    useKwDct = kwDct.copy()
    useKwDct.update(kwargs)
    nx.draw(G, **useKwDct)


if __name__ == '__main__':
    d0 = {'y': (5 + np.random.randn(24))**2, 'label': 'SMS'}
    d1 = {'y': (5 + np.random.randn(24))**2, 'label': 'Call'}
    d2 = {'y': (5 + np.random.randn(24))**2, 'label': 'Call2'}
    d3 = {'y': (5 + np.random.randn(24))**2, 'label': 'SMS2'}
    fig, (ax0, ax1) = plt.subplots(ncols=1, nrows=2, figsize=(16, 6))
    barSBS(ax0, d0, d1)
    barSBS(ax1, d0, d1, d2, d3)
    plt.show()
