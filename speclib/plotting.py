#!/usr/bin/env ipython
# -*- coding: utf8 -*-

from collections import Iterable
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import palettable
import itertools
from speclib import graph


def rgb(r, g, b):
    """Turn 0-255 spaced RGB-values into a 0-1 spaced numpy array, which is readable
    by Matplotlib.
    """
    return [np.array([r, g, b])/255]


def looseAxesLimits(ax, loosen=0.05):
    """The default Matplotlib plot have very tight axes.
       This function loosenens the axes.

    Parameters
    ----------
    ax : axis
        The axes-handle for the relevant axis.
    loosen : float/int or tuple, default = 0.05
        The fractional amount to adjust the plot.
        - If a float is passed, the plot distance will be modified by the same value in
          all directions.
        - If a tuple with 2 floats is passed, the first float will be applid in the
          horizontal direction, and the second float will be applied in the vertical
          direction.
        - If a tuple with 4 floats is passed, the floats will be used when adjusting the
          left, right, bottom and top directions.

    Raises
    ------
    ValueError
        If contents in a loosen-tuple is not all floats.
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

    Parameters
    ----------
    ax : axes
        Axes to plot on.
    *args : dict
        Data series contained in a dict. 'y' and 'label' are required keys, referenceing
        to data and label respectively. 'x' is x-values and is optional.
    offset : float, optional
        Space between individual bars, default 0.04.
    extraGroupSpace : float, optional
        Extra space between groups, default is 0 for less than 4 groups.
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
    """Attach some text labels
    From: http://matplotlib.org/examples/api/barchart_demo.html
    Don't include the last Rectangle, which is the canvas (or something like that)
    """
    rect_list = [el for el in ax.get_children() if isinstance(el, mpl.patches.Rectangle)][:-1]
    offset = np.mean([0.05*rect.get_height() for rect in rect_list])
    for rect in rect_list:
        height = rect.get_height()
        if int(height) != 0:
            ax.text(rect.get_x() + rect.get_width()/2., height + offset,
                    '%d' % int(height), ha='center', va='bottom')


def plotNeatoGraph(g, plotSettings=None, labels=None, fig_ax=None):
    """Plot a NetworkX graph, optionally add labels, and modify plot settings.

    Parameters
    ----------
    g : NetworkX graph
        The graph to plot.
    plotSettings : dict, optional
        Update plottting preference, options and defaults is:
        * 'node_color':   'steelblue',
        * 'edge_color':   'slategray',
        * 'figsize':      (16, 9),
        * 'font_color':   'mediumaquamarine',
        * 'font_size':    15,
        * 'font_weight':  'bold'
    labels : dict, optional
        A dict with {'node': 'label'}.
    fig_ax : tuple, optional
        Tuple containing (fig, ax) (Matplotlib figure and Axis).

    Returns
    -------
    (fig, ax)
        figure and axis

    Deleted Parameters
    ------------------
    figsize : tuple, optional
        Matplotlib figure size,
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

    Parameters
    ----------
    G : nx.Graph-like
        Graph to draw, can be Graph, DiGramh, MultiGraph or DiMutiGraph.
    **kwargs
        Additional keyword arguments to nx.draw.
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


def barFractionPlot(df, ax=None, userOrder=None):
    """Plots a horizontal stacked bar chart colored by percent.

    Parameters
    ----------
    df : DataFrame
        1 dimmensional dataframe.
    ax : axis, optional
        Axis to plot on.
    userOrder : list, optional
        Use this order of users.

    Returns
    -------
    axis
        Axis, given or created.

    Raises
    ------
    ValueError
    If len(df.shape) > 1
    """
    if len(df.shape) > 1:
        raise ValueError("Only 1-dimmensional dataFrames are accepted")
    if ax is None:  # Create plot and axis since none is provided
        _, ax = plt.subplots(figsize=(20, 1))
    # Colorcycler, using colorbrewer Set3, with an an appropriate number of colors,
    # limited to numbers 3 through 13. It will cycle infinitely.
    mplColors = itertools.cycle(
        palettable.colorbrewer.qualitative.__dict__[
            'Set3_%d' % max(3, min(13, df.shape[0]))].mpl_colors
    )
    oldLeft = 0.0  # position where previous bar ended
    if userOrder is None:  # No ordering of users probided, sort descending
        useDf = df.sort_values(ascending=False)
    else:  # Use provided ordering
        useDf = df.loc[userOrder]
    useDf /= df.values.sum()
    #   count
    #   |    user
    #   |    |    communicationFraction
    #   |    |    |     bar color
    #   |    |    |     |                       Iterate df as a dict
    #   |    |    |     |                       |               Tag plot color on
    for i, ((lbl, val), color) in enumerate(zip((useDf).items(), mplColors)):
        ax.barh(bottom=0, width=val, left=oldLeft, color=color, height=1.0)
        ax.annotate(lbl,
                    xy=(oldLeft, 0.010),
                    xycoords='data',
                    xytext=(60, 0),
                    textcoords='offset pixels',
                    horizontalalignment='center',
                    verticalalignment='bottom')
        oldLeft += val
    ax.set_xticks(np.cumsum(useDf.values))
    ax.set_xticklabels(["%.3f" % fl for fl in useDf.values], rotation=45)
    ax.set_yticklabels([])
    ax.set_xbound(0, 1)
    return (ax, useDf.index.tolist())


def plotPunchcard(data):
    """Plots a "punchcard of user activity, using the pcolor plot function."

    Parameters
    ----------
    data : 2d array
        An matrix with activity binned hourly.
        Users along the y-axis, hours along the x-axis.

    Returns
    -------
    Figure, Axis
        Figure and axis of plot.
    """
    fig, ax = plt.subplots()
    pc = ax.pcolorfast(data, cmap=mpl.cm.viridis)
    fig.colorbar(pc)
    ax.set_xlim(0, data.shape[1])
    ax.set_ylim(0, data.shape[0])
    tickmarks = np.arange(0, 7*24, 24) + 12
    ax.set_xticks(tickmarks)
    ax.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                        'Friday', 'Saturday', 'Sunday'], rotation=45)
    ax.grid()
    for idx in np.arange(24, 7*24, 24):
        ax.axvline(x=idx, linestyle='--', color='white', linewidth=1.5, zorder=3)
    ax.set_yticklabels([])
    ax.set_ylabel('Users')
    return fig, ax


def drawWeightedGraph(g, normailzeWeights=True, weightFunc=None, ax=None, layout=None,
                      nodeLabels=True, edgeLabels=False, kwLayout=None,
                      kwNode=None, kwEdge=None, kwNodeLabel=None, kwEdgeLabel=None):
    """Draw a weighted graph.

    Parameters
    ----------
    g : Networkx graph
        Graph to be plottet
    normailzeWeights : bool, optional
        Normalize weight to be in range 0-1 before applying weightFunc. Default = True.
    weightFunc : function, optional
        Function to apply to weight before it's drawn as a link. Default: 2.5 * weight + 1.
    ax : axes, optional
        Matplotlib axes to plot on.
    layout : dict or layout engine, optional
        A position dict or a layout engine.
    nodeLabels : bool or dict, optional
        Draw node labels if True (default True), or use provided dict for naming.
        Default name is the string representation of the node name.
    edgeLabels : bool or dict, optional
        Draw edge labels if True (default False), or use provided dict .
        Default edge labels are the edge weights.
    kwLayout : dict, optional
        Keyword arguments to layout engine.
    kwNode : dict, optional
        Keyword arguments to nx.draw_networkx_nodes.
    kwEdge : dict, optional
        Keyword arguments to nx.draw_networkx_edges.
    kwNodeLabel : None, optional
        Keyword arguments to nx.drawing.draw_networkx_labels.
    kwEdgeLabel : None, optional
        Keyword arguments to nx.drawing.draw_networkx_edge_labels.

    Returns
    -------
    axes
        Matplotlib axes.

    Raises
    ------
    ValueError
    If the input for .
    """

    # Create axes if none were given by the user
    if ax is None:
        _, ax = plt.subplots()

    # ****************************************************************************
    # *                                  Weights                                 *
    # ****************************************************************************

    # The default weight func, used to adjust the width of the link-lines, as they might
    # otherwise disappear
    if weightFunc is None:
        weightFunc = lambda weight: 2.5*weight + 1

    # Check for negative weights, and define the
    # weight-normalization function if necessary
    weightLst = [g[edge[0]][edge[1]]['weight'] for edge in g.edges()]
    if normailzeWeights:
        weightMin = min(weightLst)
        if weightMin < 0:
            raise ValueError("Can't handle negative weights.")
        weightMax = max(weightLst)
        weightNorm = lambda weight: weight/weightMax
    else:
        weightNorm = lambda weight: weight

    # ****************************************************************************
    # *                           Update with default values                     *
    # ****************************************************************************

    # Dicts with default arguments to the functions nx.draw_networkx_nodes and
    # nx.draw_networkx_edges
    # The dicts are updated with the user supplied arguments kwNode and kwEdge
    kwargsNode = {'node_color': rgb(39, 139, 176), 'node_size': 50}
    kwargsEdge = {'edge_color': rgb(186, 199, 207)}
    kwargsNodeLabel = {'font_size': 10}
    kwargsEdgeLabel = {'font_size': 10}
    if kwNode is not None:
        kwargsNode.update(kwNode)
    if kwEdge is not None:
        kwargsEdge.update(kwEdge)
    if kwNodeLabel is not None:
        kwargsNodeLabel.update(kwNodeLabel)
    if kwEdgeLabel is not None:
        kwargsEdgeLabel.update(kwEdgeLabel)

    # ****************************************************************************
    # *                                  Layout                                  *
    # ****************************************************************************

    if layout is None:  # User didn't supply custom layout specificaiton
        # Get positions for nodes using Fruchterman Reingold layout
        # Initial positioning, notmally only using 50 iterations
        # Also avaiable in; nx.spring_layout
        pos = nx.drawing.layout.spring_layout(g, iterations=100)
        # Fine tune the layout, starting with the previous layout
        pos = nx.drawing.layout.spring_layout(g, pos=pos, iterations=30)
    elif isinstance(layout, dict):  # layout is a position dict, just use the provided layout
        pos = layout
    else:  # User supplied layout engine
        # Don't use keyword expansion with None, which is the default value
        kwLayout = dict() if kwLayout is None else kwLayout
        pos = layout(g, **kwLayout)

    # ****************************************************************************
    # *                                Draw nodes                                *
    # ****************************************************************************

    nx.draw_networkx_nodes(g, pos, ax=ax, **kwargsNode)

    # ****************************************************************************
    # *                                Draw edges                                *
    # ****************************************************************************

    for edge in g.edges():
        weight = g[edge[0]][edge[1]]['weight']
        nx.draw_networkx_edges(g, pos, ax=ax, edgelist=[edge],
                               width=weightFunc(weightNorm(weight)), **kwargsEdge)

    # ****************************************************************************
    # *                             Draw node labels                             *
    # ****************************************************************************
    if not (isinstance(nodeLabels, bool) or isinstance(nodeLabels, dict)):
        raise ValueError("Invaid input for nodeLabels, must be a bool or dict.")
    if isinstance(nodeLabels, bool) and nodeLabels:
        nodeLabels = {node: str(node) for node in g.nodes()}
    elif isinstance(nodeLabels, dict):
        pass  # use nodeLabels-dict as input to name inputs
    else:
        pass
    if nodeLabels is not False:  # None, the default value, would evalueate as False
        nx.drawing.draw_networkx_labels(g, pos, nodeLabels, ax=ax, **kwargsNodeLabel)

    # ****************************************************************************
    # *                             Draw edge labels                             *
    # ****************************************************************************
    if not (isinstance(edgeLabels, bool) or isinstance(edgeLabels, dict)):
        raise ValueError("Invaid input for edgeLabels, must be a bool or dict.")
    if isinstance(edgeLabels, bool) and edgeLabels:
        edgeLabels = {edge: '{:.3e}'.format(weight) for (edge, weight) in
                      nx.get_edge_attributes(g, 'weight').items()}  # {edge: weight} dict
    elif isinstance(edgeLabels, dict):
        pass  # use edgeLabels-dict as input to name inputs
    if edgeLabels:
        nx.drawing.draw_networkx_edge_labels(g, pos, edgeLabels, ax=ax, **kwargsEdgeLabel)

    return ax


class PcaPlotter(object):
    """Class functioning as a convinience wrapper for plotting the of the PCA objects.

    Attributes
    ----------
    comDelta : float
        Cutoff value for when a value in the eigenvector is considerd a connection.
    firstN : array
        The first N eignvectors, explaining at least `explanationCut` of variance.
    graphLst : list
        List with graphs generated from firstN.
    n : int
        Number of eigenvectors used, equal to len(firstN).
    pca : pca
        pca instance from scikit learn.
    users : tuple
        tuple with users in community in the pca.
    """

    def __init__(self, pca, users, explanationCut=0.95, comDelta=1e-6):
        """Initializer for the PcaPlotter class.

        Parameters
        ----------
        pca : pca
            pca instance from scikit learn.
        users : tuple
            tuple with users in community in the pca.
        explanationCut : float, optional
            Cutoff value determening how much of the variance should be explaned by the
            eigenvectors / principal components.
        comDelta : float, optional
            Cutoff value for when a value in the eigenvector is considerd a connection.
        """
        super(PcaPlotter, self).__init__()
        self.pca = pca
        self.users = users
        self.comDelta = comDelta
        self.setExplanationCut(explanationCut)
        self._makeGraphList()

    def setExplanationCut(self, val):
        """Set cut value for fraction of explanied variance.

        Parameters
        ----------
        val : float
            Cut value in range [0, 1].

        Raises
        ------
        ValueError
            If val is out of bounds.
        """
        if val < 0.0 or val > 1.0:
            raise ValueError(f"val must be between 0.0 and 1.0 (was {val})")
        self._explanationCut = val
        self.n = (np.cumsum(self.pca.explained_variance_ratio_) <=
                  self.pca.explained_variance_ratio_.sum()*self._explanationCut).sum()
        self.firstN = np.abs(self.pca.components_[:, :self.n])

    def plotHeatmap(self, ax=None, cmap='RdBu_r'):
        """Plot a heatmap of the eigenvectors, number if vectors determed by
        explanationCut.

        Parameters
        ----------
        ax : axsi, optional
            Axis to draw on.
        cmap : str, optional
            Colormap to use.
            A diverging map is preferred.

        Returns
        -------
        tuple
            Figure and axis used for the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        pc = ax.pcolorfast(self.pca.components_[:, :self.n], cmap=cmap,
                           vmin=-np.abs(self.firstN).max(), vmax=np.abs(self.firstN).max())
        fig.colorbar(pc)
        return (fig, ax)

    def _makeGraphList(self):
        """Construct a list of graphs from the eigenvectors.

        Returns
        -------
        list
            List of graphs.
        """
        firstN = self.firstN + self.firstN.min()  # noqa
        firstN[firstN < self.comDelta] = 0.0
        firstN /= firstN.sum()
        firstN *= self.n
        self.graphLst = list()
        for i in range(self.n):
            if self.pca.symmetric:
                adjmat = graph.upperTril2adjMat(firstN[:, i])
            else:
                adjmat = graph.vec2squareMat(firstN[:, i])
            g = nx.from_numpy_matrix(adjmat)
            self.graphLst.append(g)

    def plotGraphs(self, weightFunc=None, layout=None, kwargs=None, weightMultiply=1000):
        """Plot the graphs corresponding to the eigenvectors/principal components.
        Note that is't a generator, so it needs to be depleted in order to produce all
        the plots.

        Parameters
        ----------
        weightFunc : function, optional
            function for transformaing the weight, see docs for drawWeightedGraph.
        layout : dict, optional
            Dict with layout, as specified på Networkx.
        kwargs : dict, optional
            Keyword arguments to drawWeightedGraph.
        weightMultiply : int, optional
            Multiply weights with this int, in order to get human readable values.

        Yields
        ------
        tuple
            Figure and axis used for the plot.
        """
        if weightFunc is None:
            weighFunc = lambda w: 5*w + 0.5
        if layout is None:
            layout = nx.drawing.layout.circular_layout(self.graphLst[0])
        for i in range(self.n):
            fig, ax = plt.subplots(figsize=(10, 6))
            edgeLabels = {edge: '{:.0f}'.format(np.round(weightMultiply*weight)) for (edge, weight)
                          in nx.get_edge_attributes(self.graphLst[i], 'weight').items()}
            drawWeightedGraph(self.graphLst[i], ax=ax, layout=layout, normailzeWeights=False,
                              weightFunc=weighFunc, nodeLabels=True, edgeLabels=edgeLabels)
            fig.suptitle(f'Vector {i+1}/{self.n}, {len(self.users)} users')
            yield (fig, ax)

    def plotStandardization(self, smooth=8):
        """Plot the standardisation values used when standardizing the data before the PCA
        algorithm processed it.
        Procuces two plots, an unsmoothed and a smoothed one.

        Parameters
        ----------
        smooth : int, optional
            Size of running average for bottom window.

        Returns
        -------
        tuple
            Figure and axis used for the plot.
        """
        fig, axi = plt.subplots(2, 1)
        for ax, n, lbl in zip(axi, [1, smooth], ['', ' (smoothed)']):
            ax.plot(np.convolve(np.ones(n)/n, self.pca.norm_mean, 'same'), label='norm_mean' + lbl, color='#20a365')
            ax.plot(np.convolve(np.ones(n)/n, self.pca.norm_std, 'same'), label='norm_std' + lbl, color='#ea8a3f')
            ax.legend(loc='best')
        return (fig, axi)


if __name__ == '__main__':
    d0 = {'y': (5 + np.random.randn(24))**2, 'label': 'SMS'}
    d1 = {'y': (5 + np.random.randn(24))**2, 'label': 'Call'}
    d2 = {'y': (5 + np.random.randn(24))**2, 'label': 'Call2'}
    d3 = {'y': (5 + np.random.randn(24))**2, 'label': 'SMS2'}
    fig, (ax0, ax1) = plt.subplots(ncols=1, nrows=2, figsize=(16, 6))
    barSBS(ax0, d0, d1)
    barSBS(ax1, d0, d1, d2, d3)
    plt.show()

