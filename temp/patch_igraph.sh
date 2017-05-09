#!/usr/bin/env sh
# From https://github.com/igraph/python-igraph/commit/8864b46849b031a3013764d03e167222963c0f5d
patch $(python -c "import igraph; print(igraph.drawing.__file__)") ./igraph_svg_draw.patch
# From http://stackoverflow.com/questions/30640489/issue-plotting-vertex-labels-using-igraph-in-ipython
patch $(python -c "import igraph; print(igraph.drawing.__file__)") ./igraph_svg_to_png_draw.patch
