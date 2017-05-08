#!/usr/bin/env sh
patch $(python -c "import igraph; print(igraph.drawing.__file__)") ./igraph_svg_draw.patch
