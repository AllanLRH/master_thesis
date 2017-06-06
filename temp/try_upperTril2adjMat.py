#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import os
sys.path.append(os.path.abspath(".."))
import numpy as np
from speclib import graph

sz = lambda n: int((n**2 - n)/2 + 1)
arr = np.arange(1, sz(4))
mat = graph.upperTril2adjMat(arr)

print(arr, end='\n\n')

print(mat)










