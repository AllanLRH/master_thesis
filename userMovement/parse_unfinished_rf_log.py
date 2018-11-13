#!/usr/bin/env pythonw
# -*- coding: utf8 -*-

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
# import seaborn as sns
# sns.set(context='paper', style='whitegrid', color_codes=True, font_scale=1.8)
# colorcycle = [(0.498, 0.788, 0.498),
#               (0.745, 0.682, 0.831),
#               (0.992, 0.753, 0.525),
#               (0.220, 0.424, 0.690),
#               (0.749, 0.357, 0.090),
#               (1.000, 1.000, 0.600),
#               (0.941, 0.008, 0.498),
#               (0.400, 0.400, 0.400)]
# sns.set_palette(colorcycle)
import parse
# import validus


with open("rf_results_dirty.txt") as fid:
    data = fid.readlines()

pattern = "[CV]  randomforestclassifier__max_depth={max_depth:d}, randomforestclassifier__n_estimators={n_estimators:d}, score={score:f}, total={total:f}min"
lst = list()
for ln in data:
    prs = parse.parse(pattern, ln)
    if prs:
        lst.append(prs.named)
df = pd.DataFrame(lst)
print(df.head())
