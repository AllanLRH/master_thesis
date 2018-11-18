#!/usr/bin/env python
# -*- coding: utf8 -*-

import parse
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(context='paper', style='whitegrid', color_codes=True, font_scale=1.8)
colorcycle = [(0.498, 0.788, 0.498),
              (0.745, 0.682, 0.831),
              (0.992, 0.753, 0.525),
              (0.220, 0.424, 0.690),
              (0.749, 0.357, 0.090),
              (1.000, 1.000, 0.600),
              (0.941, 0.008, 0.498),
              (0.400, 0.400, 0.400)]
sns.set_palette(colorcycle)



parse_pattern = "[CV]  pca__n_components={n:d}, svc__C=1{C:f}, svc__class_weight={wt:w}, svc__gamma={gamma:f}, score={auc:f}, total={time:f}{tu:w}"
rx = re.compile(r'total= +')

with open("pca_svm_progress_log.txt") as fid:
    lns = fid.read().split('\n')
    lns = [rx.sub('total=', ln) for ln in lns]

res = list()
for ln in lns:
    psd = parse.parse(parse_pattern, ln)
    if psd:
        res.append(psd.named)
df = pd.DataFrame(res)

df['tu'] = df.tu.map({'s': 1, 'min': 60})

df['time'] = df.time * df.tu
df = df.drop('tu', axis=1)
df.loc[(df.wt == 'None'), 'wt'] = 'unbalanced'

stats = df.groupby('gamma')[['time', 'auc']].agg(['mean', 'std', 'count'])
print(stats)
