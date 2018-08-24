#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
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
mpl.rcParams['figure.max_open_warning'] = 65
mpl.rcParams['figure.figsize'] = [12, 7]

# import warnings
# warnings.simplefilter("ignore", category=DeprecationWarning)
# warnings.simplefilter("ignore", category=mpl.cbook.mplDeprecation)
# warnings.simplefilter("ignore", category=UserWarning)


pd.set_option('display.max_rows', 55)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
np.set_printoptions(linewidth=145)

# import pixiedust


# In[2]:


rf = pd.DataFrame([['B7', 60.635047],
['B3', 14.030777],
['G7',  5.376852],
['G3',  3.488555],
['S7',  1.740810],
['B5',  1.709319],
['C3',  1.372402],
['B6',  1.332545],
['S3',  1.246466],
['C7',  1.102512],
['B2',  0.869122],
['B1',  0.719041],
['B8',  0.606133],
['G6',  0.502796],
['G2',  0.445082],
['C4',  0.396808],
['S4',  0.393738],
['G5',  0.362911],
['C8',  0.317319],
['S6',  0.305988],
['S2',  0.303238],
['G1',  0.291071],
['S8',  0.284002],
['G4',  0.274439],
['B4',  0.267163],
['C2',  0.261367],
['G8',  0.255945],
['C6',  0.255237],
['S5',  0.243382],
['S1',  0.214690],
['C5',  0.199162],
['C1',  0.196081]], columns=['ch', 'rf'])

gb = pd.DataFrame([['B7', 11.686819],
['B3',  6.249174],
['G7',  5.426864],
['G3',  4.500369],
['C3',  4.247084],
['S7',  4.181196],
['C7',  4.175127],
['B5',  3.794517],
['S3',  3.678496],
['B6',  3.134801],
['B1',  2.996436],
['G6',  2.885696],
['S6',  2.760858],
['B2',  2.675635],
['G2',  2.581213],
['B8',  2.539568],
['S2',  2.490446],
['C4',  2.444982],
['G4',  2.302866],
['C1',  2.216359],
['G1',  2.169698],
['C8',  2.164550],
['S1',  2.104467],
['C6',  2.098280],
['G8',  2.056467],
['S5',  1.998123],
['C5',  1.931264],
['G5',  1.837230],
['B4',  1.815814],
['S8',  1.674364],
['S4',  1.669391],
['C2',  1.511847]], columns=['ch', 'gb'])


# In[3]:


rf = rf.set_index('ch')
gb = gb.set_index('ch')


# In[7]:


df = rf.join(gb)


# In[8]:


df.head()


# In[242]:


minspace = 2.1
plt.rc('text', usetex=True)
dfd = df.diff(periods=-1).abs().fillna(0)
dfd.head()

dfdm = pd.DataFrame(dfd.min(axis=1))
dfdm['ofs'] = minspace*np.arange(dfdm.shape[0])[::-1]

dfdm['tofs'] = dfdm.sum(axis=1)

colordict = {k: colorcycle[v] for (v,k) in enumerate('CSGB')}
titleofs = 0.5

fig, ax = plt.subplots(figsize=(16, 24))
ax.grid(False)
ax.set_ybound(0, 1.1*df.max().max())
ax.set_xticks([1, 8])
# ax.set_xticklabels(['Random Forest', 'Gradient Boost'])
ax.set_xticklabels([])
ax.set_yticklabels([])
xl, xr = 1, 8
# plt.box()
for i, (cha, row) in enumerate(df.iterrows()):
    yl, yr = row.values
    sl, sr = f"{cha}: {yl:.2f} \\%", f"{cha}: {yr:.2f} \\%"
    yl, yr = row.values + dfdm.loc[cha, 'tofs']
    ax.text(xl, yl, sl, fontsize=14, horizontalalignment='right')
    ax.text(xr, yr, sr, fontsize=14)
    ax.plot([xl+0.1, xr-0.2], [yl-minspace/10, yr-minspace/10], 'o-', color=colordict[cha[0]])
ymin, ymax = ax.get_ylim()
ax.plot([-0.2, 9.3], [ymin - titleofs*0.2, ymin - titleofs], 'k-')
ax.plot([-0.2, 9.3], [ymax + titleofs, ymax + titleofs], 'k-')
ax.text(1.4, ymax + 3*titleofs, "Random Forest", fontsize=14, horizontalalignment='right')
ax.text(1.4, ymin - 4*titleofs, "Random Forest", fontsize=14, horizontalalignment='right')
ax.text(7.5, ymax + 3*titleofs, "Gradient Boostring", fontsize=14, horizontalalignment='left')
ax.text(7.5, ymin - 4*titleofs, "Gradient Boostring", fontsize=14, horizontalalignment='left')
fig.savefig('figs/rf_gb_feature_importance_rankplot.pdf')
