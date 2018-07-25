#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re
from IPython.display import display, HTML
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
mpl.rcParams['text.usetex'] = False

from speclib import misc, plotting, loaders

import missingno as msno


df = pd.read_json('../../allan_data/RGender_.json')
userAlias = loaders.Useralias()

q = misc.QuestionCompleter(df)
f = misc.QuestionFilterer(df)

msno.matrix(f.__answer)

dna = f.__answer.isna()
dnas = dna.sum(axis=0).sort_values(ascending=False)
msno.bar(f.__answer.loc[:, dnas > dnas.mean()])
plt.show()
dnas = dna.sum(axis=1)
msno.dendrogram(f.__answer)
plt.show()


def get_invert_idx_dict(idx):
    idx = sorted(idx.dropna().unique().tolist())
    return {i: j for (i, j) in zip(idx, reversed(idx))}


assert get_invert_idx_dict(pd.Series(np.arange(3))) == {0:2, 1:1, 2:0}


# ### Make an `alcohol_volume_total` column
alc = df.filter(regex=r'alcohol_volume.+__answer$')
print(*alc.columns, sep='\n')

df['alcohol_volume_total_answer'] = alc.sum(axis=1)

df['alcohol_volume_total_answer_type'] = 'radio'

df['alcohol_volume_total_condition'] = df.alcohol_volume_monday__condition.copy()

df['alcohol_volume_total_question'] = "CONSTRUCTED: sum of weekly alcohol intake"

df['alcohol_volume_total_response'] = df.filter(regex=r'alcohol_volume.+__response$').sum(axis=1)


# ### `ambition_career`
#
# Reversed scale and _ved ikke_ -> Nan.
df.loc[df.ambition_career__answer == 5, 'ambition_career__answer'] = np.NaN

df.loc[:, 'ambition_career__answer'] = df.ambition_career__answer.map(get_invert_idx_dict(df.ambition_career__answer))


# ### `ambition_job`
#
# Reverse scale and _jeg ved ikke hvilket job ..._ -> NaN
df.loc[df.ambition_job__answer == 4, 'ambition_job__answer'] = np.NaN

df.loc[:, 'ambition_job__answer'] = df.ambition_job__answer.map(get_invert_idx_dict(df.ambition_job__answer))


# ### `ambition_mark`
#
# Reverse scales and _ved ikke_ -> NaN
df.loc[df.ambition_mark__answer == 3, 'ambition_mark__answer'] = np.NaN

df.loc[:, 'ambition_mark__answer'] = df.ambition_mark__answer.map(get_invert_idx_dict(df.ambition_mark__answer))


# ### `conflicts_father`
#
# Reverse scale and set _har ingen_ -> NaN
df.loc[df.conflicts_father__answer == 1, 'conflicts_father__answer'] = np.NaN

df.loc[:, q.conflicts_father__answer] = df.conflicts_father__answer.map(get_invert_idx_dict(df.conflicts_father__answer))


# ### `conflict_friends`
#
# Reverse scale
df.loc[:, q.conflicts_friends__answer] = df.conflicts_friends__answer.map(get_invert_idx_dict(df.conflicts_friends__answer))


# ### `conflicts_mother`
#
# Reverse scale and set _har ingen_ -> NaN
df.loc[df.conflicts_mother__answer == 0, q.conflicts_mother__answer] = np.NaN

df.loc[:, q.conflicts_mother__answer] = df.conflicts_mother__answer.map(get_invert_idx_dict(df.conflicts_mother__answer))


# ### `conflicts_other_family`
#
# Reverse scale and set _har ingen_ -> NaN
df.loc[df.conflicts_other_family__answer == 1, q.conflicts_other_family__answer] = np.NaN

df.loc[:, q.conflicts_other_family__answer] = df.conflicts_other_family__answer.map(get_invert_idx_dict(df.conflicts_other_family__answer))


# ### `conflicts_partner`
#
# Reverse scale and set _har ingen_ -> NaN
df.loc[df.conflicts_partner__answer == 3, q.conflicts_partner__answer] = np.NaN

df.loc[:, q.conflicts_partner__answer] = df.conflicts_partner__answer.map(get_invert_idx_dict(df.conflicts_partner__answer))


# ### `conflicts_zieblings`
#
# Reverse scale and set _har ingen_ -> NaN
df.loc[df.conflicts_zieblings__answer == 3, q.conflicts_zieblings__answer] = np.NaN
df.loc[:, q.conflicts_zieblings__answer] = df.conflicts_zieblings__answer.map(get_invert_idx_dict(df.conflicts_zieblings__answer))


# ### `contact_father`
#
# Reverse scales, and set _har ingen_ -> NaN
df.loc[df.contact_father__answer == 5, q.contact_father__answer] = np.NaN
df.loc[:, q.contact_father__answer] = df.contact_father__answer.map(get_invert_idx_dict(df.contact_father__answer))


# ### `contact_friends`
#
# Reverse scales
df.loc[:, q.contact_friends__answer] = df.contact_friends__answer.map(get_invert_idx_dict(df.contact_friends__answer))


# ### `contact_mother`
#
# Reverse scales, and set _har ingen_ -> NaN
df.loc[df.contact_mother__answer == 5, q.contact_mother__answer] = np.NaN
df.loc[:, q.contact_mother__answer] = df.contact_mother__answer.map(get_invert_idx_dict(df.contact_mother__answer))


# ### `contact_other_familily`
#
# Reverse scales, and set _har ingen_ -> NaN
df.loc[df.contact_other_familily__answer == 0, q.contact_other_familily__answer] = np.NaN
# subtract 1 from index, because element 0 is eliminated
recode_dict = {1:4, 2:3, 3:2, 4:1, 5:0}  # noqa
df.loc[:, q.contact_other_familily__answer] = df.contact_other_familily__answer.map(recode_dict)


# ### `contact_partner`
#
# Recode scales
df.loc[df.contact_partner__answer == 1, q.contact_partner__answer] = np.NaN
recode_dict = {0:0, 2:1, 3:2, 4:3, 5:4}  # noqa
df.loc[:, q.contact_partner__answer] = df.contact_partner__answer.map(recode_dict)


# ### `contact_zieblings`
#
# Reverse scales and set _har ingen_ -> NaN
df.loc[df.contact_zieblings__answer == 5, q.contact_zieblings__answer] = np.NaN
df.loc[:, q.contact_zieblings__answer] = df.contact_zieblings__answer.map(get_invert_idx_dict(df.contact_zieblings__answer))


# ### `demands_father`
#
# Reverse scales and set _har ingen_ -> NaN
df.loc[df.demands_father__answer == 0, q.demands_father__answer] = np.NaN
# subtract 1 from index, because element 0 is eliminated
df.loc[:, q.demands_father__answer] = df.demands_father__answer.map(get_invert_idx_dict((df.demands_father__answer - 1) % 5))


# ### `demands_friends`
#
# Reverse scales
df.loc[:, q.demands_friends__answer] = df.demands_friends__answer.map(get_invert_idx_dict(df.demands_friends__answer))


# ### `demands_other_family`
#
# Recode scales, and _har ingen_ -> NaN
df.loc[df.demands_other_family__answer == 1, q.demands_other_family__answer] = np.NaN
recode_dict = {0:0, 2:1, 3:2, 4:3, 5:4}  # noqa
df.loc[:, q.demands_other_family__answer] = df.demands_other_family__answer.map(recode_dict)


# ### `demands_partner`
#
# Recode scales, and _har ingen_ -> NaN
df.loc[df.demands_partner__answer == 3, q.demands_partner__answer] = np.NaN
recode_dict = {0:0, 1:1, 2:2, 4:3}  # noqa
df.loc[:, q.demands_partner__answer] = df.demands_partner__answer.map(recode_dict)


# ### `demands_zieblings`
#
# Recode scales, and _har ingen_ -> NaN
df.loc[df.demands_zieblings__answer == 1, q.demands_zieblings__answer] = np.NaN
recode_dict = {0:3, 2:2, 3:1, 4:0}  # noqa
df.loc[:, q.demands_zieblings__answer] = df.demands_zieblings__answer.map(recode_dict)


# ### `drugs_12months`
#
# Recode
recode_dict = {0:1, 1:6, 2:2, 3:3, 4:4, 5:5 6:0}  # noqa
df[q.drugs_12months__answer] = drugs_12months__answer.map(recode_dict)


# ### `electronic_contact_family`
#
# Revert scale
df.loc[:, 'electronic_contact_family__answer'] = df.electronic_contact_family__answer.map(get_invert_idx_dict(df.electronic_contact_family__answer))











