#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import os
sys.path.append(os.path.abspath("/lscr_paper/allan/scripts"))

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

from numba import jit

from speclib import misc, plotting, loaders, graph  # noqa
PRINT = False


@jit()
def compareDfUsers(baseuser, peers, df):
    # Compute the similarity in the way they answered the questions
    dct = dict()
    for i in range(len(peers)):
        dct[(baseuser, peers[i])] = simfnc(df.loc[baseuser], df.loc[peers[i]])
    sim = pd.Series(dct).sort_values(ascending=False)
    return sim


dfa = pd.read_msgpack('/lscr_paper/allan/allan_data/participants_graph_adjacency.msgpack')
mask = dfa.sum() != 0
dfa = dfa.loc[mask, mask]  # drop zero-columns
dfa.head()
qdf = pd.read_json('/lscr_paper/allan/allan_data/RGender_.json')
q = misc.QuestionCompleter(qdf)
f = misc.QuestionFilterer(qdf)
ua = loaders.Useralias()

qdf.index = qdf.index.map(lambda el: ua[el])


# Remove persons with more than 10 % null answers
qdf = qdf[(qdf.isna().mean() < 0.10).index[:5]]


# Unify participants in qdf and dfa
user_union = qdf.index.intersection(dfa.index)
dfa = dfa.filter(items=user_union, axis=0).filter(items=user_union, axis=1)
qdf = qdf.filter(items=user_union, axis=0)

assert dfa.shape[0] == dfa.shape[1]
assert (dfa.values[np.diag_indices_from(dfa.values)] == 0).all()
assert len(qdf.index) == len(dfa.index)
assert len(qdf.index.difference(dfa.index)) == 0


# Get alcohol-relarted question answers
alcohol_questions = f['alcohol.+__answer$']
alcohol_questions.head()


# Remove alcohol NaN-users from qdf
qdf = qdf[alcohol_questions.notnull().any(axis=1)]


# Find the persons with whom each person spends the most time
big5_questions = f['bfi_.+_answer$']
big5_questions.head()
n_persons = 35
simfnc = graph.cosSim
dct = dict()

for baseuser in dfa.index:
    u = dfa[baseuser]

    # Strip out persons not present in alcohol questions dataframe
    u = u[u.index.intersection(qdf.index)].sort_values(ascending=False)
    pers_homies = u[:n_persons]
    pers_homies.head()
    pers_control_names = qdf.index.difference(pers_homies.index)  # Remove names which is in pers_homies
    np.random.shuffle(pers_control_names.values)  # shuffle names
    pers_control_names = pers_control_names[:n_persons]  # choose n_persons names
    pers_control = dfa.loc[baseuser][pers_control_names]  # select columns in dfa
    pers_control.head()  # compute pers_control
    assert pers_homies.shape == pers_control.shape

    # Compute the similarity in the way they answered the alcohol related questions
    sim_alcohol_homies = compareDfUsers(baseuser, pers_homies.index, alcohol_questions)
    sim_alcohol_homies.name = 'alcohol_homies'
    sim_alcohol_control = compareDfUsers(baseuser, pers_control.index, alcohol_questions)
    sim_alcohol_control.name = 'alcohol_control'
    if PRINT:
        print("sim_alcohol_homies.head()", sim_alcohol_homies.head(), sep=':\n', end='\n\n')
        print("sim_alcohol_control.head()", sim_alcohol_control.head(), sep=':\n', end='\n\n')

    # Compute the similarity wrt. Big Five personality related questions
    sim_big5_homies = compareDfUsers(baseuser, pers_homies.index, big5_questions)
    sim_big5_homies.name = 'big5_homies'
    sim_big5_control = compareDfUsers(baseuser, pers_control.index, big5_questions)
    sim_big5_control.name = 'big5_control'
    if PRINT:
        print("sim_big5_homies.head()", sim_big5_homies.head(), sep=':\n', end='\n\n')
        print("sim_big5_control.head()", sim_big5_control.head(), sep=':\n', end='\n\n')

    # Compute similarity in persons that they hang out around
    # Compute the similarity wrt. Big Five personality related questions
    sim_people_homies = compareDfUsers(baseuser, pers_homies.index, dfa)
    sim_people_homies.name = 'people_homies'
    sim_people_control = compareDfUsers(baseuser, pers_control.index, dfa)
    sim_people_control.name = 'people_control'
    if PRINT:
        print("sim_people_homies.head()", sim_people_homies.head(), sep=':\n', end='\n\n')
        print("sim_people_control.head()", sim_people_control.head(), sep=':\n', end='\n\n')
    df_homies = pd.DataFrame([sim_alcohol_homies, sim_big5_homies, sim_people_homies]).T
    df_control = pd.DataFrame([sim_alcohol_control, sim_big5_control, sim_people_control]).T
    dct[baseuser] = (df_homies, df_control)
