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
PRINT_PROGRESS = True


# @jit()
def compareDfUsers(baseuser, peers, df, simfnc):
    # Compute the similarity in the way they answered the questions
    dct = dict()
    for i in range(len(peers)):
        dct[(baseuser, peers[i])] = simfnc(df.loc[baseuser], df.loc[peers[i]])
    sim = pd.Series(dct).sort_values(ascending=False)
    return sim


dfa  = pd.read_msgpack('/lscr_paper/allan/allan_data/participants_graph_adjacency.msgpack')
mask = dfa.sum() != 0
dfa  = dfa.loc[mask, mask]  # drop zero-columns
if PRINT: print(dfa.head())  # noqa
qdf  = pd.read_json('/lscr_paper/allan/allan_data/RGender_.json')
f    = misc.QuestionFilterer(qdf)
ua   = loaders.Useralias()

qdf.index = qdf.index.map(lambda el: ua[el])

# Remove persons with more than 10 % null answers, and keep only the __answer-columns of the questions
qdf = f['__answer$']
qdf = qdf.loc[:, qdf.isna().mean() < 0.10]
# Unify participants in qdf and dfa
user_union = qdf.index.intersection(dfa.index)
dfa = dfa.filter(items=user_union, axis=0).filter(items=user_union, axis=1)  # remove both rows and columns!
qdf = qdf.filter(items=user_union, axis=0)
# Reinstantiate f since qdf have changed
f   = misc.QuestionFilterer(qdf)

# Sanity checks
assert dfa.shape[0] == dfa.shape[1]
assert (dfa.values[np.diag_indices_from(dfa.values)] == 0).all()
assert len(qdf.index) == len(dfa.index)
assert len(qdf.index.difference(dfa.index)) == 0

n_persons = 35  # Number of persons considered "close" to the user
simfnc    = [('cosSim', graph.cosSim), ('normDotSim', graph.normDotSim)]
qdct      = dict()

question_categories = ['bfi', 'loneliness', 'narcissism', 'symptoms', 'locus',
                       'mdi', 'homophily', 'selfesteem', 'panas', 'stress', 'alcohol']
if PRINT: big5_questions.head()  # noqa

for ui, baseuser in enumerate(dfa.index):
    u = dfa[baseuser]  # Get user entry in dfa
    if PRINT_PROGRESS and (ui % 40 == 0):
        print(f"Processing user {baseuser} ({ui}/{len(dfa.index)})")

    # ****************************************************************************
    # *               Pick homies and random control group for user              *
    # ****************************************************************************
    # Strip out persons not present in alcohol questions dataframe
    u = u[u.index.intersection(qdf.index)].sort_values(ascending=False)
    user_homies = u[:n_persons]  # select the n_persons most popular persons
    if PRINT:
        print(user_homies.head())
    user_control_names = u[n_persons:]
    user_control       = user_control_names[np.random.permutation(len(user_control_names))[:n_persons]]
    if PRINT:
        print(user_control.head())  # compute user_control
    assert user_homies.shape == user_control.shape

    # ****************************************************************************
    # *                 Do calculations for individual questions                 *
    # ****************************************************************************
    for question_cat in question_categories:
        questions = f[question_cat]
        # Compute the similarity wrt. personality related questions
        for fnc_name, fnc in simfnc:
            sim_homies       = compareDfUsers(baseuser, user_homies.index, questions, fnc).reset_index(drop=True)
            sim_homies.name  = 'homies_' + fnc_name
            sim_control      = compareDfUsers(baseuser, user_control.index, questions, fnc).reset_index(drop=True)
            sim_control.name = 'control_' + fnc_name
            qdct[(baseuser, question_cat, 'homies', fnc_name)] = sim_homies
            qdct[(baseuser, question_cat, 'control', fnc_name)] = sim_control
        if PRINT:
            print("sim_homies.head()", sim_homies.head(), sep=':\n', end='\n\n')
            print("sim_control.head()", sim_control.head(), sep=':\n', end='\n\n')

    # ****************************************************************************
    # *       Do the calculation for the people that the users hang around       *
    # ****************************************************************************
    # Compute similarity in persons that they hang out around
    for fnc_name, fnc in simfnc:
        sim_homies       = compareDfUsers(baseuser, user_homies.index, dfa, fnc).reset_index(drop=True)
        sim_homies.name  = 'homies_' + fnc_name
        sim_control      = compareDfUsers(baseuser, user_control.index, dfa, fnc).reset_index(drop=True)
        sim_control.name = 'control_' + fnc_name
        qdct[(baseuser, 'people', 'homies', fnc_name)] = sim_homies
        qdct[(baseuser, 'people', 'control', fnc_name)] = sim_control
    if PRINT:
        print("sim_homies.head()", sim_homies.head(), sep=':\n', end='\n\n')
        print("sim_control.head()", sim_control.head(), sep=':\n', end='\n\n')
