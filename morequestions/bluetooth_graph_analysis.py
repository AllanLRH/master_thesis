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

from speclib import misc, plotting, loaders, graph  # noqa


dfa = pd.read_msgpack('../../allan_data/participants_graph_adjacency.msgpack')
mask = dfa.sum() != 0
dfa = dfa.loc[mask, mask]  # drop zero-columns
dfa.head()
qdf = pd.read_json('../../allan_data/RGender_.json')
q = misc.QuestionCompleter(qdf)
f = misc.QuestionFilterer(qdf)
ua = loaders.Useralias()

qdf.index = qdf.index.map(lambda el: ua[el])


# Remove persons with more than 10 % null answers
qdf = qdf[(qdf.isna().mean() < 0.10).index]


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
u = dfa['u0001']


# Strip out persons not present in alcohol questions dataframe
u = u[u.index.intersection(qdf.index)].sort_values(ascending=False)
pers_homies = u[:n_persons]
pers_homies.head()
pers_control_names = qdf.index.difference(pers_homies.index)  # Remove names which is in pers_homies
np.random.shuffle(pers_control_names.values)  # shuffle names
pers_control_names = pers_control_names[:n_persons]  # choose n_persons names
pers_control = dfa.loc['u0001'][pers_control_names]  # select columns in dfa
pers_control.head()  # compute pers_control
assert pers_homies.shape == pers_control.shape


# Compute the similarity in the way they answered the questions
dct_alcohol = dict()
for p in pers_homies.index:
    dct_alcohol[('u0001', p)] = simfnc(alcohol_questions.loc['u0001'], alcohol_questions.loc[p])
sim_alcohol_homies = pd.Series(dct_alcohol).sort_values(ascending=False)

dct_alcohol = dict()
for p in pers_control.index:
    dct_alcohol[('u0001', p)] = simfnc(alcohol_questions.loc['u0001'], alcohol_questions.loc[p])
sim_alcohol_control = pd.Series(dct_alcohol).sort_values(ascending=False)
print(sim_alcohol_homies.head())
print(sim_alcohol_control.head())


# Compute the similarity wrt. Big Five personality related questions
dct_big5 = dict()
for p in pers_homies.index:
    dct_big5[('u0001', p)] = simfnc(big5_questions.loc['u0001'], big5_questions.loc[p])
sim_big5_homies = pd.Series(dct_big5).sort_values(ascending=False)

dct_big5 = dict()
for p in pers_control.index:
    dct_big5[('u0001', p)] = simfnc(big5_questions.loc['u0001'], big5_questions.loc[p])
sim_big5_control = pd.Series(dct_big5).sort_values(ascending=False)
print(sim_big5_homies.head())
print(sim_big5_control.head())


# Compute similarity in persons that they hang out around
dct_people = dict()
for p in pers_homies.index:
    dct_people[('u0001', p)] = simfnc(dfa['u0001'], dfa[p])
sim_people_homies = pd.Series(dct_people).sort_values(ascending=False)
dct_people = dict()
for p in pers_control.index:
    dct_people[('u0001', p)] = simfnc(dfa['u0001'], dfa[p])
sim_people_control = pd.Series(dct_people).sort_values(ascending=False)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
ax1.plot(sim_alcohol_homies.values, label="Alcohol, homies")
ax1.plot(sim_alcohol_control.values, label="Alcohol, control")
ax1.legend(loc='best')
ax2.plot(sim_big5_homies.values, label="Big 5, homies")
ax2.plot(sim_big5_control.values, label="Big 5, control")
ax2.legend(loc='best')
ax3.plot(sim_people_homies.values, label="People, homies")
ax3.plot(sim_people_control.values, label="People, control")
ax3.legend(loc='best')
print("sim_alcohol_homies", sim_alcohol_homies.values.shape, sep=':  ')
print("sim_alcohol_control", sim_alcohol_control.values.shape, sep=':  ')
print("sim_big5_homies", sim_big5_homies.values.shape, sep=':  ')
print("sim_big5_control", sim_big5_control.values.shape, sep=':  ')
print("sim_people_homies", sim_people_homies.values.shape, sep=':  ')
print("sim_people_control", sim_people_control.values.shape, sep=':  ')

