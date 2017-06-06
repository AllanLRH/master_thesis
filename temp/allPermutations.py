#!/usr/bin/env python
# -*- coding: utf8 -*-

from copy import deepcopy as copy


n = 4

prm_lst = list()
for i in range(n):
    prm = list()
    for j in range(n):
        if i != j:
            prm.append((i, j))
            prm_lst.append(copy(prm))

for p in prm_lst:
    print(p)

