#!/usr/bin/env python
# -*- coding: utf8 -*-
import itertools


def gen_permutations(n):
    for p in itertools.permutations(list(range(n))):
        yield p


n = 4
for p in gen_permutations(n):
    print(p)

