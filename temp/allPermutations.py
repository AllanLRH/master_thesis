#!/usr/bin/env python
# -*- coding: utf8 -*-

from oset import oset


def gen_permutations(n):
    seen = set()
    for i in range(n):
        prm = list()
        for j in range(n):
            if i != j:
                prm.append((i, j))
                seen.add(tuple(sorted([i, j])))
                yield prm


n = 4
seen = set()
for p in gen_permutations(n):
    print(p)
    seen.add(frozenset(p))

print(len(list(gen_permutations(n))))

print(len(seen))
