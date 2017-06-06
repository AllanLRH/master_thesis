#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import sys
sys.path.append(os.path.abspath(".."))

from speclib.loaders import loadUser, loadPythonSyntaxFile

dataPath = r"/lscr_paper/allan/data/Telefon/userfiles"
userList = os.listdir(dataPath)

user = userList[121]

user = loadUser(user, dataFilter=("call", "gps"))
for k, v in user.items():
    print(k, len(v), sep=":  ")

