#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import json

userpath = '/lscr_paper/allan/telephone'
savepath = '/lscr_paper/allan/allan_data/user_aliases.json'
users = [fld for fld in sorted(os.listdir(userpath)) if (os.path.isdir(os.path.join(userpath, fld)) and len(fld) == 30)]
userdict = {user: "u{:04d}".format(i) for (i, user) in enumerate(users, start=1)}
# print(*userdict.items(), sep='\n')
with open(savepath, 'w') as fid:
    json.dump(userdict, fid)
    print("Generated and saved {} user aliases to `{}` .".format(len(userdict), savepath))
