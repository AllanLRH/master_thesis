#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
import pandas as pd


def check_question_scales(res, debug=False):
    ans = res.index
    resp = res.iloc[:, 0]
    translation_dict = {'meget uenig': '0.0',
                        'uenig': '1.0',
                        'hverken enig eller uenig': '2.0',
                        'enig': '3.0',
                        'meget enig': '4.0',
                        'altid': '4.0',
                        'ofte': '3.0',
                        'af og til': '2.0',
                        'sjældent': '1.0',
                        'aldrig': '0.0'}
    # First lower the strings, then use Pandas replace, not Pandas-string replace
    resp = resp.str.lower().replace(translation_dict)
    resp_extract_raw = resp.str.extractall(r'(\d+)')
    resp_extract = resp_extract_raw.astype(float)
    resp_extract_mean = resp_extract.mean(level=0)[0]  # DataFrame with just one column
    if debug:
        print('ans'              , ans              , sep='\n\n' , end='\n'*5)  # noqa
        print('resp'             , resp             , sep='\n\n' , end='\n'*5)  # noqa
        print('ans_extract_raw'  , ans_extract_raw  , sep='\n\n' , end='\n'*5)  # noqa
        print('ans_extract'      , ans_extract      , sep='\n\n' , end='\n'*5)  # noqa
        print('ans_extract_mean' , ans_extract_mean , sep='\n\n' , end='\n'*5)  # noqa
    if ans.shape != resp_extract_mean.shape:
        return False
    is_sorted = (ans.argsort() == resp_extract_mean.argsort()).all()
    return is_sorted
