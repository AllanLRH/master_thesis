import numpy as np
import pandas as pd


def check_question_scales(res, debug=False):
    response_labels = dict()
    ans  = res.index.copy()
    resp = res.iloc[:, 0].copy()  # first and only column, copy is created by iloc
    numeric_responses = resp.str.contains(r'\d').all()
    response_labels['numeric_responses'] = numeric_responses
    translation_dict = {'meget uenig': '0.0',
                        'uenig': '1.0',
                        'hverken enig eller uenig': '2.0',
                        'enig': '3.0',
                        'meget enig': '4.0',
                        'aldrig': '0.0',
                        'sjældent': '1.0',
                        'af og til': '2.0',
                        'ofte': '3.0',
                        'altid': '4.0'}
    mask = resp.str.contains('ved ikke', case=False)
    response_labels['ved_ikke'] = mask.sum()
    # The response variables seem to be numeric in nature, and are thus inherently ordered
    if numeric_responses:
        # First lower the strings, then use Pandas replace, not Pandas-string replace
        resp = resp.str.lower().replace(translation_dict)
        # Assign "ved ikke" answers to -1
        if mask.sum() == 1:
            resp[mask] = '-1'
        # Extract numbers, including sign, but negating "-" after a number (indicating a range of numbers)
        resp_extract_raw  = resp.str.extractall(r'((?<!\d)-?\d+)')
        resp_extract      = resp_extract_raw.astype(float)
        resp_extract_mean = resp_extract.mean(level=0)[0]  # DataFrame with just one column
        if debug:
            print('ans'              , ans              , sep='\n\n' , end='\n'*5)  # noqa
            print('resp'             , resp             , sep='\n\n' , end='\n'*5)  # noqa
            print('ans_extract_raw'  , ans_extract_raw  , sep='\n\n' , end='\n'*5)  # noqa
            print('ans_extract'      , ans_extract      , sep='\n\n' , end='\n'*5)  # noqa
            print('ans_extract_mean' , ans_extract_mean , sep='\n\n' , end='\n'*5)  # noqa
        if ans.shape != resp_extract_mean.shape:
            response_labels['ans_resp_shape_mismatch'] = True
        else:
            is_sorted = (ans.argsort() == resp_extract_mean.argsort()).all()
            response_labels['resp_is_sorted'] = is_sorted
    return response_labels
