#!/usr/bin/env python
# -*- coding: utf8 -*-

# import sys
import traceback


def bar(c, d, x):
    (x + c)/d


def foo(a, b, c, d):
    x = a * b
    bar(c, d, x)


try:
    foo(2, 3, 4, 0)
except Exception as err:
    # traceback.print_exc()
    print(*traceback.format_tb(err.__traceback__))
    # print('\n'.join(['🦄   ' + ln for ln in traceback.format_tb(err).split('\n') if ln]))
    # tb = sys.exc_info()[2]
    # print(err.with_traceback(tb), file=sys.stderr)

