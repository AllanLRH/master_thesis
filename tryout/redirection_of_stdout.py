#!/usr/bin/env python
# -*- coding: utf8 -*-


import sys
from contextlib import contextmanager
from io import StringIO


@contextmanager
def redirector(stream):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = stream
    sys.stderr = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


f = StringIO()
with redirector(f):
    print("this in stdout inside a with block")
    print("this in stderr inside a with block", file=sys.stderr)
print("this in stdout is not")
print("this in stderr is not", file=sys.stderr)
print(f"Captured:\n{f.getvalue()}")


class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")
        self.flush = sys.stderr.flush

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)








sys.stdout = Logger('stdout_redirect.txt')
print("This is stdout")
