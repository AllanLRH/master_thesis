#!/usr/bin/env python
# -*- coding: utf8 -*-

import logging
logging.config.fileConfig('logging_client.ini', defaults={"root_handler": "multilogClientHandler"})
import multiprocessing
import numpy as np


def worker_function(*args):
    created = multiprocessing.Process()  # noqa
    current = multiprocessing.current_process()  # noqa
    a, b, c, d = args
    logging.info("fWorker {current.name}:\t" + f"a = {a}")
    logging.info("fWorker {current.name}:\t" + f"b = {b}")
    logging.info("fWorker {current.name}:\t" + f"c = {c}")
    logging.info("fWorker {current.name}:\t" + f"d = {d}")
    if c == 0:
        logging.critical("fWorker {current.name}:\t" + "c was set to 1!")
        c = 1
    res = (a + b)/c - d
    logging.info("fWorker {current.name}:\t" + res)
    return res


def main():
    args = np.random.randint(0, 10, (4, 100))
    try:
        pool = multiprocessing.Pool(4)
        res = pool.map(worker_function, args)  # noqa
    except Exception as err:
        print(err)
    finally:
        pool.close()
