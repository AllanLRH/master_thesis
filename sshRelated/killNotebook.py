#!/lscr_paper/allan/miniconda3/envs/py36/bin/python

import psutil
import sys


username = psutil.os.environ["USER"]


def isInputPort(inp):
    return inp.strip().isdigit()


def getNotebooks():
    # "jupyter-noteboo" is not an error, the "k" really is missing from the process name
    return [pr for pr in psutil.process_iter() if (pr.name() == "jupyter-noteboo" and pr.username() == username)]


def getPort(pr):
    port = [el.laddr[1] for el in psutil.net_connections() if el.pid == pr.pid][0]
    return port


def killNotebook(pr, port=None):
    if port:
        if getPort(pr) == int(port):
            pr.kill()
    else:
        pr.kill()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if not isInputPort(sys.argv[1]):
            print("Invalid port format", file=sys.stderr)
            sys.exit(1)
        for pr in getNotebooks():
            killNotebook(pr, sys.argv[1])
    else:
        for pr in getNotebooks():
            killNotebook(pr)
