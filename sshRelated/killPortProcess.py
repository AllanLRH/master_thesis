#!/lscr_paper/allan/miniconda3/envs/py36/bin/python

import sys
import psutil


if len(sys.argv) != 2 or not sys.argv[1].isdigit():
    print("Input error! Input not provided or it isn't a integer (port number)", file=sys.stderr)
    sys.exit(1)

port = int(sys.argv[1])

for nc in psutil.net_connections():
    if nc.laddr[1] == port:
        for pr in psutil.process_iter():
            if pr.pid == nc.pid:
                pr.kill()
