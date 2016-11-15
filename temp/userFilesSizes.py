from sh import wc
import pathlib
from multiprocessing.dummy import Pool
import json


nThreads = 2
dataPath = pathlib.Path("../../data/Telefon/userfiles/")
userFolders = [pt for pt in dataPath.iterdir() if pt.is_dir()]


def countLines(filepath):
    wcCall = wc("-l", str(filepath))
    cnt = int(wcCall.stdout.split()[0])
    return cnt


def countUserStats(userFolder):
    dct = {}
    for fl in userFolder.iterdir():
        if fl.is_file():
            dct[fl.name] = countLines(fl)
    return dct


def processUsers(userFolder):
    name = userFolder.name
    return {name: countUserStats(userFolder)}


if __name__ == '__main__':
    with Pool(nThreads) as pool:
        userSats = pool.map(processUsers, userFolders)
        with open("userDataCount.json", "w") as fid:
            json.dump(userSats, fid)
