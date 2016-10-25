#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import sys
import json
import codecs


def loadAndersJson(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError("The file {} doesn't seem to exist".format(filepath))

    def cleanLine(ln):
        ruleTup = (("u'", "'"),
                   ('"', '\\"'),
                   ("'", '"'),
                   ("True", "true"),
                   ("False", "false"),
                   ("None", "null"))
        for rule in ruleTup:
            ln = ln.replace(*rule)
        return ln

    def decodeLine(byteLine):
        lst = byteLine.split(b"\\n")
        toConcat = [codecs.escape_decode(line)[0].decode("iso-8859-15") for line in lst]
        return "\\n".join([el for el in toConcat if el])

    with open(filepath, "br") as fid:
        for line in fid:
            lineDecoded = decodeLine(line)
            # yield ujson.loads(cleanLine(lineDecoded))
            try:
                cleaned = cleanLine(lineDecoded)
                yield json.loads(cleaned)
            except json.decoder.JSONDecodeError as err:
                print(err, file=sys.stderr)
                print(repr(cleanLine(lineDecoded)), file=sys.stderr)


def loadUser(user, dataPath='/lscr_paper/allan/data/Telefon/userfiles'):
    userPath = os.path.join(dataPath, user)
    datafileList = os.listdir(userPath)
    userDict = dict()
    for filename in datafileList:
        print(filename)
        dataType = filename.split('_')[0]
        datafilePath = os.path.join(userPath, filename)
        print(datafilePath)
        userDict[dataType] = list(loadAndersJson(datafilePath))
    return userDict


if __name__ == '__main__':
    datapath = '/lscr_paper/allan/data/Telefon/userfiles'
    userList = [el for el in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, el))]
    print(userList[0])
    data = loadUser(userList[0])
