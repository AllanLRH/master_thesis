#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import sys
import json
import codecs


def loadAndersJson(filepath):
    """Loads Anders' "json" data files.
       It does so by converting the strings to valid json, and subsequently interpreting
       it using a json library


    Args:
        filepath (str): Path to date file

    Yields:
        TYPE: dict

    Raises:
        FileNotFoundError: Raised if filepath doesn't point to a file.
    """
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
                cleanedLine = cleanLine(lineDecoded)
                yield json.loads(cleanedLine)
            except json.decoder.JSONDecodeError as err:
                print(err, file=sys.stderr)
                print(repr(cleanLine(lineDecoded)), file=sys.stderr)


def loadUser(user, dataPath='/lscr_paper/allan/data/Telefon/userfiles', dataFilter=None):
    """Loads a users data as dict.

    Args:
        user (str): Name of user data folder
        dataPath (str, optional): Path to folder which contains user data folder.
        dataFilter (Iterable containing str, optional): Only return certain datasets from
            a user. Allowed values are 'sms', 'question', 'gps', 'bluetooth', 'screen',
            'facebook' and 'call'.

    Returns:
        TYPE: dict

    Raises:
        ValueError: If a wrong parameter is passed to dataFilter
    """
    userPath = os.path.join(dataPath, user)

    # Relating to dataFilter argument...
    if dataFilter is not None:  # Not all data files in the user folder should be loaded
        # Check that dataFilter arguments are valid, raise ValueError if they aren't
        validFilterSet = {'sms', 'question', 'gps', 'bluetooth', 'screen', 'facebook', 'call'}
        if any({el not in validFilterSet for el in dataFilter}):
            raise ValueError("Invalied filter argument provided. Allowed values are %r"
                             % validFilterSet)
        # Filter data files in user folder according to dataFilter
        datafileList = [el for el in os.listdir(userPath) if el.split("_")[0] in dataFilter]
    else:  # If no dataFilter is set, use all avaiable data files in user folder
        datafileList = os.listdir(userPath)

    userDict = dict()
    for filename in datafileList:
        dataType = filename.split('_')[0]
        datafilePath = os.path.join(userPath, filename)
        userDict[dataType] = list(loadAndersJson(datafilePath))
    return userDict


if __name__ == '__main__':
    datapath = '/lscr_paper/allan/data/Telefon/userfiles'
    userList = [el for el in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, el))]
    print(userList[0])
    data = loadUser(userList[0])
