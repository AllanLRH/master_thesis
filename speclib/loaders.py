#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import sys
import json
import codecs
import pickle


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

    def _cleanLine(ln):
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
        for i, line in enumerate(fid):
            lineDecoded = decodeLine(line)
            # yield ujson.loads(_cleanLine(lineDecoded))
            try:
                cleanedLine = _cleanLine(lineDecoded)
                yield json.loads(cleanedLine)
            except json.decoder.JSONDecodeError as err:
                print(err, file=sys.stderr)
                print("%s  &d:\t%r" % (filepath, i, _cleanLine(lineDecoded)), file=sys.stderr)


def loadUser(user, datapath='/lscr_paper/allan/data/Telefon/userfiles', dataFilter=None):
    """Loads a users data as dict.

    Args:
        user (str): Name of user data folder
        datapath (str, optional): Path to folder which contains user data folder.
        dataFilter (Iterable containing str, optional): Only return certain datasets from
            a user. Allowed values are 'sms', 'question', 'gps', 'bluetooth', 'screen',
            'facebook' and 'call'.

    Returns:
        TYPE: dict

    Raises:
        ValueError: If a wrong parameter is passed to dataFilter
    """
    userPath = os.path.join(datapath, user)

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
        datafileList = [el for el in os.listdir(userPath) if el.lower().endswith(".txt")]

    userDict = dict()
    for filename in datafileList:
        dataType = filename.split('_')[0]
        datafilePath = os.path.join(userPath, filename)
        userDict[dataType] = list(loadAndersJson(datafilePath))
    return userDict


def loadUserPhonenumberDict(filepath="/lscr_paper/allan/phonenumbers.p"):
    """Loads the dictionary which relates a phone number to a user.
    Format is phoneID -> userID

    Args:
        filepath (str, optional): Path to the phonenumbers.p pickle-file.

    Returns:
        dict: phoneID -> userID
    """
    with open(filepath, "rb") as fid:
        data = pickle.load(fid)
        return data


def getUserList(datapath='/lscr_paper/allan/data/Telefon/userfiles'):
    return os.listdir(datapath)


if __name__ == '__main__':
    datapath = '/lscr_paper/allan/data/Telefon/userfiles'
    userList = [el for el in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, el))]
    print(userList[0])
    data = loadUser(userList[0])
