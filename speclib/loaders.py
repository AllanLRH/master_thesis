#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
from glob import glob
import pickle


def loadPythonSyntaxFile(filepath):
    """Loads data stored in a text file, where each line is a python dict.

    Args:
        filepath (str): Path of file to be read

    Returns:
        list: Return a list containing all the dicts from the file.

    Raises:
        FileNotFoundError: If filepath doesn't point to a .txt-file.
    """
    if not os.path.isfile(filepath) or not filepath.lower().endswith(".txt"):
        raise FileNotFoundError("The file {} doesn't seem to exist".format(filepath))
    localNamespace = dict()
    with open(filepath) as fid:
        cl0 = fid.read()
        cl1 = cl0.replace("\n{", ",\n{")  # Add "," to seperate dicts
        cl2 = cl1.strip()
        cl3 = "data = [" + cl2 + "]"  # Put all lines into a list.
        # set_trace()
        exec(cl3, localNamespace)  # Execute the python command
        return localNamespace["data"]  # Return values from localNamespace-dict


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
    # Turn "/foo/bar/baz/gps_log.txt" -> "gps"
    _filepath2dictkey = lambda el: el.rsplit("/", maxsplit=1)[1].split("_")[0]
    userPath = os.path.join(datapath, user)

    # Relating to dataFilter argument...
    datafileList = glob(os.path.join(userPath, "*.txt"))
    if dataFilter is not None:  # Not all data files in the user folder should be loaded
        # Check that dataFilter arguments are valid, raise ValueError if they aren't
        validFilterSet = {'sms', 'question', 'gps', 'bluetooth', 'screen', 'facebook', 'call'}
        if any({el not in validFilterSet for el in dataFilter}):
            raise ValueError("Invalied filter argument provided. Allowed values are %r"
                             % validFilterSet)
        # Filter data files in user folder according to dataFilter
        datafileList = [el for el in datafileList if _filepath2dictkey(el) in dataFilter]

    userDict = dict()
    for filepath in datafileList:
        dictkey = _filepath2dictkey(filepath)
        userDict[dictkey] = loadPythonSyntaxFile(filepath)
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
