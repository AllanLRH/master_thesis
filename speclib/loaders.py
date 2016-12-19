#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
from glob import glob
import pickle
from multiprocessing import Pool


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


def loadUser(user, datapath='/lscr_paper/allan/data/Telefon/userfiles',
             dataFilter=('sms', 'question', 'gps', 'bluetooth', 'screen', 'facebook', 'call')):
    """Loads a users data as dict.

    Args:
        user (str): Name of user data folder
        datapath (str, optional): Path to folder which contains user data folder.
        dataFilter (Iterable containing str, optional): Only return certain datasets from
            a user. Allowed values are 'sms', 'question', 'gps', 'bluetooth', 'screen',
            'facebook' and 'call'.

    Returns:
        dict: User data in a dict, maps to None for missing data types.

    Raises:
        ValueError: If a wrong parameter is passed to dataFilter
    """
    # Turn "/foo/bar/baz/gps_log.txt" -> "gps"
    _filepath2dictkey = lambda el: el.rsplit(os.path.sep, maxsplit=1)[1].split("_")[0]
    userPath = os.path.join(datapath, user)
    userDict = dict()
    loadedFilesSet = set()
    datafileList = glob(os.path.join(userPath, "*.txt"))

    # Check that dataFilter arguments are valid, raise ValueError if they aren't
    validFilterSet = {'sms', 'question', 'gps', 'bluetooth', 'screen', 'facebook', 'call'}
    if any({el not in validFilterSet for el in dataFilter}):
        raise ValueError("Invalied filter argument provided. Allowed values are %r"
                         % validFilterSet)

    for filepath in datafileList:
        dictkey = _filepath2dictkey(filepath)
        if dictkey not in dataFilter:
            continue
        userDict[dictkey] = loadPythonSyntaxFile(filepath)
        loadedFilesSet.add(dictkey)
    for missingFileKey in (set(dataFilter) - loadedFilesSet):
        userDict[missingFileKey] = None
    return userDict if userDict else None  # Return None if there's no contents in userDict


def _loadUserHandler(userSpec):
    if len(userSpec) == 2:
        user, alias = userSpec
        return (alias, loadUser(user))
    else:
        user, alias, dataFilter = userSpec
        return (alias, loadUser(user, dataFilter=dataFilter))


def loadUserParallel(userSpec, n=16):
    pool = Pool(n)
    users = None
    try:
        users = pool.map(_loadUserHandler, userSpec)
    finally:
        pool.terminate()
    return dict(users)


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


class Useralias(object):
    """Class used to rename user from the human ureadable hash values.
    Works like a dict for lookups, but return a new (sequentially generetad) alias for
    unknown users, whereas previous look-up users aliases is saved, and returned when
    asked for.

    Attributes:
        formatStr (str): String template for returned aliases. Must be compatible with
        the .format()-method
        i (int): Holds value for the next user in the sequence
        userdct (dict): Holds previously seen hash -> alias pairs
    """

    def __init__(self, formatStr="u{:04d}"):
        super(Useralias, self).__init__()
        self.formatStr = formatStr
        self.i = 0
        self.userdct = dict()
        self.reversed = dict()

    def __setitem__(self, key, value):
        self.userdct[key] = value

    def __getitem__(self, key):
        if key not in self.userdct:
            self.i += 1
            self.userdct[key] = self.formatStr.format(self.i)
        return self.userdct[key]

    def lookup(self, alias):
        """Reverse lookup: given a useralias, return the username (hash-like string)

        Args:
            alias (str): useralias, default on the form u0001, u0123, u0435, u1023 and so on....

        Returns:
            TYPE: Original username (hash-like), that is, the name of the user data-folder.
        """
        if len(self.reversed) != len(self.userDict):
            self.reversed = {v: k for (k, v) in self.userDict.items()}
        return reversed[alias]


if __name__ == '__main__':
    datapath = '/lscr_paper/allan/data/Telefon/userfiles'
    userList = [el for el in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, el))]
    print(userList[0])
    data = loadUser(userList[0])


