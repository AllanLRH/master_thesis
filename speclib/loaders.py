#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
from glob import glob
import pickle
import pandas as pd
from multiprocessing import Pool, cpu_count
import json


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


def loadUser2(user, datapath='/lscr_paper/allan/telephone',
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
    userPath = os.path.join(datapath, user)
    userDict = dict()
    loadedFilekeySet = set()
    datafileList = os.listdir(userPath)

    # Check that dataFilter arguments are valid, raise ValueError if they aren't
    validFilterSet = {'sms', 'question', 'gps', 'bluetooth', 'screen', 'facebook', 'call'}
    if any({el not in validFilterSet for el in dataFilter}):
        raise ValueError("Invalied filter argument provided. Allowed values are %r"
                         % validFilterSet)

    for filename in datafileList:
        filekey = filename.split('.')[0]
        if filekey not in dataFilter:
            continue
        with open(os.path.join(userPath, filename)) as fid:
            userDict.update(json.load(fid))
        loadedFilekeySet.add(filekey)
    for missingFileKey in (set(dataFilter) - loadedFilekeySet):
        userDict[missingFileKey] = None
    return userDict if userDict else None  # Return None if there's no contents in userDict


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
    loadedFilekeySet = set()
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
        loadedFilekeySet.add(dictkey)
    for missingFileKey in (set(dataFilter) - loadedFilekeySet):
        userDict[missingFileKey] = None
    return userDict if userDict else None  # Return None if there's no contents in userDict


def _loadUserHandler(userSpec):
    """
    Helper function for loadUserParallel.
    """
    if len(userSpec) == 2:
        user, alias = userSpec
        return (alias, loadUser2(user))
    else:
        user, alias, dataFilter = userSpec
        return (alias, loadUser2(user, dataFilter=dataFilter))


def loadUserParallel(userSpec, n=None):
    """Loads users in parallel.

    Args:
        userSpec (tuple): Contains (username, useralias) or (username, useralias, dataFilter).
                          See the documentation for loadUser2 (which uses theese arguments).
        n (None, optional): Number of provessor cores to use when loading the users in parallel.
                            Default is 16, but will fall back to number of cores minus 1 if 16
                            cores aren't avaiable.

    Returns:
        dict: Dictionary representation of all users. Can easily be converted to pandas DataFrame.
    """
    if n is None:
        n = 16 if 16 < cpu_count() else cpu_count() - 1
    pool = Pool(n)
    users = None
    try:
        users = pool.map(_loadUserHandler, userSpec)
    finally:
        pool.terminate()
    return dict(users)


def loadUserPhonenumberDict(useralias=None, filepath="/lscr_paper/allan/phonenumbers.p"):
    """Loads the dictionary which relates a phone number to a user.
    Format is phoneID -> userID

    Args:
        useralias (Useralias, optional): An Useralias instance, may be None.
        filepath (str, optional): Path to the phonenumbers.p pickle-file.

    Returns:
        dict: phoneID -> userID
    """
    with open(filepath, "rb") as fid:
        data = pickle.load(fid)
        if useralias is not None:
            return {k: useralias[v] for (k, v) in data.items()}
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


def dict2DataFrame(dct, useralias):
    """Convert the dict-based output from loadUser to a DataFrame

    Args:
        dct (dict): A single communication typy dict from a user, as returned by loadUser.
        useralias (Useralias): A Useralias-instance or a dict mapping hashlike
                               usernames to a human readable format.

    Returns:
        DataFrame: Pandas DataFrame with 'user' and 'comtype' as the index, and columns:
        - body: Hash of SMS body, NaN for calls
        - duration: Duration of call, NaN for SMS
        - number: Hash of recieving number
        - timestamp: Datetime for when the event occured
        - weekday: Weekday on which the event occured
        - hour: Hour of the day when the envent occured
    """
    df = pd.DataFrame(dct)
    if 'id' in df.columns:
        df = df.drop("id", axis=1)
    if 'type' in df.columns:
        df = df.drop('type', axis=1)
    df.rename(columns={'timestamp': 'timeint'}, inplace=True)
    df['timestamp'] = df.timeint.astype('datetime64[s]')  # convert Unix integer timestamps to datetime-objects
    df["weekday"] = df.timestamp.dt.weekday  # Add weekday column
    df["hour"] = df.timestamp.dt.hour  # Add hour column
    df.user = df.user.apply(lambda x: useralias[x])  # convert hex-usernames to enumerated usernames
    # Set comtype and rename 'address' to 'number' for datatype 'sms'
    if 'address' in df.columns:
        df["comtype"] = "sms"  # create the column "comtype", set it to "sms"
        df.rename(columns={'address': 'number'}, inplace=True)
    else:
        df["comtype"] = "call"
    df = df.set_index(["user", "comtype"])  # move columns "user" and "comtype" to (multi)index
    return df


def _user2DataFrameHandler(args):
    """Helper function for users2DataFrame"""
    return dict2DataFrame(*args)


def users2DataFrame(dct, useralias, n=None):
    """Convert a dict of users into a DataFrame using multiple CPU cores.

    Args:
        dct (dict): Dict of users like the one returned by loadUserParallel.
        useralias (Useralias): An Useralias-instance.
        n (None, optional): Number of CPU cores to use.
                            Default is 16, but will fall back to number of cores
                            minus 1 if 16 cores aren't avaiable.

    Returns:
        DataFrame: All the users from dct as a DataFrame, as the one returned
                   from dict2DataFrame.
    """
    if n is None:
        n = 16 if 16 < cpu_count() else cpu_count() - 1
    gen = ((comDct, useralias) for user in dct.values()
                      for comDct in user.values() if comDct)  # noqa
    try:
        pool = Pool(processes=n)
        call = pool.map_async(_user2DataFrameHandler, gen)
        call.wait()
        toConcatenate = call.get()
    finally:
        pool.terminate()
    df = pd.concat(toConcatenate)
    df.sortlevel(level=0, inplace=True)
    return df


if __name__ == '__main__':
    datapath = '/lscr_paper/allan/data/Telefon/userfiles'
    userList = [el for el in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, el))]
    print(userList[0])
    data = loadUser(userList[0])


