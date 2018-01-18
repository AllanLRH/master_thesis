#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
from glob import glob
import pickle
import pandas as pd
from multiprocessing import Pool, cpu_count
import json
import collections


def loadPythonSyntaxFile(filepath):
    """Loads data stored in a text file, where each line is a python dict.

    Parameters
    ----------
    filepath : str
        Path of file to be read.

    Returns
    -------
    list
        Return a list containing all the dicts from the file.

    Raises
    ------
    FileNotFoundError
    If filepath doesn't point to a .txt-file.
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

    Parameters
    ----------
    user : str
        Name of user data folder
    datapath : str, optional
        Path to folder which contains user data folder.
    dataFilter : Iterable containing str, optional
        Only return certain datasets from a user. Allowed values are 'sms', 'question',
        'gps', 'bluetooth', 'screen', 'facebook' and 'call'.

    Returns
    -------
    dict
        User data in a dict, maps to None for missing data types.

    Raises
    ------
    ValueError
    If a wrong parameter is passed to dataFilter
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

    Parameters
    ----------
    user : str
        Name of user data folder
    datapath : str, optional
        Path to folder which contains user data folder.
    dataFilter : Iterable containing str, optional
        Only return certain datasets from a user. Allowed values are 'sms', 'question',
        'gps', 'bluetooth', 'screen', 'facebook' and 'call'.

    Returns
    -------
    dict
        User data in a dict, maps to None for missing data types.

    Raises
    ------
    ValueError
    If a wrong parameter is passed to dataFilter
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

    Parameters
    ----------
    userSpec : tuple
        Contains (username, useralias) or (username, useralias, dataFilter).
        See the documentation for loadUser2 (which uses theese arguments).
    n : None, optional
        Number of provessor cores to use when loading the users in parallel.
        Default is 16, but will fall back to number of cores minus 1 if 16 cores
        aren't avaiable.

    Returns
    -------
    dict
        Dictionary representation of all users. Can easily be converted to pandas DataFrame.
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

    Parameters
    ----------
    useralias : Useralias, optional
        An Useralias instance, may be None.
    filepath : str, optional
        Path to the phonenumbers.p pickle-file.

    Returns
    -------
    dict
        phoneID -> userID
    """
    with open(filepath, "rb") as fid:
        data = pickle.load(fid)
        if useralias is not None:
            return {k: useralias[v] for (k, v) in data.items()}
        return data


def getUserList(datapath='/lscr_paper/allan/data/Telefon/userfiles'):
    folders = (pth for pth in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, pth)))
    userFolders = (fld for fld in folders if (len(fld) == 30))
    valudUsernames = [fld for fld in userFolders if not (set(fld) - set('0123456789abcdef'))]
    return valudUsernames


class Useralias(object):
    """Class used to rename user from the human ureadable hash values.
    Works like a dict for lookups, but return a new (sequentially generetad) alias for
    unknown users, whereas previous look-up users aliases is saved, and returned when
    asked for.

    Attributes
    ----------
    formatStr : str
        String template for returned aliases. Must be compatible with
    i : int
        Holds value for the next user in the sequence
    userdct : dict
        Holds previously seen hash -> alias pairs the .format()-method
    """

    def __init__(self, formatStr="u{:04d}"):
        super(Useralias, self).__init__()
        self.formatStr = formatStr
        self._json_file_path = '/lscr_paper/allan/allan_data/user_aliases.json'
        if not os.path.isfile(self._json_file_path):
            raise FileNotFoundError("The file user_aliases.json could not be found!")
        with open(self._json_file_path) as fid:
            self.userdct = json.load(fid)
        self.i = len(self.userdct)

    def __setitem__(self, key, value):
        raise IndexError("The username aliases are frozen!")

    def __getitem__(self, key):
        return self.userdct[key]

    def lookup(self, alias):
        """Reverse lookup: given a useralias, return the username (hash-like string)

        Parameters
        ----------
        alias : str
            useralias, default on the form u0001, u0123, u0435, u1023 and so on....

        Returns
        -------
        dict
            Original username (hash-like), that is, the name of the user data-folder.
        """
        if len(self.reversed) != len(self.userdct):
            self.reversed = {v: k for (k, v) in self.userdct.items()}
        return self.reversed[alias]


def dict2DataFrame(dct, useralias):
    """Convert the dict-based output from loadUser to a DataFrame

    Parameters
    ----------
    dct : dict
        A single communication typy dict from a user, as returned by loadUser.
    useralias : Useralias
        A Useralias-instance or a dict mapping hashlike usernames to a human
        readable format.

    Returns
    -------
    DataFrame
        Pandas DataFrame with 'user' and 'comtype' as the index, and columns:
        - body: Hash of SMS body, NaN for calls
        - duration: Duration of call, NaN for SMS
        - number: Hash of recieving number
        - timestamp: Datetime for when the event occured
        - weekday: Weekday on which the event occured
        - hour: Hour of the day when the envent occured
    """
    df = pd.DataFrame(dct)
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

    Parameters
    ----------
    dct : dict
        Dict of users like the one returned by loadUserParallel.
    useralias : Useralias
        An Useralias-instance.
    n : None, optional
        Number of CPU cores to use.
        Default is 16, but will fall back to number of cores minus 1 if 16 cores
        aren't avaiable.

    Returns
    -------
    DataFrame
        All the users from dct as a DataFrame, as the one returned from dict2DataFrame.
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


def quickSaveHdf5(filepath, *data):
    """Given a path to a HDF5 file, save data to file.
    Data is converted to a pandas Series or DataFrame if it's too high dimmensioal for
    a Series to be used.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file.
    *data
        Data to be saved. Can either be a dict or two iterables containing keys and the
        data to save.

    Raises
    ------
    ValueError
    If *data input is not correct.
    """

    def saveElement(key, data, store):
        """Save 'data' to HDF5 'store' with 'key'.
        Tries to convert everything to a Pandas Series or DataFrame if it's too
        high dimmensioal.

        Parameters
        ----------
        key : str
            Key to save data under.
        data
            Data to save.
        store : HDFStore
            The HDFStore in which the file is saved.
        """
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            try:
                data = pd.Series(data)
                print("Attention! The data {} was convert to a Series".format(key))
            except Exception:
                data = pd.DataFrame(data)
                print("Attention! The data {} was convert to a DataFrame".format(key))
        store[key] = data

    with pd.HDFStore(filepath, mode='w') as store:
        if len(data) == 1 and isinstance(data[0], dict):
            for key, value in data[0].items():
                saveElement(key, value, store)
        # It's two iterables of equal length, containing the data names and the data
        elif all( len(data) == 2,  # noqa
                  all([isinstance(data[i], collections.Iterable) for i in range(2)]),
                  len(data[0] == len(data[1])) ):  # noqa
            for key, value in data:
                saveElement(key, value, store)
        else:
            raise ValueError("Incorrect data input.")


def loadUserBluetooth(userhash, useralias):

    def _load_bluetooth(user):
        try:
            with open(f'/lscr_paper/allan/telephone/{user}/bluetooth.json') as fid:
                data = json.load(fid)
            return data['bluetooth']
        except Exception as err:
            print(f"Couldn't read user {user}")
            raise err

    btdata = _load_bluetooth(userhash)
    df = pd.DataFrame(btdata)
    df['timestamp'] = df.timestamp.astype('datetime64[s]')
    df = df[df.bt_mac != '-1']
    # ua = loaders.Useralias()
    df['scanned_user'] = df.scanned_user.map(useralias.userdct)
    df['user'] = df.user.map(useralias.userdct)
    df = df.set_index('timestamp')
    df = df[df.index.year >= 2013]
    return df
