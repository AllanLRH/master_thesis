#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import sys
import json
import logging
sys.path.append(os.path.abspath(".."))
from speclib.loaders import getUserList, loadUser
from speclib.pushbulletNotifier import JobNotification


###########
# Logging #
###########
log = logging.getLogger('convertDataToJson.py')
log.setLevel(logging.DEBUG)
fh = logging.FileHandler('convertDataToJson.py.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
log.addHandler(fh)


def saveDict(key, data, userDir):
    dct = {key: data}
    filepath = os.path.join(userDir, key + '.json')
    with open(filepath, 'w') as fid:
        json.dump(dct, fid, check_circular=False)
        log.info("Saved the file {}".format(filepath))


def main():
    saveDataDir = '/lscr_paper/allan/telephone'
    # userlist = getUserList()
    userlist = ["28b76d7b7879d364321f164df5169f"]

    for i, user in enumerate(userlist):
        try:
            userDir = os.path.join(saveDataDir, user)
            os.mkdir(userDir)
        except Exception as e:
            log.error("An error occured when creating a folder for user {} ({}). Skipping to next user: \n\n{}".format(user, i, str(e)))
            continue
        for k, v in loadUser(user).items():
            saveDict(k, v, userDir)


if __name__ == '__main__':
    jn = JobNotification(devices='phone')
    try:
        main()
    except Exception as e:
        jn.send(e)
        print(e)
