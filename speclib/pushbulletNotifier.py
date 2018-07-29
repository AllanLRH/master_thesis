#!/usr/bin/env python
# -*- coding: utf8 -*-

from pushbullet import Pushbullet
import datetime
import inspect
import traceback
import logging
import os

dirname   = os.path.dirname(__file__)
logfolder = os.path.abspath(os.path.join(dirname, '../logs/'))
logging.basicConfig(filename=os.path.join(logfolder, 'pushbulletnotifier.log'),
                    level=logging.INFO,
                    format="%(asctime)s :: %(filename)s:%(lineno)s :: %(funcName)s() ::    %(message)s")
logger = logging.getLogger('pushbulletnotifier')

phoneName = "OnePlus ONEPLUS A6003"


class JobNotification(object):
    """Class to send push notification when a job finishes, either because of an error, or
    because it's completed.
    """

    def __init__(self, devices="phone"):
        super(JobNotification, self).__init__()
        logger.info(f"Created JobNotification instance for devices = {devices}")
        self.startTime = datetime.datetime.now()
        self.timeFormatString = r'%d/%m/%Y %H:%M:%S'
        if devices == "all":
            self.pb = Pushbullet("o.B8s0B7VUNVgrzU4fNpvTjan4VKPO6qhJ")
        elif devices == "phone":
            for dv in Pushbullet("o.B8s0B7VUNVgrzU4fNpvTjan4VKPO6qhJ").devices:
                if dv.nickname == phoneName:
                    self.pb = dv

    def pretty_time_delta(self, td):
        seconds          = int(td.total_seconds())
        days, seconds    = divmod(seconds, 86400)
        hours, seconds   = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        logger.debug(f"seconds = {seconds}, minutes = {minutes}, hours = {hours}, days = {days}")
        if days > 0:
            formatted_timedelta = '%dd%dh%dm%ds' % (days, hours, minutes, seconds)
        elif hours > 0:
            formatted_timedelta = '%dh%dm%ds' % (hours, minutes, seconds)
        elif minutes > 0:
            formatted_timedelta = '%dm%ds' % (minutes, seconds)
        else:
            formatted_timedelta = '%ds' % (seconds,)
        logging.debug(f"formatted_timedelta = {formatted_timedelta}")
        return formatted_timedelta

    def send(self, exception=None, message=None):
        logger.info(f"Entered send, type(exception) = {type(exception)}, type(message) = {type(message)}")
        logger.debug(f"message = {message}")
        logger.debug(f"exception = {repr(exception)}")
        self.endTime   = datetime.datetime.now()
        callerFilename = inspect.stack()[1].filename
        if message is None:
            msgLst = list()
        else:
            msgLst = [message]
        msgLst = msgLst + ["Runtime: {timeDelta}",
                           "Started: {timeStarted}",
                           "Finished: {timeEnded}"]
        logger.debug(f"msgLst = {msgLst}")
        if exception is not None:
            title                = f"Error! {callerFilename} have thrown an exception!"
            logger.debug(f"title = {title}")
            traceback_string     = "".join(traceback.format_exc())
            logger.debug(f"traceback_string = {traceback_string}")
            visual_division_line = '* '*len(str(exception))
            logger.debug(f"visual_division_line = {visual_division_line}")
            msgLst               = [str(exception).title(),
                                    visual_division_line,
                                    traceback_string,
                                    visual_division_line] + msgLst
            logger.debug(f"msgLst (there was an exception) = {msgLst}")
        else:
            title = f"{callerFilename} is done"
        msg = "\n".join(msgLst).format(
            callerFilename = callerFilename,  # noqa
            timeDelta      = self.pretty_time_delta(self.endTime - self.startTime),  # noqa
            timeStarted    = self.startTime.strftime(self.timeFormatString),  # noqa
            timeEnded      = self.endTime.strftime(self.timeFormatString))  # noqa
        self.pb.push_note(title, msg)


if __name__ == '__main__':
    from time import sleep

    jn = JobNotification(devices="phone")
    sleep(2)
    try:
        1 / 0
    except Exception as e:
        jn.send(e)
    sleep(2)
    jn.send(message="Hello, World!")
