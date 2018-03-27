#!/usr/bin/env python
# -*- coding: utf8 -*-

from pushbullet import Pushbullet
import datetime
import inspect
import traceback

phoneName = "OnePlus ONE A2003"


class JobNotification(object):
    """Class to send push notification when a job finishes, either because of an error, or
    because it's completed.
    """

    def __init__(self, devices="phone"):
        super(JobNotification, self).__init__()
        self.startTime = datetime.datetime.now()
        self.timeFormatString = r'%d/%m/%Y %H:%M:%S'
        if devices == "all":
            self.pb = Pushbullet("o.B8s0B7VUNVgrzU4fNpvTjan4VKPO6qhJ")
        elif devices == "phone":
            for dv in Pushbullet("o.B8s0B7VUNVgrzU4fNpvTjan4VKPO6qhJ").devices:
                if dv.nickname == phoneName:
                    self.pb = dv

    def pretty_time_delta(self, td):
        seconds = int(td.total_seconds())
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        if days > 0:
            return '%dd%dh%dm%ds' % (days, hours, minutes, seconds)
        elif hours > 0:
            return '%dh%dm%ds' % (hours, minutes, seconds)
        elif minutes > 0:
            return '%dm%ds' % (minutes, seconds)
        else:
            return '%ds' % (seconds,)

    def send(self, exception=None, message=None):
        self.endTime = datetime.datetime.now()
        callerFilename = inspect.stack()[1].filename
        if message is None:
            msgLst = list()
        else:
            msgLst = [message]
        msgLst = msgLst + ["Runtime: {timeDelta}",
                           "Started: {timeStarted}",
                           "Finished: {timeEnded}"]
        if exception is not None:
            title = f"Error! {callerFilename} have thrown an exception!"
            traceback_string = "".join(traceback.format_exc())
            visual_division_line = '* '*len(str(exception))
            msgLst = [str(exception).title(),
                      visual_division_line,
                      traceback_string,
                      visual_division_line] + msgLst
        else:
            title = f"{callerFilename} is done"
        msg = "\n".join(msgLst).format(
            callerFilename=callerFilename,
            timeDelta=self.pretty_time_delta(self.endTime - self.startTime),
            timeStarted=self.startTime.strftime(self.timeFormatString),
            timeEnded=self.endTime.strftime(self.timeFormatString))
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
