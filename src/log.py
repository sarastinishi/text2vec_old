import sys
from datetime import timedelta


def info(message, *args):
    message = str(message)
    message = message.format(*args)
    message = '\r' + message

    sys.stdout.write(message)
    sys.stdout.flush()

    lineBreak()


def lineBreak():
    sys.stdout.write('\n')
    sys.stdout.flush()


def progress(messageFormat, index=None, count=None, *args):
    if index == None or count == None:
        index = 1
        count = 1

    maxFrequency = 10000
    body = count - count % maxFrequency
    maxFrequency = count / maxFrequency
    if maxFrequency != 0 and index % maxFrequency > 0 and index < body:
        return

    index = float(index)
    count = float(count)
    percentage = 100 * index / count
    percentage = min(100, percentage)
    args = [percentage] + list(args)

    messageFormat = messageFormat.format(*args)
    messageFormat = '\r' + messageFormat

    sys.stdout.write(messageFormat)
    sys.stdout.flush()


def delta(seconds):
    deltaString = str(timedelta(seconds=seconds))
    deltaString = deltaString.split('.')[0]

    return deltaString
