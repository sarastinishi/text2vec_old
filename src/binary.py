import struct
import operator
import functools
import numpy as np


def read(file, dtype, count):
    buffer = np.fromfile(file, dtype=dtype, count=count)

    if count == 1:
        return buffer[0]

    return buffer


def write(file, format, buffer):
    if isinstance(buffer, np.ndarray):
        buffer = buffer.flatten()
    elif not isinstance(buffer, list):
        buffer = [buffer]

    format = format.format(len(buffer))
    buffer = struct.pack(format, *buffer)

    file.write(buffer)


def readi(file, count=1):
    return read(file, 'int32', count)


def writei(file, buffer):
    write(file, '{0}i', buffer)


def readf(file, count=1):
    return read(file, 'float32', count)


def writef(file, buffer):
    write(file, '{0}f', buffer)


def reads(file, length):
    return file.read(length)


def writes(file, buffer):
    file.write(buffer)


def dumpTensor(path, tensor):
    tensor = np.asarray(tensor)

    shape = list(tensor.shape)
    dimensions = len(shape)

    values = np.asarray(tensor).flatten()

    with open(path, 'wb+') as file:
        writei(file, dimensions)
        writei(file, shape)
        writef(file, values)


def loadTensor(path):
    with open(path, 'rb') as file:
        dimensions = readi(file)
        shape = readi(file, dimensions)
        count = functools.reduce(operator.mul, shape, 1)
        values = readf(file, count)
        matrix = np.reshape(values, shape)

        return matrix