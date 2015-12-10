import re
import pandas
import log
import os


def getSyntacticWordRelationsWords(filePath):
    with open(filePath, 'r') as swrFile:
        lines = swrFile.readlines()
        words = [tuple(line.lower().split(' ')) for line in lines if not line.startswith(':')]
        words = [(word0.strip(), word1.strip(), word2.strip(), word3.strip()) for word0, word1, word2, word3 in words]

    ret = []
    for word0, word1, word2, word3 in words:
        ret.append(word0)
        ret.append(word1)
        ret.append(word2)
        ret.append(word3)

    return ret


def getSATWords(filePath):
    maxLineLength = 50

    words = []

    with open(filePath) as satFile:
        line = satFile.readline()
        while line != '':
            if len(line) < maxLineLength:
                match = re.match('(?P<word0>[\w-]+)\s(?P<word1>[\w-]+)\s[nvar]:[nvar]', line)
                if match:
                    stemWord0, stemWord1 = match.group('word0'), match.group('word1')

                    words.append(stemWord0)
                    words.append(stemWord1)

                    line = satFile.readline()
                    match = re.match('(?P<word0>[\w-]+)\s(?P<word1>[\w-]+)\s[nvar]:[nvar]', line)
                    while match:
                        choiceWord0, choiceWord1 = match.group('word0'), match.group('word1')

                        words.append(choiceWord0)
                        words.append(choiceWord1)

                        line = satFile.readline()
                        match = re.match('(?P<word0>[\w-]+)\s(?P<word1>[\w-]+)\s[nvar]:[nvar]', line)

            line = satFile.readline()

    return words


def getSimLex999Words(filePath):
    data = pandas.read_csv(filePath, sep='\t')

    words = []

    for word1, word2, targetScore in zip(data['word1'], data['word2'], data['SimLex999']):
        words.append(word1)
        words.append(word2)

    return words


def getWordSimilarity353Words(filePath):
    data = pandas.read_csv(filePath)

    words = []
    for word1, word2, score in zip(data['Word1'], data['Word2'], data['Score']):
        words.append(word1)
        words.append(word2)

    return words


def getRubensteinGoodenoughWords(filePath):
    with open(filePath) as rgFile:
        lines = rgFile.readlines()

    words = []
    for line in lines:
        word1, word2, score = tuple(line.strip().split('\t'))
        words.append(word1)
        words.append(word2)

    return words


def make():
    words = []

    words += getSyntacticWordRelationsWords('res/Syntactic-Word-Relations/questions-words.txt')
    words += getSATWords('res/SAT-Questions/SAT-package-V3.txt')
    words += getSimLex999Words('res/SimLex-999/SimLex-999.txt')
    words += getWordSimilarity353Words('res/WordSimilarity-353/combined.csv')
    words += getRubensteinGoodenoughWords('res/RG/EN-RG-65.txt')

    words = list(set(words))
    words = sorted(words)

    log.info('Found {0} words.', len(words))

    whiteListPath = 'res/Tools/white_list.txt'
    if os.path.exists(whiteListPath):
        os.remove(whiteListPath)

    with open(whiteListPath, 'w+') as whiteListFile:
        batchSize = 10
        batchesCount = len(words) / batchSize + 1
        for batchIndex in xrange(0, batchesCount):
            batch = words[batchIndex * batchSize : (batchIndex + 1) * batchSize]
            line =  ' '.join(batch) + '\n'
            line = line.lower()

            whiteListFile.write(line)

            log.progress('Saving white list: {0:.0f}%.', batchIndex + 1, batchesCount)

    log.lineBreak()
    log.info('White list has been saved.')


def load():
    whiteListFilePath = 'res/Tools/white_list.txt'

    with open(whiteListFilePath, 'r') as whiteListFile:
        text = whiteListFile.read()
        whiteList = [word for word in re.split('\s+', text) if word]

        return whiteList