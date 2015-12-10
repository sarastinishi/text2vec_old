import os
import glob
import gzip
import re
import numpy as np
import pandas as pd


class TextFilesConnector:
    def __init__(self, inputDirectoryPath):
        pathName = inputDirectoryPath + '/*.txt'
        self.textFilePaths = glob.glob(pathName)
        self.textFilePaths = sorted(self.textFilePaths)


    def count(self):
        return len(self.textFilePaths)


    def iterate(self):
        for textFileIndex, textFilePath in enumerate(self.textFilePaths):
            with open(textFilePath, 'r') as textFile:
                textFileName = os.path.basename(textFilePath).split('.')[0]
                text = textFile.read()

                yield textFileIndex, textFileName, text


class WikipediaConnector:
    def __init__(self, inputDirectoryPath):
        pathName = inputDirectoryPath + '/*.txt.gz'
        self.dumpPaths = glob.glob(pathName)

        self.dumpPaths = self.dumpPaths[:10]


    @staticmethod
    def filterPage(page):
        name, text = page

        if ':' in name:
            return False

        mayReferTo = '{0} may refer to'.format(name).lower()
        if text.startswith(mayReferTo):
            return False

        if text.startswith('#redirect'):
            return False

        if len(text) < 10:
            return False

        return True


    @staticmethod
    def unpackDump(dumpPath):
        dumpName = os.path.basename(dumpPath).split('.')[0]
        pages = []

        try:
            with gzip.open(dumpPath, 'rb') as dumpFile:
                dumpText = dumpFile.read()

            names = [name.strip() for name in re.findall('^\[\[(?P<title>[^\]]+)\]\]\s?$', dumpText, flags=re.M)]

            texts = [text.strip() for text in re.split('^\[\[[^\]]+\]\]\s?$', dumpText, flags=re.M) if text]

            pages = zip(names, texts)
            pages = filter(WikipediaConnector.filterPage, pages)
        except:
            pass

        return dumpName, pages


    @staticmethod
    def stripWikiMarkup(name, text):
        name = re.sub('[^_a-zA-Z0-9\s\(\)]', '', name).strip()

        restrictedHeaders = ['see also', 'footnotes', 'references', 'further reading', 'external links', 'books']

        headings = [name] + re.findall('^=+\s*([^=]+)\s*=+$', text, flags=re.M)
        paragraphs = re.split('^=+\s*[^=]+\s*=+$', text, flags=re.M)

        text = ''

        for heading, paragraph in zip(headings, paragraphs):
            if heading.lower() not in restrictedHeaders:
                text += paragraph

        return name, text


    def count(self):
        return len(self.dumpPaths)


    def iterate(self):
        for dumpIndex, dumpPath in enumerate(self.dumpPaths):
            dumpName, pages = WikipediaConnector.unpackDump(dumpPath)

            if any(pages):
                for name, text in pages:
                    name, text = WikipediaConnector.stripWikiMarkup(name, text)

                    yield dumpIndex, name, text


class ImdbConnector:
    def __init__(self, inputDirectoryPath):
        self.inputDirectoryPath = inputDirectoryPath

        self.trainDir = os.path.join(self.inputDirectoryPath, 'train')
        self.trainNegativeDir = os.path.join(self.trainDir, 'neg')
        self.trainPositiveDir = os.path.join(self.trainDir, 'pos')
        self.trainUnsupervisedDir = os.path.join(self.trainDir, 'unsup')

        self.testDir = os.path.join(self.inputDirectoryPath, 'test')
        self.testNegativeDir = os.path.join(self.testDir, 'neg')
        self.testPositiveDir = os.path.join(self.testDir, 'pos')

        dirs = [self.trainNegativeDir, self.trainPositiveDir, self.trainUnsupervisedDir,
                self.testNegativeDir, self.testPositiveDir]

        self.textFilesPaths = []
        for dir in dirs:
            pathName = dir + '/*.txt'
            self.textFilesPaths += glob.glob(pathName)

        self.textFilesPaths = self.textFilesPaths


    def count(self):
        return len(self.textFilesPaths)


    def iterate(self):
        for textFileIndex, textFilePath in enumerate(self.textFilesPaths):
            with open(textFilePath, 'r') as textFile:
                text = textFile.read()

                yield textFileIndex, textFilePath, text


class RottenTomatosConnector:
    def __init__(self, inputDirectoryPath):
        self.inputDirectoryPath = inputDirectoryPath

        self.trainFilePath = os.path.join(self.inputDirectoryPath, 'train.tsv')
        self.testFilePath = os.path.join(self.inputDirectoryPath, 'test.tsv')


    def count(self):
        trainSet = pd.read_csv(self.trainFilePath, sep='\t')
        testSet = pd.read_csv(self.testFilePath, sep='\t')

        dataSet = pd.concat([trainSet, testSet])
        dataSet = dataSet.groupby('SentenceId').first()

        return len(dataSet)


    def iterate(self):
        trainSet = pd.read_csv(self.trainFilePath, sep='\t')
        testSet = pd.read_csv(self.testFilePath, sep='\t')

        dataSet = pd.concat([trainSet, testSet])
        dataSet = dataSet.groupby('SentenceId').first()

        phraseIndex = 0
        for phraseId, phrase in zip(dataSet['PhraseId'], dataSet['Phrase']):
            yield phraseIndex, str(phraseId), phrase
            phraseIndex += 1