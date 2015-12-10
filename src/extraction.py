import os
import shutil
import time
import re

import kit
import log
import connectors
import parameters


def clean(text):
    text = text.lower()

    text = re.sub('\s+', ' ', text)
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
    text = re.sub('\([^\)]+\)', '', text)
    text = re.sub('\[[^\]]+\]', '', text)
    text = re.sub('(:[^\.]\.)', '', text)
    text = re.sub('[,":_\*]', ' ', text)
    text = re.sub('!', '.', text)
    text = re.sub('\?', '.', text)
    text = re.sub('\s(\.{4,})\s', ' ', text)
    text = re.sub('\s(\.{3})\s', '.', text)
    text = re.sub('\s(\.{2})\s', ' ', text)
    text = re.sub('<[^>]+>', '', text)
    text = re.sub('([0-9\-]+)s', ' NUMBER ', text)
    text = re.sub('([0-9\-]+)th', ' NUMBER ', text)
    text = re.sub('[^a-z]+([0-9\-]+)[^a-z]+', ' NUMBER ', text)
    text = re.sub('\s([^a-zA-Z0-9\.\-\s]+)\s', ' SYMBOL ', text)
    text = re.sub('\s([bcdefghjklmnopqrstuvwxyz])\s', ' SYMBOL ', text)

    sentences = re.split('[(\n{2,})\.;]', text)
    sentences = [re.sub('[\s]+', ' ', sentence).strip() for sentence in sentences]
    sentences = [sentence for sentence in sentences
                 if len(sentence.split(' ')) > 2 and sentence.count('NUMBER') < 3]

    text = '. '.join(sentences) + '.'

    return text

def saveText(filePath, text):
    with open(filePath, 'a+') as pageFile:
        if pageFile.tell():
            pageFile.write(' ')

        pageFile.write(text)


def extract(outputDirectoryPath, outputConcatFilePath, connector):
    if os.path.exists(outputDirectoryPath):
        shutil.rmtree(outputDirectoryPath, ignore_errors=True)

    os.mkdir(outputDirectoryPath)
    os.chown(outputDirectoryPath, 1000, 1000)

    textContainersCount = connector.count()
    log.info('Found {0} text containers.', textContainersCount)

    if os.path.exists(outputConcatFilePath):
        os.remove(outputConcatFilePath)

    pagesCount = 0
    startTime = time.time()

    for textContainerIndex, name, text in connector.iterate():
        text = clean(text)

        outputFilePath = os.path.join(outputDirectoryPath, name + '.txt')

        saveText(outputFilePath, text)
        saveText(outputConcatFilePath, text)

        currentTime = time.time()
        elapsed = currentTime - startTime
        pagesCount += 1

        log.progress('Extracting text containers: {0:.3f}%. Elapsed: {1}. Pages: {2}.',
                     textContainerIndex + 1,
                     textContainersCount,
                     log.delta(elapsed),
                     pagesCount)

    log.lineBreak()


def launch(pathTo, hyper):
    extract(pathTo.extractedDir, pathTo.concatenated, hyper.connector)


if __name__ == '__main__':
    pathTo = kit.PathTo('Cockatoo', experiment='cockatoo', w2vEmbeddings='wiki_full_s1000_w10_mc20_hs1.bin')
    # pathTo = kit.PathTo('Duplicates', experiment='duplicates', w2vEmbeddings='wiki_full_s1000_w10_mc20_hs1.bin')
    hyper = parameters.HyperParameters(connector = connectors.TextFilesConnector(pathTo.dataSetDir))

    launch(pathTo, hyper)
