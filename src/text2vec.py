import collections
import gc
import numpy
from os.path import exists

import h5py

import connectors
import extraction
import kit
import log
import parameters
import processing
import weeding
import binary
import traininig

batchSize = 10000


def extract(connector):
    textFilesCount = connector.count()

    names = []
    texts = []
    for textFileIndex, name, text in connector.iterate():
        text = extraction.clean(text)

        names.append(name)
        texts.append(text)

        log.progress('Extracting text: {0:.3f}%. Texts: {1}.', textFileIndex + 1, textFilesCount, textFileIndex + 1)

    log.lineBreak()

    return names, texts


def buildWordMaps(texts, w2vWordIndexMap, w2vWordEmbeddings):
    wordIndexMap = collections.OrderedDict()
    wordFrequencyMap = collections.OrderedDict()

    for textIndex, text in enumerate(texts):
        for word in weeding.iterateWords(text):
            if word not in w2vWordIndexMap:
                continue

            if word not in wordIndexMap:
                wordIndexMap[word] = len(wordIndexMap)
                wordFrequencyMap[word] = 1
            else:
                wordFrequencyMap[word] += 1

        log.progress('Building word maps: {0:.3f}%. Words: {1}.', textIndex + 1, len(texts), len(wordIndexMap))

    log.lineBreak()

    wordEmbeddings = numpy.zeros((len(wordIndexMap), w2vWordEmbeddings.shape[1]))
    for wordIndexPair in wordIndexMap.items():
        word, index = wordIndexPair
        wordEmbeddings[index] = w2vWordEmbeddings[index]

        log.progress('Copying w2v embeddings: {0:.3f}%.', index + 1, len(wordIndexMap))

    log.lineBreak()

    return wordIndexMap, wordFrequencyMap, wordEmbeddings


def subsampleAndPrune(texts, wordFrequencyMap, sample, minCount):
    totalLength = 0.
    prunedLength = 0.

    maxFrequency = wordFrequencyMap.items()[0][1]

    for textIndex, text in enumerate(texts):
        totalLength += len(text)

        texts[textIndex] = weeding.subsampleAndPrune(text, wordFrequencyMap, maxFrequency, sample, minCount)

        prunedLength += len(texts[textIndex])

        log.progress('Subsampling and pruning text: {0:.3f}%. Removed {1:.3f}% of original text.',
                     textIndex + 1,
                     len(texts),
                     (1 - prunedLength/totalLength) * 100)

    log.lineBreak()

    return texts


def inferContexts(contextsPath, names, texts, wordIndexMap, windowSize, negative, strict, contextsCount):
    textIndexMap = collections.OrderedDict()

    def wordsToIndices(textContext):
        indices = map(lambda word: wordIndexMap[word], textContext)
        return indices

    wordIndices = map(lambda item: item[1], wordIndexMap.items())
    wordIndices = numpy.asarray(wordIndices)
    maxWordIndex = max(wordIndices)

    with h5py.File(contextsPath, 'w') as contextsFile:
        tensor = contextsFile.create_dataset('contexts',
                                             dtype='int32',
                                             shape=(0, contextsCount, 1 + windowSize + negative), # 1 for file index
                                             maxshape=(None, contextsCount, 1 + windowSize + negative), # 1 for file index
                                             chunks=(1, contextsCount, 1 + windowSize + negative)) # 1 for file index

        textsCount = 0
        for name, text in zip(names, texts):
            contextProvider = processing.WordContextProvider(text=text, minContexts=contextsCount, maxContexts=contextsCount)
            contexts = list(contextProvider.iterate(windowSize))

            if len(contexts) > 0:
                contexts = map(wordsToIndices, contexts)
                textIndexMap[name] = len(textIndexMap)
                contexts = numpy.asarray(contexts)
                textIndices = [[textIndexMap[name]]] * len(contexts)
                contexts = numpy.concatenate([textIndices, contexts], axis=1)

                negativeSamples = processing.generateNegativeSamples(negative, contexts, wordIndices, maxWordIndex, strict)
                contexts = numpy.concatenate([contexts, negativeSamples], axis=1)
                tensor.resize(tensor.shape[0] + 1, axis=0)
                tensor[-1] = contexts

            textsCount += 1
            log.progress('Creating contexts: {0:.3f}%. Text index map: {1}. Contexts: {2}.',
                         textsCount,
                         len(texts),
                         len(tensor),
                         tensor.shape[0] * tensor.shape[1])

    log.lineBreak()

    return textIndexMap


def trainTextVectors(connector, w2vEmbeddingsPath, wordIndexMapPath, wordFrequencyMapPath, wordEmbeddingsPath, contextsPath,
                     sample, minCount, windowSize, negative, strict, contextsPerText, superBatchSize, fileEmbeddingSize,
                     epochs, learningRate, fileEmbeddingsPath):
    if exists(wordIndexMapPath) and exists(wordFrequencyMapPath) and exists(wordEmbeddingsPath) \
            and exists(contextsPath) and exists(pathTo.textIndexMap):
        wordIndexMap = parameters.loadMap(wordIndexMapPath)
        wordFrequencyMap = parameters.loadMap(wordFrequencyMapPath)
        wordEmbeddings = parameters.loadEmbeddings(wordEmbeddingsPath)
        textIndexMap = parameters.loadMap(pathTo.textIndexMap)
    else:
        w2vWordIndexMap, w2vWordEmbeddings = parameters.loadW2VParameters(w2vEmbeddingsPath)

        names, texts = extract(connector)
        wordIndexMap, wordFrequencyMap, wordEmbeddings = buildWordMaps(texts, w2vWordIndexMap, w2vWordEmbeddings)

        parameters.dumpWordMap(wordIndexMap, wordIndexMapPath)
        del w2vWordIndexMap
        del w2vWordEmbeddings
        gc.collect()

        parameters.dumpWordMap(wordFrequencyMap, wordFrequencyMapPath)

        log.progress('Dumping contexts...')
        parameters.dumpEmbeddings(wordEmbeddings, wordEmbeddingsPath)
        log.info('Dumped indices, frequencies and embeddings')

        texts = subsampleAndPrune(texts, wordFrequencyMap, sample, minCount)

        textIndexMap = inferContexts(contextsPath, names, texts, wordIndexMap, windowSize, negative, strict, contextsPerText)

        parameters.dumpWordMap(textIndexMap, pathTo.textIndexMap)

    with h5py.File(contextsPath, 'r') as contextsFile:
        contexts = contextsFile['contexts']
        log.info('Loaded {0} contexts. Shape: {1}', len(contexts), contexts.shape)

        fileEmbeddings = numpy.random.rand(len(contexts), fileEmbeddingSize).astype('float32')
        trainingBatch = numpy.zeros((superBatchSize, contextsPerText, 1+windowSize+negative)).astype('int32')
        superBatchesCount = len(contexts) / superBatchSize

        for superBatchIndex in xrange(0, superBatchesCount):
            log.info('Text batch: {0}/{1}.', superBatchIndex + 1, superBatchesCount)

            # TODO: this only works if superBatchSize == textsCount; otherwise text indices do not match
            contexts.read_direct(trainingBatch, source_sel=numpy.s_[superBatchIndex*superBatchSize:(superBatchIndex+1)*superBatchSize])
            trainingBatchReshaped = trainingBatch.reshape((superBatchSize*contextsPerText, 1+windowSize+negative))

            fileEmbeddingsBatch = fileEmbeddings[superBatchIndex*superBatchSize:(superBatchIndex+1)*superBatchSize]

            model = traininig.Model(fileEmbeddingsBatch, wordEmbeddings, contextSize=windowSize-2, negative=negative)
            traininig.train(model, textIndexMap, wordIndexMap, wordEmbeddings, trainingBatchReshaped, epochs, 1, learningRate)

            fileEmbeddings[superBatchIndex*superBatchSize:(superBatchIndex+1)*superBatchSize] = model.fileEmbeddings.get_value()
            contextsFile.flush()

        log.progress('Dumping text embeddings...')
        binary.dumpTensor(fileEmbeddingsPath, fileEmbeddings)
        log.info('Dumping text embeddings complete')



def launch(pathTo, hyper):
    w2vEmbeddingsPath = pathTo.w2vEmbeddings
    contextsPath = pathTo.contexts
    wordIndexMap = pathTo.wordIndexMap
    wordFrequencyMap = pathTo.wordFrequencyMap
    wordEmbeddings = pathTo.wordEmbeddings

    connector = hyper.connector
    windowSize = hyper.windowSize
    negative = hyper.negative
    strict = hyper.strict

    trainTextVectors(connector, w2vEmbeddingsPath, wordIndexMap, wordFrequencyMap, wordEmbeddings, contextsPath,
                     hyper.sample, hyper.minCount, windowSize, negative, strict, hyper.contextsPerText, hyper.superBatchSize,
                     hyper.fileEmbeddingSize, hyper.epochs, hyper.learningRate, pathTo.fileEmbeddings)


if __name__ == '__main__':
    pathTo = kit.PathTo('Cockatoo', experiment='cockatoo', w2vEmbeddings='wiki_full_s1000_w10_mc20_hs1.bin')
    hyper = parameters.HyperParameters(
        connector=connectors.TextFilesConnector(pathTo.dataSetDir),
        threshold=1e-3,
        minCount=1,
        windowSize=3,
        negative=100,
        strict=False,
        contextsPerText=600,
        fileEmbeddingSize=1000,
        epochs=5,
        batchSize=1,
        learningRate=0.025,
        superBatchSize=25
    )

    launch(pathTo, hyper)