import numpy as np
import time
import os

import theano
import theano.tensor as T

import parameters
import log
import validation
import kit
import binary


floatX = theano.config.floatX
empty = lambda *shape: np.empty(shape, dtype='int32')
rnd2 = lambda d0, d1: np.random.rand(d0, d1).astype(dtype=floatX)
asfx = lambda x: np.asarray(x, dtype=floatX)


class Model:
    def __init__(self, fileEmbeddings, wordEmbeddings, weights=None, contextSize=None, negative=None):
        filesCount, fileEmbeddingSize = fileEmbeddings.shape
        wordsCount, wordEmbeddingSize = wordEmbeddings.shape

        trainWeights = weights is None
        if trainWeights:
            weights = rnd2(fileEmbeddingSize + contextSize * wordEmbeddingSize, wordsCount)
        else:
            featuresCount, activationsCount = weights.shape
            contextSize = (featuresCount - fileEmbeddingSize) / wordEmbeddingSize
            negative = activationsCount - 1

        self.fileEmbeddings = theano.shared(asfx(fileEmbeddings), 'fileEmbeddings', borrow=False)
        self.wordEmbeddings = theano.shared(asfx(wordEmbeddings), 'wordEmbeddings', borrow=False)
        self.weights = theano.shared(asfx(weights), 'weights', borrow=False)

        fileIndexOffset = 0
        wordIndicesOffset = fileIndexOffset + 1
        indicesOffset = wordIndicesOffset + contextSize

        contexts = T.imatrix('contexts')
        fileIndices = contexts[:,fileIndexOffset:wordIndicesOffset]
        wordIndices = contexts[:,wordIndicesOffset:indicesOffset]
        indices = contexts[:,indicesOffset:indicesOffset + negative]

        files = self.fileEmbeddings[fileIndices]
        fileFeatures = T.flatten(files, outdim=2)
        words = self.wordEmbeddings[wordIndices]
        wordFeatures = T.flatten(words, outdim=2)
        features = T.concatenate([fileFeatures, wordFeatures], axis=1)

        subWeights = self.weights[:,indices].dimshuffle(1, 0, 2)

        probabilities = T.batched_dot(features, subWeights)

        parameters = [self.fileEmbeddings]
        subParameters = [files]
        consider_constant = [self.wordEmbeddings]

        if trainWeights:
            parameters.append(self.weights)
            subParameters.append(subWeights)
        else:
            consider_constant.append(self.weights)

        # cost = -T.mean(T.log(T.nnet.sigmoid(probabilities[:,0])) + T.sum(T.log(T.nnet.sigmoid(-probabilities[:,1:])), dtype=floatX, acc_dtype=floatX), dtype=floatX, acc_dtype=floatX)
        cost = -T.log(T.nnet.sigmoid(probabilities[:,0])) - T.sum(T.log(T.nnet.sigmoid(-probabilities[:,1:])), dtype=floatX, acc_dtype=floatX)

        learningRate = T.scalar('learningRate', dtype=floatX)

        updates = []
        for p, subP in zip(parameters, subParameters):
            if subP is not None:
                gradient = T.jacobian(cost, wrt=subP)
                update = (p, T.inc_subtensor(subP, -learningRate * gradient))
            else:
                gradient = T.jacobian(cost, wrt=p)
                update = (p, p - learningRate * gradient)

            updates.append(update)

        contextIndex = T.iscalar('contextIndex')
        self.trainingContexts = theano.shared(empty(1,1,1), 'trainingContexts', borrow=False)

        self.trainModel = theano.function(
            inputs=[contextIndex, learningRate],
            outputs=cost,
            updates=updates,
            givens={
                contexts: self.trainingContexts[:,contextIndex]
            }
        )


    def dump(self, fileEmbeddingsPath, weightsPath):
        fileEmbeddings = self.fileEmbeddings.get_value()
        binary.dumpTensor(fileEmbeddingsPath, fileEmbeddings)

        weights = self.weights.get_value()
        binary.dumpTensor(weightsPath, weights)


    @staticmethod
    def load(fileEmbeddingsPath, wordEmbeddingsPath, weightsPath):
        fileEmbeddings = binary.loadTensor(fileEmbeddingsPath)
        wordEmbeddings = binary.loadTensor(wordEmbeddingsPath)
        weights = binary.loadTensor(weightsPath)

        return Model(fileEmbeddings, wordEmbeddings, weights)



def train(model, fileIndexMap, wordIndexMap, wordEmbeddings, contexts,
          epochs, batchSize, learningRate, metricsPath=None):
    model.trainingContexts.set_value(contexts)

    textsCount, contextsCount, contextSize = contexts.shape

    initialiLearningRate = learningRate
    startTime = time.time()
    metrics = {
        'meanError': np.nan,
        'medianError': np.nan,
        'maxError': np.nan,
        'minError': np.nan,
        'learningRate': learningRate
    }

    for epoch in xrange(0, epochs):
        errors = []
        for contextIndex in xrange(0, contextsCount):
            error = model.trainModel(contextIndex, learningRate)
            errors.append(error)

            log.progress('Training model: {0:.3f}%. Epoch: {1}. Elapsed: {2}. Error(mean,median,min,max): {3:.3f}, {4:.3f}, {5:.3f}, {6:.3f}. Learning rate: {7}.',
                     epoch * contextsCount + contextIndex + 1,
                     epochs * contextsCount,
                     epoch + 1,
                     log.delta(time.time() - startTime),
                     metrics['meanError'],
                     metrics['medianError'],
                     metrics['minError'],
                     metrics['maxError'],
                     learningRate)

        learningRate = learningRate * (1 - (float(epoch) + 1) / epochs)
        learningRate = max(initialiLearningRate * 0.0001, learningRate)

        metrics = {
            'meanError': np.mean(errors),
            'medianError': np.median(errors),
            'maxError': np.max(errors),
            'minError': np.min(errors),
            'learningRate': learningRate
        }

        if metricsPath is not None:
            validation.dump(metricsPath, epoch, metrics)


def launch(pathTo, hyper):
    fileIndexMap = parameters.loadMap(pathTo.textIndexMap)
    filesCount = len(fileIndexMap)
    fileEmbeddingSize = hyper.fileEmbeddingSize
    wordIndexMap = parameters.loadMap(pathTo.wordIndexMap)
    wordEmbeddings = parameters.loadEmbeddings(pathTo.wordEmbeddings)
    metricsPath = pathTo.metrics('history.csv')

    if os.path.exists(metricsPath):
        os.remove(metricsPath)

    contextProvider = parameters.IndexContextProvider(pathTo.contexts)
    windowSize = contextProvider.windowSize - 1
    contextSize = windowSize - 1
    negative = contextProvider.negative
    contexts = contextProvider[:]

    log.info('Contexts loading complete. {0} contexts loaded {1} words and {2} negative samples each.',
             len(contexts),
             contextProvider.windowSize,
             contextProvider.negative)

    fileEmbeddings = rnd2(filesCount, fileEmbeddingSize)
    model = Model(fileEmbeddings, wordEmbeddings, contextSize=contextSize, negative=negative)
    # model = Model.load(pathTo.fileEmbeddings, pathTo.wordEmbeddings, pathTo.weights)

    train(model, fileIndexMap, wordIndexMap, wordEmbeddings, contexts,
          epochs=hyper.epochs,
          batchSize=hyper.batchSize,
          learningRate=hyper.learningRate,
          metricsPath=metricsPath)

    model.dump(pathTo.fileEmbeddings, pathTo.weights)


if __name__ == '__main__':
    pathTo = kit.PathTo('Duplicates', experiment='duplicates', w2vEmbeddings='wiki_full_s1000_w10_mc20_hs1.bin')
    hyper = parameters.HyperParameters(fileEmbeddingSize=1000, epochs=5, batchSize=1, learningRate=0.01)

    launch(pathTo, hyper)