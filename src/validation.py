import itertools
import numpy
import os
import random
import re
import scipy.spatial.distance as ssd
import scipy.stats
from scipy.cluster.hierarchy import dendrogram, linkage

import pandas
from matplotlib import colors
from matplotlib import pyplot as plt

import vectors
from libs import tsne

rubensteinGoodenoughData = None
def rubensteinGoodenough(wordIndexMap, embeddings):
    global rubensteinGoodenoughData

    if rubensteinGoodenoughData is None:
        rubensteinGoodenoughData = []

        rubensteinGoodenoughFilePath = 'res/RG/EN-RG-65.txt'

        with open(rubensteinGoodenoughFilePath) as rgFile:
            lines = rgFile.readlines()

        for line in lines:
            word0, word1, targetScore = tuple(line.strip().split('\t'))
            targetScore = float(targetScore)

            rubensteinGoodenoughData.append((word0, word1, targetScore))

    scores = []
    targetScores = []
    for word0, word1, targetScore in rubensteinGoodenoughData:
        if word0 in wordIndexMap and word1 in wordIndexMap:
            targetScores.append(targetScore)

            word0Index = wordIndexMap[word0]
            word1Index = wordIndexMap[word1]
            word0Embedding = embeddings[word0Index]
            word1Embedding = embeddings[word1Index]

            score = vectors.cosineSimilarity(word0Embedding, word1Embedding)
            scores.append(score)

    if len(scores) == 0:
        return numpy.nan

    pearson, pearsonDeviation = scipy.stats.pearsonr(scores, targetScores)
    spearman, spearmanDeviation = scipy.stats.spearmanr(scores, targetScores)

    rubensteinGoodenoughMetric = numpy.mean([pearson, spearman])

    return rubensteinGoodenoughMetric


wordSimilarity353Data = None
def wordSimilarity353(wordIndexMap, embeddings):
    global wordSimilarity353Data

    if wordSimilarity353Data is None:
        wordSimilarity353Data = []

        wordSimilarity353FilePath = 'res/WordSimilarity-353/combined.csv'
        data = pandas.read_csv(wordSimilarity353FilePath)

        for word0, word1, score in zip(data['Word1'], data['Word2'], data['Score']):
            wordSimilarity353Data.append((word0, word1, score))

    scores = []
    targetScores = []
    for word0, word1, targetScore in wordSimilarity353Data:
        if word0 in wordIndexMap and word1 in wordIndexMap:
            targetScores.append(targetScore)

            word0Index = wordIndexMap[word0]
            word1Index = wordIndexMap[word1]
            word0Embedding = embeddings[word0Index]
            word1Embedding = embeddings[word1Index]

            score = vectors.cosineSimilarity(word0Embedding, word1Embedding)
            scores.append(score)

    if len(scores) == 0:
        return numpy.nan

    pearson, pearsonDeviation = scipy.stats.pearsonr(scores, targetScores)
    spearman, spearmanDeviation = scipy.stats.spearmanr(scores, targetScores)

    metric = numpy.mean([pearson, spearman])

    return metric


simLex999Data = None
def simLex999(wordIndexMap, embeddings):
    global simLex999Data

    if simLex999Data is None:
        simLex999Data = []
        simLex999FilePath = 'res/SimLex-999/SimLex-999.txt'
        data = pandas.read_csv(simLex999FilePath, sep='\t')

        for word0, word1, targetScore in zip(data['word1'], data['word2'], data['SimLex999']):
            simLex999Data.append((word0, word1, targetScore))

    targetScores = []
    scores = []
    for word0, word1, targetScore in simLex999Data:
        if word0 in wordIndexMap and word1 in wordIndexMap:
            targetScores.append(targetScore)

            word0Index = wordIndexMap[word0]
            word1Index = wordIndexMap[word1]
            word0Embedding = embeddings[word0Index]
            word1Embedding = embeddings[word1Index]

            score = vectors.cosineSimilarity(word0Embedding, word1Embedding)
            scores.append(score)

    if len(scores) == 0:
        return numpy.nan

    pearson, pearsonDeviation = scipy.stats.pearsonr(scores, targetScores)
    spearman, spearmanDeviation = scipy.stats.spearmanr(scores, targetScores)

    simLex999Metric = numpy.mean([pearson, spearman])

    return simLex999Metric


syntacticWordData = None
def syntacticWordRelations(wordIndexMap, embeddings, maxWords=10):
    global syntacticWordData

    if syntacticWordData is None:
        syntacticWordData = []
        syntWordRelFilePath = 'res/Syntactic-Word-Relations/questions-words.txt'

        with open(syntWordRelFilePath, 'r') as swrFile:
            lines = swrFile.readlines()
            syntacticWordData = [tuple(line.lower().split(' ')) for line in lines if not line.startswith(':')]
            syntacticWordData = [(word0.strip(), word1.strip(), word2.strip(), word3.strip()) for word0, word1, word2, word3 in syntacticWordData]

    scores = []
    for word0, word1, word2, word3 in syntacticWordData:
        if word0 not in wordIndexMap or word1 not in wordIndexMap or word2 not in wordIndexMap or word3 not in wordIndexMap:
            continue

        word0Index = wordIndexMap[word0]
        word1Index = wordIndexMap[word1]
        word2Index = wordIndexMap[word2]
        word3Index = wordIndexMap[word3]

        word0Embedding = embeddings[word0Index]
        word1Embedding = embeddings[word1Index]
        word2Embedding = embeddings[word2Index]
        word3Embedding = embeddings[word3Index]

        similarity01 = vectors.cosineSimilarity(word0Embedding, word1Embedding)
        similarity23 = vectors.cosineSimilarity(word2Embedding, word3Embedding)

        score = 1
        minSimilarityDelta = abs(similarity01 - similarity23)
        for embedding in embeddings[:maxWords]:
            similarity2N = vectors.cosineSimilarity(word2Embedding, embedding)
            similarityDelta = abs(similarity01 - similarity2N)

            score = not (similarityDelta < minSimilarityDelta)
            if not score:
                break

        scores.append(score)

    if len(scores) == 0:
        return numpy.nan

    syntacticWordRelationsMetric = float(sum(scores)) / len(scores)

    return syntacticWordRelationsMetric


satQuestionsData = None
def satQuestions(wordIndexMap, embeddings):
    global satQuestionsData

    if satQuestionsData is None:
        satQuestionsData = []
        satQuestionsFilePath = 'res/SAT-Questions/SAT-package-V3.txt'

        maxLineLength = 50
        aCode = ord('a')

        with open(satQuestionsFilePath) as satFile:
            line = satFile.readline()
            while line != '':
                if len(line) < maxLineLength:
                    match = re.match('(?P<word0>[\w-]+)\s(?P<word1>[\w-]+)\s[nvar]:[nvar]', line)
                    if match:
                        stemWord0, stemWord1 = match.group('word0'), match.group('word1')
                        satQuestion = [stemWord0, stemWord1]

                        line = satFile.readline()
                        match = re.match('(?P<word0>[\w-]+)\s(?P<word1>[\w-]+)\s[nvar]:[nvar]', line)
                        while match:
                            choiceWord0, choiceWord1 = match.group('word0'), match.group('word1')
                            satQuestion.append(choiceWord0)
                            satQuestion.append(choiceWord1)

                            line = satFile.readline()
                            match = re.match('(?P<word0>[\w-]+)\s(?P<word1>[\w-]+)\s[nvar]:[nvar]', line)

                        correctChoiceIndex = ord(line.strip()) - aCode
                        satQuestion.append(correctChoiceIndex)

                        satQuestionsData.append(satQuestion)

                line = satFile.readline()

    scores = []
    for satQuestion in satQuestionsData:
        if any([word not in wordIndexMap for word in satQuestion[:-1]]):
            continue

        stemWord0, stemWord1 = satQuestion[:2]

        stemWord0Index = wordIndexMap[stemWord0]
        stemWord1Index = wordIndexMap[stemWord1]
        stemWord0Embedding, stemWord1Embedding = embeddings[stemWord0Index], embeddings[stemWord1Index]
        stemSimilarity = vectors.cosineSimilarity(stemWord0Embedding, stemWord1Embedding)

        correctChoiceIndex = satQuestion[-1]
        choiceSimilarityDeltas = []

        choices = satQuestion[2:-1]
        for i in xrange(0, len(choices), 2):
            choiceWord0, choiceWord1 = choices[i], choices[i+1]
            choiceWord0Index, choiceWord1Index = wordIndexMap[choiceWord0], wordIndexMap[choiceWord1]
            choiceWord0Embedding, choiceWord1Embedding = embeddings[choiceWord0Index], embeddings[choiceWord1Index]

            choiceSimilarity = vectors.cosineSimilarity(choiceWord0Embedding, choiceWord1Embedding)

            choiceSimilarityDelta = abs(stemSimilarity - choiceSimilarity)
            choiceSimilarityDeltas.append(choiceSimilarityDelta)

            choiceIndex = numpy.argmin(choiceSimilarityDeltas)
            scores.append(int(choiceIndex == correctChoiceIndex))

    if len(scores) == 0:
        return numpy.nan

    metric = float(sum(scores)) / len(scores)

    return metric


def validate(wordIndexMap, embeddings):
    rg = rubensteinGoodenough(wordIndexMap, embeddings)
    sim353 = wordSimilarity353(wordIndexMap, embeddings)
    sl999 = simLex999(wordIndexMap, embeddings)
    syntRel = syntacticWordRelations(wordIndexMap, embeddings)
    sat = satQuestions(wordIndexMap, embeddings)

    return rg, sim353, sl999, syntRel, sat


def dump(metricsPath, epoch, customMetrics):
    metrics = {
        'epoch': epoch
    }

    for name, value in customMetrics.items():
        metrics[name] = value

    metrics = [metrics]

    if os.path.exists(metricsPath):
        with open(metricsPath, 'a') as metricsFile:
            metricsHistory = pandas.DataFrame.from_dict(metrics)
            metricsHistory.to_csv(metricsFile, header=False)
    else:
        metricsHistory = pandas.DataFrame.from_dict(metrics)
        metricsHistory.to_csv(metricsPath, header=True)


def compareMetrics(metricsHistoryPath, *metricNames):
    metrics = pandas.DataFrame.from_csv(metricsHistoryPath)
    iterations = range(0, len(metrics))

    plt.grid()

    metricScatters = []
    colorNames = colors.cnames.keys()
    for metricIndex, metricName in enumerate(metricNames):
        metric = metrics[metricName]

        random.shuffle(colorNames)
        metricScatter = plt.scatter(iterations, metric, c=colorNames[metricIndex % len(colorNames)])
        metricScatters.append(metricScatter)

    metricsFileName = os.path.basename(metricsHistoryPath)
    plt.title(metricsFileName)

    plt.legend(metricScatters, metricNames, scatterpoints=1, loc='lower right', ncol=3, fontsize=8)

    plt.show()


def compareHistories(metricName, *metricsHistoryPaths):
    plt.grid()

    metricScatters = []
    metricsHistoryNames = []
    colorNames = colors.cnames.keys()

    for metricsHistoryIndex, metricsHistoryPath in enumerate(metricsHistoryPaths):
        metrics = pandas.DataFrame.from_csv(metricsHistoryPath)
        iterations = range(0, len(metrics))
        metric = metrics[metricName]

        random.shuffle(colorNames)
        metricScatter = plt.scatter(iterations, metric, c=colorNames[metricsHistoryIndex % len(colorNames)])
        metricScatters.append(metricScatter)

        metricsHistoryName = os.path.basename(metricsHistoryPath)
        metricsHistoryNames.append(metricsHistoryName)

    plt.title(metricName)
    plt.legend(metricScatters, metricsHistoryNames, scatterpoints=1, loc='lower right', ncol=3, fontsize=8)

    plt.show()


def plotEmbeddings(fileIndexMap, embeddings):
    embeddingsCount, embeddingSize = embeddings.shape
    embeddings = numpy.asarray(embeddings, 'float64')
    lowDimEmbeddings = tsne.tsne(embeddings, 2, embeddingSize, 20.0, 1000)

    filePaths = fileIndexMap.keys()
    fileNames = [os.path.basename(filePath).split('.')[0] for filePath in filePaths]

    labels = set(fileNames)
    labels = zip(labels, numpy.arange(0, len(labels)))
    labels = [(label, index) for label, index in labels]
    labels = dict(labels)
    labels = [labels[fileName] for fileName in fileNames]

    lowDimEmbeddingsX, lowDimEmbeddingsY = lowDimEmbeddings[:,0], lowDimEmbeddings[:,1]

    figure, axis = plt.subplots()
    axis.scatter(lowDimEmbeddingsX, lowDimEmbeddingsY, 20, labels)

    for index, fileName in enumerate(fileNames):
        axis.annotate(fileName, (lowDimEmbeddingsX[index],lowDimEmbeddingsY[index]))

    plt.grid()
    plt.show()


def mapEmbeddings2LowDim(indexMap, embeddingsList):
    filePaths = indexMap.keys()
    fileNames = [os.path.basename(filePath).split('.')[0] for filePath in filePaths]

    labels = set(fileNames)
    labels = zip(labels, numpy.arange(0, len(labels)))
    labels = [(label, index) for label, index in labels]
    labels = dict(labels)
    labels = [labels[fileName] for fileName in fileNames]

    figure, axises = plt.subplots(1, len(embeddingsList))
    for embeddings, axis in zip(embeddingsList, axises):
        embeddingsCount, embeddingSize = embeddings.shape
        embeddings = numpy.asarray(embeddings, 'float64')
        lowDimEmbeddings = tsne.tsne(embeddings, 2, embeddingSize, 20.0, 1000)

        lowDimEmbeddingsX, lowDimEmbeddingsY = lowDimEmbeddings[:,0], lowDimEmbeddings[:,1]

        axis.grid()
        axis.scatter(lowDimEmbeddingsX, lowDimEmbeddingsY, 20, labels)

        for index, fileName in enumerate(fileNames):
            axis.annotate(fileName, (lowDimEmbeddingsX[index], lowDimEmbeddingsY[index]))

    figureManager = plt.get_current_fig_manager()
    figureManager.resize(*figureManager.window.maxsize())

    plt.show()


def compareEmbeddings(indexMap, embeddingsList, comparator=None, annotate=False, axisLabels=True):
    embeddingsCount = len(indexMap)
    embeddingIndices = numpy.arange(0, embeddingsCount)

    xy = [xy for xy in itertools.product(embeddingIndices, embeddingIndices)]
    xx, yy = zip(*xy)

    if comparator is None:
        comparator = lambda a, b: vectors.cosineSimilarity(a, b) + 1 / vectors.euclideanDistance(a, b)

    function = lambda xy: comparator(embeddingsList[xy[0]], embeddingsList[xy[1]]) if xy[0] != xy[1] else numpy.nan
    comparisons = map(function, xy)
    comparisons = numpy.reshape(comparisons, (embeddingsCount, embeddingsCount))

    nanxx, nanyy = numpy.where(numpy.isnan(comparisons))
    nanxy = zip(nanxx, nanyy)
    leftx = lambda x: max(x, 0)
    rightx = lambda x: min(x, comparisons.shape[0])
    lefty = lambda y: max(y, 0)
    righty = lambda y: min(y, comparisons.shape[1])
    for x, y in nanxy:
        neighbours = comparisons[leftx(x-1):rightx(x+2),lefty(y-1):righty(y+2)]
        neighbours = neighbours[neighbours > 0]
        comparisons[x,y] = numpy.mean(neighbours)

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)

    if axisLabels:
        filePaths = indexMap.keys()
        fileNames = [os.path.basename(filePath).split('.')[0] for filePath in filePaths]
        indices = [indexMap[filePath] for filePath in filePaths]

        plt.xticks(indices, fileNames, size='small', rotation='vertical')
        plt.yticks(indices, fileNames, size='small')

    plt.contourf(comparisons)

    if annotate:
        for x, y, c in zip(xx, yy, comparisons.flatten()):
            c = '{0:.1f}'.format(c*100)
            plt.annotate(c, (x, y))

    plt.show()


def buildEmbeddingsTree(indexMap, embeddings, comparator=None):
    embeddingsCount = len(embeddings)
    embeddingIndices = numpy.arange(0, embeddingsCount)
    xy = [xy for xy in itertools.product(embeddingIndices, embeddingIndices)]

    comparator = lambda a, b: vectors.euclideanDistance(a, b) + 1 / (2 + 2*vectors.cosineSimilarity(a, b))

    function = lambda xy: comparator(embeddings[xy[0]], embeddings[xy[1]]) if xy[0] != xy[1] else 0
    comparisons = map(function, xy)
    maxComparison = max(comparisons)
    comparisons = numpy.reshape(comparisons, (embeddingsCount, embeddingsCount)) / maxComparison
    comparisons = ssd.squareform(comparisons)
    links = linkage(comparisons)

    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.8)

    names = map(lambda nameIndexPair: nameIndexPair[0].split('/')[-1], indexMap.items())
    names = sorted(names)
    dendrogram(
        links,
        leaf_rotation=90.,
        leaf_font_size=8.,
        orientation='right',
        labels=names,
        show_contracted=True,
        show_leaf_counts=True)

    plt.show()