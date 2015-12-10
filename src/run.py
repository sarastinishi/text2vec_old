from binary import *
from parameters import *
from validation import *
from vectors import *
import kit
import math


if __name__ == '__main__':
    pathTo = kit.PathTo('Cockatoo', experiment='cockatoo', w2vEmbeddings='wiki_full_s1000_w10_mc20_hs1.bin')
    # pathTo = kit.PathTo('Duplicates', experiment='duplicates', w2vEmbeddings='wiki_full_s1000_w10_mc20_hs1.bin')
    textIndexMap = loadMap(pathTo.textIndexMap)
    fileEmbeddings = loadTensor(pathTo.fileEmbeddings)

    comparator = lambda a, b: cosineSimilarity(a, b) / math.pow(euclideanDistance(a, b), 2)
    # comparator = lambda a, b: cosineSimilarity(a, b)

    compareEmbeddings(textIndexMap, fileEmbeddings, comparator=comparator, annotate=False, axisLabels=True)
    # buildEmbeddingsTree(textIndexMap, fileEmbeddings, comparator=comparator)
    # compareMetrics(pathTo.metrics('history.csv'), 'meanError', 'medianError', 'minError', 'maxError')
