import os
from os.path import join


class PathTo:
    def __init__(self, datasetName, experiment='default', w2vEmbeddings=''):
        self.datasetName = datasetName
        self.experiment = experiment
        self.w2vEmbeddings = w2vEmbeddings
        self.dataDir = '../data'
        self.dataSetDir = join(self.dataDir, 'Datasets', datasetName)
        self.experimentsDir = join(self.dataDir, 'Experiments')
        self.experimentDir = join(self.experimentsDir, experiment)

        self.w2vEmbeddingsDir = join(self.dataDir, 'WordEmbeddings')
        self.extractedDir = join(self.experimentDir, 'Extracted')
        self.weededDir = join(self.experimentDir, 'Weeded')
        self.concatenatedDir = join(self.experimentDir, 'Concatenated')
        self.processedDir = join(self.experimentDir, 'Processed')
        self.parametersDir = join(self.experimentDir, 'Parameters')
        self.metricsDir = join(self.experimentDir, 'Metrics')

        self.ensureDirectories(
            self.experimentsDir,
            self.experimentDir,
            self.extractedDir,
            self.weededDir,
            self.concatenatedDir,
            self.processedDir,
            self.parametersDir,
            self.metricsDir,
            self.w2vEmbeddingsDir)

        self.concatenated = join(self.concatenatedDir, 'concatenated.txt')
        self.contexts = join(self.processedDir, 'contexts.bin')
        self.textIndexMap = join(self.parametersDir, 'file_index_map.bin')
        self.fileEmbeddings = join(self.parametersDir, 'file_embeddings.bin')
        self.wordIndexMap = join(self.parametersDir, 'word_index_map.bin')
        self.wordFrequencyMap = join(self.parametersDir, 'word_frequency_map.bin')
        self.wordEmbeddings = join(self.parametersDir, 'word_embeddings.bin')
        self.weights = join(self.parametersDir, 'weights.bin')
        self.w2vEmbeddings = join(self.w2vEmbeddingsDir, self.w2vEmbeddings)


    @staticmethod
    def ensureDirectories(*directories):
        for directory in directories:
            if not os.path.exists(directory):
                os.mkdir(directory)
                os.chown(directory, 1000, 1000)


    def metrics(self, fileName):
        return join(self.metricsDir, fileName)