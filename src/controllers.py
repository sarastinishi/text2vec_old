from multiprocessing import Process

import kit
import parameters
import connectors
import extraction
import weeding
import processing
import traininig


class DataPreparationController:
    def launch(self, pathTo, hyper):
        targets = [extraction.launch, weeding.launch, processing.launch]

        for target in targets:
            proces = Process(target=target, args=(pathTo, hyper))
            proces.start()
            proces.join()

        traininig.launch(pathTo, hyper)


if __name__ == '__main__':
    pathTo = kit.PathTo('Duplicates', experiment='duplicates', w2vEmbeddings='wiki_full_s1000_w10_mc20_hs1.bin')
    hyper = parameters.HyperParameters(
        connector = connectors.TextFilesConnector(pathTo.dataSetDir),
        threshold=2e-2,
        minCount=1,
        windowSize=3,
        negative=200,
        strict=False,
        fileEmbeddingSize=1000,
        epochs=50,
        batchSize=1,
        learningRate=0.025,
        contextsPerText=600
    )

    controller = DataPreparationController()
    controller.launch(pathTo, hyper)