from chatbot.textdata import TextData


class Argument(object):

    def __init__(self):
        self.test = 'interactive'
        self.createDataset = False
        self.playDataset = None
        self.reset = False
        self.verbose = False
        self.debug = False
        self.keepAll = True
        self.modelTag = 'cornell-tf1.3'
        self.rootDir = None
        self.watsonMode = False
        self.autoEncode = False
        self.device = None
        self.seed = None

        self.corpus = TextData.corpusChoices()[0]
        self.datasetTag = ''
        self.ratioDataset = 1.0
        self.maxLength = 10
        self.filterVocab = 1
        self.skipLines = False
        self.vocabularySize = 40000

        self.hiddenSize = 512
        self.numLayers = 2
        self.initEmbeddings = False
        self.embeddingSize = 64
        self.embeddingSource = "GoogleNews-vectors-negative300.bin"

        self.numEpochs = 30
        self.saveEvery = 2000
        self.batchSize = 256
        self.learningRate = 0.002
        self.dropout = 0.9
