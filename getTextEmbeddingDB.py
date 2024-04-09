import json
from threading import Lock, Thread
import os
from math import ceil

from fugashi import Tagger
import gensim

import numpy as np
import torch

import re

exactCount = 0
simCount = 0


class word2vec():
    def __init__(self):
        self.vectors = gensim.models.KeyedVectors.load(
            r"C:\Users\CSE-P07-2176-G13\Desktop\IQON3000\chive-1.2-mc5_gensim\chive-1.2-mc5.kv")

        self.exactCount = 0
        self.simCount = 0
        self.mutex = Lock()

    def getEmbedding(self, word):
        try:
            self.exactCount += 1
            return self.vectors.get_vector(word)
        except:
            try:
                self.simCount += 1
                return self.vectors.similar_by_word(word, topn=1)[0]
            except:
                return np.zeros(300)

    def get_vectors(self):
        return self.vectors.vectors


class Categorize:
    def __init__(self):

        self.word2vec = word2vec()

        self.mutex = Lock()

        self.setAccessCounter = 0

        self.doneFileName = 'dictionary'

        self.done = {}

        self.regex = re.compile('[^ａ-ｚＡ-Ｚ０-９a-zA-Z0-9ァ-ヴーぁ-ゔ一-龠々〆〤]')

        file_exists = os.path.isfile(self.doneFileName)

        if (not file_exists):
            f = open(self.doneFileName, 'w')
            f.close()

            with open(self.doneFileName) as json_file:
                self.done = json.load(json_file)

        self.cat_set = dict()

        path = r'C:\Users\CSE-P07-2176-G13\Desktop\IQON3000\IQON3000'
        usersDirs = [f.path for f in os.scandir(path) if f.is_dir()]
        outfitsDirs = []
        self.outfitsJSON = []

        for userDir in usersDirs:
            outfitsDirs.extend(
                [f.path for f in os.scandir(userDir) if f.is_dir()])

        for outfitDir in outfitsDirs:
            self.outfitsJSON.extend([each.path for each in os.scandir(
                outfitDir) if os.path.splitext(each.name)[1] == '.json'])

        outfitsJson_partitioned = []

        threads = []
        n = 16
        length = len(self.outfitsJSON)
        i = ceil(length / n)
        j = 0

        for k in range(n):
            outfitsJson_partitioned.append(self.outfitsJSON[j:j+i])
            j += i

        for i in range(n):
            threads.append(Thread(target=self.getCategorys,
                                  args=(outfitsJson_partitioned[i],)))

        for i in range(n):
            threads[i].start()

        for i in range(n):
            threads[i].join()

        self.writeToFile()

    def getCategorys(self, outfitsJSON):
        zeros = np.zeros(300)
        tagger = Tagger()
        for outfitJSON in outfitsJSON:
            try:
                f = open(outfitJSON)
                outfit = json.load(f)
                f.close()
                for i in outfit['items']:
                    result = self.regex.sub(' ', i['itemName'])
                    list_of_tokens = tagger(result)
                    for word in list_of_tokens:
                        vector = self.word2vec.getEmbedding(str(word))
                        if (not np.array_equal(zeros, vector)):
                            self.updateSet(str(word), vector)

            except json.decoder.JSONDecodeError:
                pass

    def updateSet(self, key, array):
        self.mutex.acquire()
        self.setAccessCounter += 1
        self.cat_set[key] = array
        self.mutex.release()

    def writeToFile(self):
        for key, value in self.cat_set.items():
            self.cat_set[key] = torch.FloatTensor(value.tolist())

        torch.save(self.cat_set, 'textDict')


tr = Categorize()
"""
import json
from threading import Lock, Thread
import os
from math import ceil

from fugashi import Tagger
import gensim

import numpy as np
import torch

import re

exactCount = 0
simCount = 0


class word2vec():
    def __init__(self):
        self.vectors = torch.load('smallnwjc2vec')
        self.exactCount = 0
        self.simCount = 0
        self.mutex = Lock()

    def getEmbedding(self, word):
        try:
            self.exactCount += 1
            return self.vectors[word]
        except:
            return np.zeros(300)
            try:
                self.simCount += 1
                return self.vectors.similar_by_word(word, topn=1)[0]
            except:
                return np.zeros(300)

    def get_vectors(self):
        return self.vectors.vectors


class Categorize:
    def __init__(self):

        self.word2vec = word2vec()

        self.mutex = Lock()

        self.setAccessCounter = 0

        self.doneFileName = 'dictionary'

        self.done = {}

        self.regex = re.compile('[^ａ-ｚＡ-Ｚ０-９a-zA-Z0-9ァ-ヴーぁ-ゔ一-龠々〆〤]')

        file_exists = os.path.isfile(self.doneFileName)

        if (not file_exists):
            f = open(self.doneFileName, 'w')
            f.close()

            with open(self.doneFileName) as json_file:
                self.done = json.load(json_file)

        self.cat_set = dict()

        path = r'C:\Users\CSE-P07-2176-G13\Desktop\IQON3000\IQON3000'
        usersDirs = [f.path for f in os.scandir(path) if f.is_dir()]
        outfitsDirs = []
        self.outfitsJSON = []

        for userDir in usersDirs:
            outfitsDirs.extend(
                [f.path for f in os.scandir(userDir) if f.is_dir()])

        for outfitDir in outfitsDirs:
            self.outfitsJSON.extend([each.path for each in os.scandir(
                outfitDir) if os.path.splitext(each.name)[1] == '.json'])

        outfitsJson_partitioned = []

        threads = []
        n = 16
        length = len(self.outfitsJSON)
        i = ceil(length / n)
        j = 0

        for k in range(n):
            outfitsJson_partitioned.append(self.outfitsJSON[j:j+i])
            j += i

        for i in range(n):
            threads.append(Thread(target=self.getCategorys,
                                  args=(outfitsJson_partitioned[i],)))

        for i in range(n):
            threads[i].start()

        for i in range(n):
            threads[i].join()

        self.writeToFile()

    def getCategorys(self, outfitsJSON):
        zeros = np.zeros(300)
        tagger = Tagger()
        for outfitJSON in outfitsJSON:
            try:
                f = open(outfitJSON)
                outfit = json.load(f)
                f.close()
                for i in outfit['items']:
                    result = self.regex.sub(' ', i['itemName'])
                    list_of_tokens = tagger(result)
                    for word in list_of_tokens:
                        vector = self.word2vec.getEmbedding(str(word))
                        if (not np.array_equal(zeros, vector)):
                            self.updateSet(str(word), vector)

            except json.decoder.JSONDecodeError:
                pass

    def updateSet(self, key, array):
        self.mutex.acquire()
        self.setAccessCounter += 1
        self.cat_set[key] = array
        self.mutex.release()

    def writeToFile(self):
        for key, value in self.cat_set.items():
            self.cat_set[key] = value

        torch.save(self.cat_set, 'textDictNWJC')


tr = Categorize()

"""