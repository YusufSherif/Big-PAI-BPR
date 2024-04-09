from math import ceil
import json
import os
from threading import Thread, Lock
import random

class GenerateDataset:
    def __init__(self):
        self.tops = {}

        self.dictMutex = Lock()

        self.outList = []

        self.topsNdBottomsJSON = dict()
        self.outfits = []

        filename = 'TopsAndBottoms.json'
        with open(filename, 'r') as file:
            self.topsNdBottomsJSON = json.load(file)

        self.bottoms = set(self.topsNdBottomsJSON['bottoms'])

        filename = 'outfits.json'
        with open(filename, 'r') as file:
            self.outfits = json.load(file)

        self.matching_bottoms = self.outfits['topsNdBottoms']

        for i in self.matching_bottoms:
            self.matching_bottoms[i] = set(self.matching_bottoms[i])

        self.doneFileName = 'dataset.json'

        file_exists = os.path.isfile(self.doneFileName)

        if (not file_exists):
            f = open(self.doneFileName, 'w')
            f.close()

        partitions = []

        threads = []
        n = 32
        length = len(self.outfits['outfits'])
        i = ceil(length / n)
        j = 0

        for k in range(n):
            partitions.append(self.outfits['outfits'][j:j + i])
            j += i

        for i in range(n):
            threads.append(Thread(target=self.genDataset,
                                  args=(partitions[i],)))

        for i in range(n):
            threads[i].start()

        for i in range(n):
            threads[i].join()

        self.writeToFile()

    def genDataset(self, outfits):
        for outfit in outfits:
            newOutfit = dict()
            newOutfit['top'], newOutfit['bottom'], newOutfit['userId'] = outfit['top'],outfit['bottom'],outfit['userId']
            newOutfit['negative-bottom'] = random.choice(tuple(self.bottoms.difference(self.matching_bottoms[outfit['top']])))
            self.dictMutex.acquire()
            self.outList.append(newOutfit)
            self.dictMutex.release()

    def writeToFile(self):
        class SetEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, set):
                    return list(obj)
                return json.JSONEncoder.default(self, obj)

        with open(self.doneFileName, 'r+') as file:
            data = self.outList
            file.seek(0)
            json.dump(data, file, cls=SetEncoder)


s = GenerateDataset()
