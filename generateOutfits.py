from math import ceil
import json
import os
from threading import Thread, Lock


class outfit:
    def __init__(self):
        self.user = ''
        self.top = ''
        self.posotiveBottom = ''
        self.negativeBottom = ''


class generateOutfits:
    def __init__(self):
        self.tops = {}

        self.dictMutex = Lock()
        self.outfitsMutes = Lock()

        self.topNames = ["チュニック", "Tシャツ", "カーディガン", "トップス", "パーカー", "タンクトップ", "ニット", "ブラウス"]
        self.bottomNames = ["ロングスカート", "ショートパンツ", "ロングパンツ", "スカート", "レッグウェア"]

        self.topsNdBottoms = dict()
        self.outfits = []

        self.doneFileName = 'OutiftsNdNegatives.json'

        filename = 'TopsAndBottoms.json'
        with open(filename) as json_file:
            self.topsNdBottoms = json.load(json_file)

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
            outfitsJson_partitioned.append(self.outfitsJSON[j:j + i])
            j += i

        for i in range(n):
            threads.append(Thread(target=self.genOutfits,
                                  args=(outfitsJson_partitioned[i],)))

        for i in range(n):
            threads[i].start()

        for i in range(n):
            threads[i].join()

        self.writeToFile()

    def genOutfits(self, outfitsJSON):
        offset = len(r'C:\Users\CSE-P07-2176-G13\Desktop\IQON3000\IQON3000')
        # Generate positive bottoms for said top - top = [{'userId','bottomId'},]
        for outfitJSON in outfitsJSON:
            try:
                userId, outfitId, = outfitJSON[offset:].split('\\')[1:3]

                f = open(outfitJSON)
                outfit = json.load(f)
                f.close()
                topFound = False
                bottomArray = []
                topKey = ''
                for i in outfit['items']:
                    itemCat = i['category x color'].split('×')[0].strip()
                    key = i['itemId']

                    if itemCat in self.topNames:
                        topFound = True
                        topKey = key
                        if key not in self.topsNdBottoms:
                            self.dictMutex.acquire()
                            self.topsNdBottoms[key] = set()
                            self.dictMutex.release()

                    elif itemCat in self.bottomNames:
                        bottomArray.append(key)

                if topFound:
                    self.dictMutex.acquire()
                    self.topsNdBottoms[topKey].update(bottomArray)
                    self.dictMutex.release()

                    if len(bottomArray):
                        for j in bottomArray:
                            self.outfitsMutes.acquire()
                            self.outfits.append({"top": topKey, "bottom": j, "userId": userId, "outfitId": outfitId})
                            self.outfitsMutes.release()

            except json.decoder.JSONDecodeError:
                pass

    def writeToFile(self):
        with open(self.doneFileName, 'r+') as file:
            data = ({'topsNdBottoms': self.topsNdBottoms, 'outfits': self.outfits})
            file.seek(0)
            json.dump(data, file)
