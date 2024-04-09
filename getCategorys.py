from math import ceil

import json
import os
from threading import Thread, Lock


class Categorize:
    def __init__(self):

        self.mutex = Lock()

        self.setAccessCounter = 0

        self.doneFileName = 'categorys.json'

        self.done = {}

        file_exists = os.path.isfile(self.doneFileName)

        if (not file_exists):
            f = open(self.doneFileName, 'w')
            f.write('{"categorys":[]}')
            f.close()

            with open(self.doneFileName) as json_file:
                self.done = json.load(json_file)

        self.cat_set = set()

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
        for outfitJSON in outfitsJSON:
            try:
                f = open(outfitJSON)
                outfit = json.load(f)
                f.close()
                for i in outfit['items']:

                    cat = i['category x color'].split('×')[0].strip()
                    self.updateSet(cat)

            except json.decoder.JSONDecodeError:
                pass

    def updateSet(self, string):
        self.mutex.acquire()
        self.setAccessCounter += 1
        self.cat_set.add(string)
        self.mutex.release()

    def writeToFile(self):
        with open(self.doneFileName, 'r+') as file:
            data = (list(self.cat_set))
            file.seek(0)
            json.dump(data, file)


tr = Categorize()
