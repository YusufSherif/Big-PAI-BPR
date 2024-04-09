from math import ceil

from googletrans import Translator
import json, os
from threading import Thread, Lock
import httpx

class TranslateJSONs:
    def __init__(self):

        self.mutex = Lock()

        self.done = {}

        timeout = httpx.Timeout(5)
        self.translator = Translator(timeout=timeout)

        self.doneFileName = 'data.json'

        file_exists = os.path.isfile(self.doneFileName)

        if(file_exists):
            with open(self.doneFileName) as json_file:
                self.done = json.load(json_file)
        else:
            f = open(self.doneFileName,'w')
            f.write('{"files":[]}')
            f.close()

            with open(self.doneFileName) as json_file:
                self.done = json.load(json_file)

        path = './IQON3000.nosync/'
        usersDirs = [f.path for f in os.scandir(path) if f.is_dir()]
        outfitsDirs = []
        self.outfitsJSON = []

        for userDir in usersDirs:
            outfitsDirs.extend([f.path for f in os.scandir(userDir) if f.is_dir()])

        for outfitDir in outfitsDirs[1:105]:
            self.outfitsJSON.extend([each.path for each in os.scandir(outfitDir) if os.path.splitext(each.name)[1]=='.json'])


        outfitsJson_partitioned = []

        threads = []
        n = 1
        i = 0
        while (i < len(self.outfitsJSON)):
            j = i
            length = len(self.outfitsJSON)
            i = (i+1) * ceil(length/ n)
            outfitsJson_partitioned.append(self.outfitsJSON[j:i])

        for i in range(n):
            threads.append(Thread(target=self.translate, args=(outfitsJson_partitioned[i],)))

        for i in range(n):
            threads[i].start()

        for i in range(n):
            threads[i].join()


    def translate(self, outfitsJSON):
        for outfitJSON in outfitsJSON:
            if outfitJSON not in self.done['files']:
                f = open(outfitJSON)
                outfit = json.load(f)
                f.close()
                for item_index in range(len(outfit['items'])):
                    arr = []
                    arr.append(outfit['items'][item_index]['category x color'])
                    arr.append(outfit['items'][item_index]['itemName'])
                    arr.extend([j for i in outfit['items'][item_index]['breadcrumb'] for j in i])
                    arr.extend([i for i in outfit['items'][item_index]['brands']])
                    arr.extend([i for i in outfit['items'][item_index]['categorys']])
                    arr.extend([i for i in outfit['items'][item_index]['options']])
                    arr.extend([i for i in outfit['items'][item_index]['colors']])
                    arr.extend([i for i in outfit['items'][item_index]['expressions']])

                    x = self.translator.translate(arr, src='ja', dest='en')

                    offset = 0

                    outfit['items'][item_index]['category x color'] = x[offset].text
                    offset+=1

                    outfit['items'][item_index]['itemName'] = x[offset].text
                    offset+=1

                    for i in range(len(outfit['items'][item_index]['breadcrumb'])):
                        outfit['items'][item_index]['breadcrumb'][i] = x[offset].text
                        offset+=1

                    for i in range(len(outfit['items'][item_index]['brands'])):
                        outfit['items'][item_index]['brands'][i] = x[offset].text
                        offset += 1

                    for i in range(len(outfit['items'][item_index]['categorys'])):
                        outfit['items'][item_index]['categorys'][i] = x[offset].text
                        offset += 1

                    for i in range(len(outfit['items'][item_index]['options'])):
                        outfit['items'][item_index]['options'][i] = x[offset].text
                        offset += 1

                    for i in range(len(outfit['items'][item_index]['colors'])):
                        outfit['items'][item_index]['colors'][i] = x[offset].text
                        offset += 1

                    for i in range(len(outfit['items'][item_index]['expressions'])):
                        outfit['items'][item_index]['expressions'][i] = x[offset].text
                        offset+=1

                self.updateDoneFile(outfitJSON)

                with open(outfitJSON, 'w+') as file:
                    file.seek(0)
                    file.truncate(0)
                    json.dump(outfit, file)

                print("Updated: ", outfitJSON)




    def updateDoneFile(self,string):
        self.mutex.acquire()
        with open(self.doneFileName, 'r+') as file:
            data = json.load(file)
            data['files'].append(string)
            file.seek(0)
            json.dump(data, file)
        self.mutex.release()


tr = TranslateJSONs()
