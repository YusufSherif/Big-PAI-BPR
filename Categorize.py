from math import ceil

import json
import os
from threading import Thread, Lock


class Categorize:
    def __init__(self):

        self.mutex = Lock()

        self.cat_dict = {"": [], "ブローチ": [], "リュック": [], "ルームウェア": [], "ヘアアクセサリー": [], "ハット": [], "コート": [],
                         "ボディケア": [], "ロングスカート": [], "チュニック": [], "Tシャツ": [], "キャップ": [], "靴": [], "インテリア": [],
                         "ワンピース": [], "ブーツ": [], "カーディガン": [], "ショートパンツ": [], "トップス": [], "バッグ": [], "コスメ": [],
                         "ダウンジャケット": [], "スニーカー": [], "パンプス": [], "財布": [], "ファッション小物": [], "ロングパンツ": [],
                         "ショルダーバッグ": [], "ストール": [], "スカート": [], "小物": [], "パーカー": [], "浴衣": [], "ブレスレット": [],
                         "ボストンバッグ": [], "ジャケット": [], "タンクトップ": [], "リング": [], "腕時計": [], "サンダル": [], "クラッチバッグ": [],
                         "ベルト": [], "フレグランス": [], "ニット": [], "ルームシューズ": [], "ステーショナリー": [], "手袋": [], "帽子": [],
                         "レッグウェア": [], "傘": [], "トートバッグ": [], "メガネ": [], "水着": [], "アクセサリー": [], "ピアス": [], "ネックレス": [],
                         "サングラス": [], "ネイル": [], "ブラウス": [], "ハンドバッグ": [], "ニット帽": [], "アンダーウェア": []}

        self.fileName = 'categorized.json'

        file_exists = os.path.isfile(self.fileName)

        if (not file_exists):
            f = open(self.fileName, 'w')
            f.close()

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
            threads.append(Thread(target=self.categorize,
                                  args=(outfitsJson_partitioned[i],)))

        for i in range(n):
            threads[i].start()

        for i in range(n):
            threads[i].join()

        self.writeToFile()

    def categorize(self, outfitsJSON):
        offset = len(r'C:\Users\CSE-P07-2176-G13\Desktop\IQON3000\IQON3000')
        for outfitJSON in outfitsJSON:
            try:
                d = dict()
                f = open(outfitJSON)
                outfit = json.load(f)
                f.close()
                for i in outfit['items']:
                    self.mutex.acquire()
                    self.cat_dict[i['category x color'].split('×')[0].strip()].append({"path": outfitJSON[offset:],
                                                                                       "itemId": i['itemId']})
                    self.mutex.release()
            except json.decoder.JSONDecodeError:
                pass

    def writeToFile(self):
        with open(self.fileName, 'r+') as file:
            data = self.cat_dict
            file.seek(0)
            json.dump(data, file)


tr = Categorize()
