import json
filename = 'categorized.json'
data = {}
with open(filename) as json_file:
    data = json.load(json_file)

"""
 "ロングスカート" bottom
 "ショートパンツ" bottom
 "ロングパンツ" bottom
 "スカート" bottom
 "レッグウェア" bottom

 "チュニック" top
 "Tシャツ" top
 "カーディガン" top
 "トップス" top
 "パーカー" top
 "タンクトップ" top
 "ニット" top
 "ブラウス" top
"""

bottoms = set()
tops = set()
paths = list()


def do_add(s, x):
    l = len(s)
    s.add(x)
    return len(s) != l


for key, array in data.items():
    if (key in ["ロングスカート", "ショートパンツ", "ロングパンツ", "スカート", "レッグウェア"]):
        for i in array:
            if(do_add(bottoms, (i['itemId']))):
                paths.append({"path": '\\'.join(i['path'].split('\\')[
                             1:3]), "itemId": i['itemId']})

    elif(key in ["チュニック", "Tシャツ", "カーディガン", "トップス", "パーカー", "タンクトップ", "ニット", "ブラウス"]):
        for i in array:
            if(do_add(tops, (i['itemId']))):
                paths.append({"path": '\\'.join(i['path'].split('\\')[
                             1:3]), "itemId": i['itemId']})

all = dict()

all["tops"] = list(tops)
all["bottoms"] = list(bottoms)

outFileName = 'TopsAndBottoms.json'
f = open(outFileName, 'w')
f.close()

with open(outFileName, 'r+') as file:
    data = all
    file.seek(0)
    json.dump(data, file)

f = open('kmeans_todo.json', 'w')
f.close()

with open('kmeans_todo.json', 'r+') as file:
    data = paths
    file.seek(0)
    json.dump(data, file)

print('Top count: ', len(tops))
print('Bottom count: ', len(bottoms))
