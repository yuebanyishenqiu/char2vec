import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
import pickle
import codecs
import json
import random
import sys
import os

# hierarchical clustering algorithm for doc2vec_path
def hclust_poem(FilePath):
    fp = codecs.open(FilePath, "rb")
    dic = pickle.load(fp)
    # X 用于存放所有宋词的doc2vec
    X = []
    for i in dic:
            X.append(dic[i]["doc2vec"])
    # 将 X 转化成 array类型
    X = np.array(X)
    print("make .json file...")
    #迭代100次 选出分数最高的聚类总数K
    max = 0
    for j in range(2,100):
        n_clusters = j
        model = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward')
        y_pred = model.fit_predict(X)
        # 用calinski_harabasz_score 对聚类结果进行评估
        error = metrics.calinski_harabasz_score(X,y_pred)
        if error > max:
            max = error
            index = j
        # print("n_cluster ", j, "calinski_harabasz_score :",error)
    n_clusters = index
    # print("optimal K：",index)
    # print("calinski_harabasz_score:", max )
    y_pred = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward').fit_predict(X)
    y_raw = y_pred
    #print("make poem.json file ...")

    # position 存放每个类别的元素在宋词的序号
    position = {}
    num = {}
    for key in y_pred:
        num[key] = num.get(key, 0) + 1
        position[key] = []
    for i in range(len(y_pred)):
        position[y_pred[i]].append(i)

    dictionary = {}
    dictionary["name"] = "poems"
    # dic["children"] 是个列表，列表的数量为聚类后的类别数量，列表中的每个元素为一个类
    dictionary["children"] = []

    # 对于类别2，3，所含的宋词数量较少不需要随机取样
    for n in range(len(position)):
        name = "cluster{}".format(n)
        # sub_dic这个字典代表一个类别
        sub_dic = {}
        # 当前第i个类别名字为cluster${i}
        sub_dic["name"] = name
        # sub_dic["children"]为列表，列表长度为当前类别样本数量，列表每个元素为属于这个类别的样本
        sub_dic["children"] = []
        for i in range(len(position[n])):
            if position[n][i] in dic:
                sub_name = "{}".format(dic[position[n][i]]["author"] + "-" + dic[position[n][i]]["rhythmic"])
                sub_value = i
                # 样本名字为 name${i}，名字为”作者-词牌名“
                # value数值不重要，随意给
                sub_dic["children"].append({"name": sub_name, "value": sub_value})
            dictionary["children"].append(sub_dic)

    # 对于类别0，1，4，所含的宋词数量较多（全部可视化较困难，所以随即取出每个类别中的100个宋词）
    # for n in [0, 1, 4]:
    #     name = "cluster{}".format(n)
    #     # sub_dic这个字典代表一个类别
    #     sub_dic = {}
    #     # 当前第i个类别名字为cluster${i}
    #     sub_dic["name"] = name
    #     # sub_dic["children"]为列表，列表长度为当前类别样本数量，列表每个元素为属于这个类别的样本
    #     sub_dic["children"] = []
    #     cluster_num = []
    #     cluster_num = main1(position[n], 100)
    #     for i in range(len(cluster_num)):
    #         if position[n][i] in dic:
    #             sub_name = "{}".format(dic[position[n][i]]["author"] + "-" + dic[position[n][i]]["rhythmic"])
    #             sub_value = i
    #             # 样本名字为 name${i}，名字为”作者-词牌名“
    #             # value数值不重要，随意给
    #             sub_dic["children"].append({"name": sub_name, "value": sub_value})
    #         dictionary["children"].append(sub_dic)

    clusters = dictionary["children"]
    s = []
    for key in clusters:
        if key not in s:
            s.append(key)
    dictionary["children"] = s
    visualization_name = "{}.json".format(FilePath)
    name_var = visualization_name[14:]
    fp = codecs.open("{}".format(name_var), "w", encoding="utf8")
    json.dump(dictionary, fp, ensure_ascii=False)
    fp.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: {} doc2vec_dir".format(sys.argv[0]))
        exit(0)
    hclust_poem(sys.argv[1])
    print("poem.json file succeeded!")
