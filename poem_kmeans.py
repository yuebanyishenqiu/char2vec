import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
import pickle
import codecs
import json
import random
import sys
import os

# kmenas algorithm for doc2vec_pith
def kmeans_poem(FilePath):
    fp = codecs.open(FilePath, "rb")
    dic = pickle.load(fp)
    # X 用于存放所有宋词的doc2vec
    X = []
    for i in dic:
            X.append(dic[i]["doc2vec"])
    # 将 X 转化成 array类型
    X = np.array(X)
    print(X.shape)
    #迭代100次 选出分数最高的聚类总数K
    max = 0
    for j in range(2,100):
        n_clusters = j
        y_pred = KMeans(n_clusters, random_state=4).fit_predict(X)
        # 用calinski_harabasz_score 对聚类结果进行评估
        error = metrics.calinski_harabasz_score(X,y_pred)
        if error > max:
            max = error
            index = j
        print("n_cluster ", j, "calinski_harabasz_score:",error)
    n_clusters = index
    print("optimal K：",index)
    print("calinski_harabasz_score:", max )
    y_pred = KMeans(n_clusters, random_state=4).fit_predict(X)
    y_raw = y_pred

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: {} doc2vec_dir".format(sys.argv[0]))
        exit(0)
    kmeans_poem(sys.argv[1])

