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
    print(X.shape)
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
        print("n_clusters ", j, ":",error)
    n_clusters = index
    print("optimal K：",index)
    print("calinski_harabasz_score:", max )
    y_pred = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward').fit_predict(X)
    y_raw = y_pred

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: {} doc2vec_dir".format(sys.argv[0]))
        exit(0)
    hclust_poem(sys.argv[1])

