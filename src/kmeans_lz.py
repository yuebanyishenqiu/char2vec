#!/usr/bin/env python
# coding: utf-8

# ## 小组成员：李柱，学号：201921198685；负责对训练好的doc2vec进行Kmeans聚类

# #### 需要训练好宋词的词向量文件（1）不带图像信息的 char2vec_raw.pickle（2）带有图像信息的 char2vec_with_glyce.pickle

# #### 使用sklearn下的Kmeans进行聚类，并且迭代100次选出最佳的聚类数目K，最后对聚好的类进行可视化

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
import pickle
import codecs
import json
import random


# #### 1，对不带图像信息的 char2vec_raw.pickle 进行聚类

# In[2]:


#读取训练好的词向量文件
fin = "/tmp/char2vec/char2vec_raw.pickle"
fp = codecs.open(fin, "rb")
dic = pickle.load(fp)
# X 用于存放所有宋词的doc2vec 
X = []
for i in dic:
        X.append(dic[i]["doc2vec"])
# 将 X 转化成 array类型
X = np.array(X)

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
    print("n_cluster ", j, ":",error)
n_clusters = index
print("optimal K：",index)
print("calinski_harabasz_score:", max )
y_pred = KMeans(n_clusters, random_state=4).fit_predict(X)
y_raw = y_pred


# #### 2,对带有图像信息的 char2vec_with_glyce.pickle进行聚类

# In[3]:


#读取训练好的词向量文件
fin = "/tmp/char2vec/char2vec_with_glyce.pickle"
fp = codecs.open(fin, "rb")
dic = pickle.load(fp)
# X 用于存放所有宋词的doc2vec 
X = []
for i in dic:
        X.append(dic[i]["doc2vec"])
# 将 X 转化成 array类型
X = np.array(X)

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
    print("n_cluster ", j, ":",error)
n_clusters = index
print("optimal K：",index)
print("calinski_harabasz_score:", max )
y_pred = KMeans(n_clusters, random_state=4).fit_predict(X)
y_glyce = y_pred


# #### 3,对聚类结果（以加图像信息的char2vec_with_glyce.pickle为例）进行可视化（生成json文件在d3中进行可视化）

# #### 注意由于聚类结果的类别较少，每个类别中的宋词数目，在可视化中较困难，所以对含有较多宋词数目类别进行随机取样（随机该类别中100首宋词进行可视化）

# In[8]:


def main1(data, num):
    """
    实现从data中随机取num个元素，生成一个新的列表
    原因是把聚类的全部结果加进去，文件会很大（可视化不支持）
    """
    return random.sample(data, num)

# posion 存放每个类别的元素在宋词的序号
position = {}
num = {}
for key in y_pred:
    num[key] = num.get(key,0) + 1
    position[key] = []
for i in range(len(y_glyce)):
    position[y_glyce[i]].append(i)
    
    
dictionary = {}
dictionary["name"] = "poems"
#dic["children"] 是个列表，列表的数量为聚类后的类别数量，列表中的每个元素为一个类
dictionary["children"] = []

# 对于类别2，3，所含的宋词数量较少不需要随机取样
for n in [2,3]:
    name = "cluster{}".format(n)
    # sub_dic这个字典代表一个类别
    sub_dic = {}
    # 当前第i个类别名字为cluster${i}
    sub_dic["name"] = name
    # sub_dic["children"]为列表，列表长度为当前类别样本数量，列表每个元素为属于这个类别的样本
    sub_dic["children"] = []
    for i in range(len(position[n])):
        if position[n][i] in dic:
            sub_name = "{}".format(dic[position[n][i]]["author"]+ "-" + dic[position[n][i]]["rhythmic"])
            sub_value = i
        # 样本名字为 name${i}，名字为”作者-词牌名“ 
        # value数值不重要，随意给
            sub_dic["children"].append({"name": sub_name, "value": sub_value})
        dictionary["children"].append(sub_dic)

# 对于类别0，1，4，所含的宋词数量较多（全部可视化较困难，所以随即取出每个类别中的100个宋词）        
for n in [0,1,4]:
    name = "cluster{}".format(n)
    # sub_dic这个字典代表一个类别
    sub_dic = {}
    # 当前第i个类别名字为cluster${i}
    sub_dic["name"] = name
    # sub_dic["children"]为列表，列表长度为当前类别样本数量，列表每个元素为属于这个类别的样本
    sub_dic["children"] = []
    cluster_num = []
    cluster_num = main1(position[n],100)
    for i in range(len(cluster_num)):
        if position[n][i] in dic:
            sub_name = "{}".format(dic[position[n][i]]["author"]+ "-" + dic[position[n][i]]["rhythmic"])
            sub_value = i
        # 样本名字为 name${i}，名字为”作者-词牌名“
        # value数值不重要，随意给
            sub_dic["children"].append({"name": sub_name, "value": sub_value})
        dictionary["children"].append(sub_dic)

clusters = dictionary["children"]
s = []
for key in clusters:
    if key not in s:
        s.append(key)
dictionary["children"] = s

fp = codecs.open("poem.json", "w", encoding = "utf8")
json.dump(dictionary, fp, ensure_ascii = False)
fp.close()


# #### 最后在 https://observablehq.com/@d3/cluster-dendrogram 上传生成的poem.json文件进行聚类可视化
