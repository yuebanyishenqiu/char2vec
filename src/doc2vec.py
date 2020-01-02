# Author: Wenjie Peng 201921198481
import h5py
import codecs
import json
from data_helper import load_json
import pickle
import numpy as np
from sklearn.decomposition import PCA
import pickle


fci = "../../../data/ci/ci.song.10000.json"
dic = load_json(fci)


def load_hdf(fin):

    hdf = h5py.File(fin, "r")
    dst = hdf["data"][:]
    hdf.close()
    return dst

fin = "../data/wvec.hf"
init_vec = load_hdf(fin)

fin = "./wvec.hf"
train_vec = load_hdf(fin)

fdic = "../data/dictionary.json"
fdic = load_json(fdic)
init_ch2idx = fdic["char2idx"]

ch2idx = load_json("../md/v1/char2idx.json")


pca = PCA(n_components = 1)

res_dic = {}

co = 0
for i in range(len(dic)):

    poem = dic[i]
    paras = poem["paragraphs"]
    author = poem["author"]
    rhythmic = poem["rhythmic"]
    vec = []
    for line in paras:
        for ch in line:
            idx = ch2idx[ch]
            tvec = train_vec[idx]

            ivec = init_vec[init_ch2idx[ch]]

            #v = np.concatenate((tvec, ivec))
            vec.append(ivec)
    if len(vec) < 20:
        continue
    vec = np.array(vec)
    vec = vec.T
    print(vec.shape)
    new_vec = pca.fit_transform(vec)
    print(new_vec.shape)
    new_vec = new_vec.reshape((new_vec.shape[0]))
    print(new_vec.shape)
    co+=1
    res_dic[i] = {}
    res_dic[i]["author"] = author
    res_dic[i]["rhythmic"] = rhythmic
    res_dic[i]["paragraphs"] = paras
    res_dic[i]["doc2vec"] = new_vec
print(co)
fout = "../data/res_dic_raw.pickle"
with codecs.open(fout, "wb") as fp:
    pickle.dump(res_dic, fp, protocol = pickle.HIGHEST_PROTOCOL)
