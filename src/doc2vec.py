# Author: Wenjie Peng 201921198481
import h5py
import codecs
import json
from data_helper import load_json
import pickle
import numpy as np
from sklearn.decomposition import PCA
import sys
import pickle


def load_hdf(fin):

    hdf = h5py.File(fin, "r")
    dst = hdf["data"][:]
    hdf.close()
    return dst

def load_data_all():

    # glyce embedding
    fin = "../data/wvec.hf"
    init_vec = load_hdf(fin)

    # char vec from training
    fin = "../data/trn_wvec.hf"
    train_vec = load_hdf(fin)
    
    # glyce char2vec idx
    fdic = "../data/dictionary.json"
    fdic = load_json(fdic)
    init_ch2idx = fdic["char2idx"]

    # char2vec idx from training
    ch2idx = load_json("../md/char2idx.json")
    return init_ch2idx, init_vec, ch2idx, train_vec


def make_data(cdic, ch2idx, train_vec, init_ch2idx, init_vec):

    pca = PCA(n_components = 1)
    dic = {}
    raw_dic = {}

    for i in range(len(cdic)):

        poem = cdic[i]
        paras = poem["paragraphs"]
        author = poem["author"]
        rhythmic = poem["rhythmic"]
        vec = [] # vec with glyce embedding
        raw_vec = [] # vec only from training
        for line in paras:
            for ch in line:
                idx = ch2idx[ch]
                tvec = train_vec[idx]
                ivec = init_vec[init_ch2idx[ch]]

                v = np.concatenate((tvec, ivec))
                vec.append(v)
                raw_vec.append(tvec)
        if len(vec) < 20:
            continue

        raw_dic, dic = add2dic(raw_dic, dic, raw_vec, vec, pca, i, author, rhythmic, paras)
    return raw_dic, dic

def add2dic(raw_dic, dic, raw_vec, vec, pca, i, author, rhythmic, paras):

    raw_dic = add(raw_dic, i, raw_vec, pca, author, rhythmic, paras)
    dic = add(dic, i, vec, pca, author, rhythmic, paras)

    return raw_dic, dic

def add(dic, i, vec, pca, author, rhythmic, paras):

    vec = np.array(vec)
    vec = vec.T
    new_vec = pca.fit_transform(vec)
    new_vec = new_vec.reshape((new_vec.shape[0]))
    dic[i] = {}
    dic[i]["author"] = author
    dic[i]["rhythmic"] = rhythmic
    dic[i]["paragraphs"] = paras
    dic[i]["doc2vec"] = new_vec

    return dic

def write2pickle(fout, dic):
    with codecs.open(fout, "wb") as fp:
        pickle.dump(dic, fp, protocol = pickle.HIGHEST_PROTOCOL)

def doc2vec(fci):
    dic = load_json(fci)
    init_ch2idx, init_vec, ch2idx, train_vec = load_data_all()

    raw_dic, dic = make_data(dic, ch2idx, train_vec, init_ch2idx, init_vec)

    fout = "../data/char2vec_with_glyce.pickle"
    write2pickle(fout, dic)

    fout = "../data/char2vec_raw.pickle"
    write2pickle(fout, raw_dic)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: {} fci".format(sys.argv[0]))
        exit(0)

    doc2vec(sys.argv[1])
