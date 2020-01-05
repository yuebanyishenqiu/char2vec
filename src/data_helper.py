# TODO
# prepare data for model training
# Author: Wenjie Peng 201921198481
from zhconv import convert
from PIL import ImageFont
import numpy as np
import json
from sklearn import preprocessing
import os
import sys
import codecs
import pickle
import h5py



def load_json(fin):
    fp =  codecs.open(fin, "r", encoding = "utf8") 
    dic = json.load(fp)
    fp.close()

    return dic

def write2json(fout, dic):
    fp = codecs.open(fout, "w", encoding = "utf8")
    json.dump(dic, fp, ensure_ascii = False)
    fp.close()

def add2vocab(data):
    
    vocab = set()
    text = ""
    for poem in data:
        paragraphs = poem["paragraphs"]
        text = "{} <p>".format(text)
        for line in paragraphs:
            line = line.replace(" ", "")
            for ch in line:
                vocab.add(ch)
                text = "{} {}".format(text, ch)
        text = "{} </p>\n".format(text)
    return vocab, text

def make_dictionary(vocab, fout):

    dic = {"char2idx": {}, "idx2char": {}}
    vocab = list(vocab)
    for i in range(len(vocab)):
        ch = vocab[i]
        dic["char2idx"][ch] = i
        dic["idx2char"][i] = ch

    write2json(fout, dic)

def tokenize_poem(src):
    # vocab includs three parts:
    # (1) ordinary chinese word
    # (2) ， 。 ？ markers
    X = ""
    vocab = set()
    for f in os.listdir(src):
        if f.startswith("ci.song"):
            fin = "{}/{}".format(src, f)
            data = load_json(fin)
            v, t = add2vocab(data)
            X = "{} {}".format(X, t)
            vocab |= v
    
    fout = "../data/dictionary.json"
    make_dictionary(vocab, fout)
    fout = "../data/train.txt"
    fp = codecs.open(fout, "w", encoding = "utf8")
    fp.write(X)
    fp.close()

def vocab_embedding(idx2char, font, font_size = 24, use_traditional = False, normalize = False):
    r = np.array([make_char_embedding(i, font, use_traditional, idx2char) for i in range(len(idx2char))])
    return (r - np.mean(r))/np.std(r) if use_traditional else r

def pad_mask(mask, font_size):

    padded_mask = []
    for l in mask:
        padded_mask.append(l.tolist()+[0]*(font_size+1-len(l)))
    for i in range(font_size+1-len(padded_mask)):
        padded_mask.append([0]*(font_size+1))
    return np.array(padded_mask)[:font_size+1, : font_size+1]


def embed_char(char, font):
    mask = font.getmask(char)
    size = mask.size[::-1]
    a = np.asarray(mask).reshape(size)
    return a


def make_char_embedding(idx, font, use_traditional, idx2char):
    idx = str(idx)
    char = idx2char[idx]
    feat = ""
    if len(char) > 1:
        feat =  np.zeros((font.size+1, font.size+1))
    else:
        feat =  pad_mask(embed_char(char, font), font.size)
    
    vec = []
    # make the feature distinct using binary numeric system
    base = np.arange(-6,7,1)
    base = np.exp2(base)
    for v in feat:
        val = 0
        for i in range(len(v)):
            val += base[i]*v[i]
        vec.append(val)
    return np.array(vec)

def normalize_data(dic, nomalizer):
    new_dic = {}
    for k in dic:
        feat = [dic[k]]
        new_feat = nomalizer.transform(feat)
        new_dic[k] = new_feat
    return new_dic

def init_char_embedding(f, fttf):

    # generate char embedding from glyce
    # store as ../data/wvec.hf
    
    dic = load_json(f)
    idx2char = dic["idx2char"]
    char2idx = dic["char2idx"]
    
    font = ImageFont.truetype(fttf, 12)
    emb  = vocab_embedding(idx2char, font, 24)

    #fout = "../data/init_char_embedding.pickle"
    fdic = {}
    X = np.zeros((len(idx2char), 13))
    for idx in range(len(idx2char)):
        feat = make_char_embedding(idx, font, False, idx2char)
        X[idx,:] = feat
    X = preprocessing.Normalizer().fit_transform(X)
    fout = "../data/wvec.hf"
    hdf = h5py.File(fout, "w")
    dst = hdf.create_dataset("data", data = X, dtype = "float")
    hdf.close()
#    return 
#    char = idx2char[str(idx)]
#    char_idx = char2idx[char]
#    fdic[char_idx] = feat
#    from sklearn import preprocessing
#    nomalizer = preprocessing.Normalizer().fit(X)
#    fdic = normalize_data(fdic, nomalizer)
#    fp = codecs.open(fout, "wb")
#    pickle.dump(fdic, fp, protocol = pickle.HIGHEST_PROTOCOL)
#    fp.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: {} ci_dir ttf".format(sys.argv[0]))
        exit(0)

    tokenize_poem(sys.argv[1])
    init_char_embedding("../data/dictionary.json", sys.argv[2])
