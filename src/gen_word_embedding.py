# Author: Wenjie Peng 201921198481
import torch
import torch.nn as nn
import torch.nn.functional as F

from main import CBOW
import json
import codecs
import numpy as np

fin = "../md/char2idx.json"
fp = codecs.open(fin, "r", encoding = "utf8")
dic = json.load(fp)
fp.close()

fmd = "../md/char2vec.md"

batch_size = 64
vocab_size = len(dic)
embedding_dim = 16
context_size = 3

model = CBOW(batch_size, vocab_size, embedding_dim, context_size)

model.load_state_dict(torch.load(fmd))

cdic = {}

embed_layer = model.embeddings
vec = np.zeros((vocab_size, embedding_dim))
for ch in dic:
    idx = dic[ch]
    idx = torch.LongTensor([idx])
    tensor = embed_layer(idx).data.numpy()
    vec[idx] = tensor

import h5py

# char2vec from training
fout = "../data/trn_wvec.hf"
hdf = h5py.File(fout, "w")
dst = hdf.create_dataset("data", data = vec, dtype = "float")
hdf.close()
