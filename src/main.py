# TODO
# word embedding implementation using pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import urllib.request
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import codecs
from data_helper import load_json
import h5py


torch.manual_seed(42)

CONTEXT_SIZE = 3
EMBEDDING_DIM = 13

def load_data(fin):
    fp = codecs.open(fin, "r", encoding = "utf8")
    data = fp.readlines()
    fp.close()
    X = ""
    for line in data:
        line = line.strip()
        X = "{} {}".format(X, line)
    return X


def get_key(word_id):
    for key, val in char2idx.items():
        if(val == word_id):
            print(key)

def cluster_embeddings(fin, nclusters):
    X = np.load(fin)
    kmeans = KMeans(n_clusters = nclusters, random_state = 0).fit(X)
    center = kmeans.cluster_centers_
    distances = euclidean_distances(X, center)

    for i in np.arange(0, distances.shape[1]):
        word_id = np.argmin(distances[:,i])
        print(word_id)
        get_key(word_id)

def read_data(fpath):

    tokenizer = RegexpTokenizer(r'\w+')
    data = urllib.request.urlopen(fpath)
    data = data.read().decode('utf8')
    tokenize_data = word_tokenize(data)
    stop_words = set(stopwords.words('english'))
    cleaned_words = [i for i in tokenize_data if i not in stop_words]
    return(cleaned_words)


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 64)
        self.linear2 = nn.Linear(64, vocab_size)

    def forward(self, inputs):

        embeds = self.embeddings(inputs).view((1,-1))
        out1 = F.relu(self.linear1(embeds))
        out2 = self.linear2(out1)

        log_probs = F.log_softmax(out2, dim=1)
        return log_probs


    def predict(self, inputs):
        res = self.forward(inputs)
        res_arg = torch.argmax(res)
        res_val, res_ind = res.sort(descending=True)
        res_val = res_val[0][:3]
        res_ind = res_ind[0][:3]

        for arg in zip(res_val, res_ind):
            print([(key, val, arg[0]) for key, val in char2idx.items() if val == arg[1]])


    def freeze_layer(self, layer):
        for name, child in model.named_children():
            print(name, child)
            if(name == layer):
                for names, params in child.named_parameters():
                    print(names, params)
                    print(params.size())
                    params.requires_grad = False

    def print_layer_parameters(self):
        for name, child in model.named_children:
            print(name, child)
            for name, params in child.named_parameters():
                print(names, params)
                print(params.size())

    def write_embedding_to_file(self, fout):
        for i in self.embeddings.parameters():
            weights = i.data.numpy()
        np.save(fout, weights)

test_sentence = load_data("../data/train.txt").split()
test_sentence = [e for e in test_sentence if e != " "]
ngrams = []
for i in range(len(test_sentence)-CONTEXT_SIZE):
    tup = [test_sentence[j] for j in np.arange(i, i+CONTEXT_SIZE)]
    ngrams.append((tup, test_sentence[i+CONTEXT_SIZE]))

dic = load_json("../data/dictionary.json")
vocab = set(test_sentence)
vocab_size = len(vocab)
print("Length of vocabulary: {}".format(len(vocab)))
char2idx = {char: i for i, char in enumerate(vocab)}

#import pickle

#fin = "../data/wvec.hf"
#hdf = h5py.File(fin, "r")
#wvec = hdf["data"][:]
#hdf.close()
#wvec = torch.from_numpy(wvec)
#print("wvec: ", wvec.size)

losses = []
loss_func = nn.NLLLoss()
model = CBOW(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr = 0.001)

for epoch in range(400):
    total_loss = 0

    for context, target in ngrams:

        context_idxs = torch.LongTensor([char2idx[w] for w in context])
        target = torch.LongTensor([char2idx[target]])
        model.zero_grad()

        log_probs = model(context_idxs)
        loss = loss_func(log_probs, target)

        loss.backward()
        optimizer.step()

        total_loss+=loss.item()

    print(total_loss)
    losses.append(total_loss)


#model.predict(['of', 'all', 'human'])
model.write_embedding_to_file('embeddings.npy')
#cluster_embeddings('embeddings.npy', 2)
