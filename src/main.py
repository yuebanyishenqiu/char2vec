# TODO
# word embedding implementation using pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import codecs
from data_helper import load_json, write2json
import time
import random
import os

torch.manual_seed(42)


def load_data(fin):
    
    fp = codecs.open(fin, "r", encoding = "utf8")
    data = fp.readlines()
    fp.close()

    return data


class CBOW(nn.Module):

    def __init__(self, batch_size, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.batch_size = batch_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(2*context_size*embedding_dim, 64)
        self.linear2 = nn.Linear(64, vocab_size)

    def forward(self, inputs):

        embeds = self.embeddings(inputs).view((self.batch_size, self.context_size*self.embedding_dim*2))
        out1 = F.relu(self.linear1(embeds))
        out2 = self.linear2(out1)
        log_probs = F.log_softmax(out2, dim = 1)
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
            weights = i.data.cpu().numpy()
        np.save(fout, weights)


def make_input_data(fin, context_size):
    data = load_data(fin)
    vocab = set()
    ngrams = []
    for line in data:
        line = line.strip().split()
        line = [ ch for ch in line if ch != " "]
        for i in range(context_size, len(line)-context_size):
            left_ctx = line[i-context_size:i]
            right_ctx = line[i:i+context_size]
            ctx = left_ctx+right_ctx
            ngrams.append([ctx, line[i]])
        vocab |= set(line)

    char2idx = {ch: idx for idx, ch in enumerate(vocab)}
    return ngrams, vocab, char2idx


def train(fin, batch_size, context_size, embedding_dim, patient_num, learning_rate, threshold, num_epoch, fmd, fchar2idx):
    ngrams, vocab, char2idx = make_input_data(fin, context_size)

    loss_func = nn.NLLLoss()
    vocab_size = len(vocab)
    model = CBOW(batch_size, vocab_size, embedding_dim, context_size)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    pre_loss = 0
    num = 0
    epoch_idx = 0

    print("INFO: vocab_size: {} ngram_size: {}".format(vocab_size, len(ngrams)))
    while True:
        cur_loss = 0
        random.shuffle(ngrams)
        for i in range(len(ngrams)-batch_size):
            batch_data = ngrams[i:i+batch_size]
            context_idxs = []
            target_idxs = []

            for j in range(batch_size):
                sample_wrds = batch_data[j][0]
                target_wrd = batch_data[j][1]

                x_idxs = [char2idx[w] for w in sample_wrds]
                y_idx = char2idx[target_wrd]
                
                context_idxs.append(x_idxs)
                target_idxs.append(y_idx)
        
            context_idxs = torch.LongTensor(context_idxs).cuda()
            target_idxs = torch.LongTensor(target_idxs).cuda()

            model.zero_grad()

            log_probs = model(context_idxs)
            loss = loss_func(log_probs, target_idxs)

            loss.backward()
            optimizer.step()

            cur_loss+=loss.item()
        if abs(cur_loss-pre_loss) < threshold:
            num += 1
            if num >= patient_num:
                break
        else:
            num = 0
        pre_loss = cur_loss
        
        print("INFO: Epoch: {} Training Loss: {:.3f}".format(epoch_idx, cur_loss))
        epoch_idx += 1
        if epoch_idx >= num_epoch:
            break
    
    write2json(fchar2idx, char2idx)
    model.cpu()
    torch.save(model.state_dict(), fmd)

if __name__ == "__main__":
    batch_size = 64
    context_size = 3
    embedding_dim = 16
    patient_num = 10
    learning_rate = 0.0001
    threshold = 0.01
    num_epoch = 500
    path = "../md"
    if not os.path.exists(path):
        os.makedirs(path)
    fin = "../data/train.txt"
    fmd = "{}/char2vec.md".format(path)
    fchar2idx = "{}/char2idx.json".format(path)
    beg = time.time()
    train(fin, batch_size, context_size, embedding_dim, patient_num, learning_rate, threshold, num_epoch, fmd, fchar2idx)
    end = time.time()
    print("Total Time: {:.3f} min".format((end-beg)/60))
