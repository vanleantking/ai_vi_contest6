#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 17:41:43 2018

@author: dtvo
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Embs(nn.Module):
    """
    This module take embedding inputs (characters or words) feeding to an RNN layer to extract:
        - all hidden features
        - last hidden features
        - all attentional hidden features
        - last attentional hidden features
    """
    def __init__(self, HPs):
        super(Embs, self).__init__()
        [nnmode, size, dim, pre_embs, hidden_dim, dropout, layers, bidirect, zero_padding, attention] = HPs
        self.zero_padding = zero_padding
        rnn_dim = hidden_dim // 2 if bidirect else hidden_dim
            
        self.embeddings = nn.Embedding(size, dim, padding_idx=0)
        if pre_embs is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(pre_embs))
        else:
            self.embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(size, dim)))

        self.drop = nn.Dropout(dropout)

        if nnmode == "rnn":
            self.hidden_layer = nn.RNN(dim, rnn_dim, num_layers=layers, batch_first=True, bidirectional=bidirect)
        elif nnmode == "gru":
            self.hidden_layer = nn.GRU(dim, rnn_dim, num_layers=layers, batch_first=True, bidirectional=bidirect)
        else: 
            self.hidden_layer = nn.LSTM(dim, rnn_dim, num_layers=layers, batch_first=True, bidirectional=bidirect)
        
        self.attention = attention
        if attention:
            self.att_hidden = nn.Linear(hidden_dim, hidden_dim)
            self.att_alpha = nn.Linear(hidden_dim,1, bias=False)
            self.att_norm = nn.Softmax(-1)
            
    def forward(self, inputs, input_lengths):
        return self.get_last_hiddens(inputs, input_lengths)

    def get_last_hiddens(self, inputs, input_lengths):
        """
            input:  
                inputs: tensor(batch_size, seq_length)
                input_lengths: tensor(batch_size,  1)
            output: 
                tensor(batch_size, hidden_dim)
        """
        if self.zero_padding:
            # set zero vector for padding, unk, eot, sot
            self.set_zeros([0, 1, 2, 3])
        batch_size = inputs.size(0)
        seq_length = inputs.size(1)
        # embs = tensor(batch_size, seq_length,input_dim)
        embs = self.embeddings(inputs)
        embs_drop = self.drop(embs)
        pack_input = pack_padded_sequence(embs_drop, input_lengths.cpu().data.numpy(), True)
        #pack_cu_input=pack_input
        #pack_cu_input=pack_cu_input.cuda()
        # rnn_out = tensor(batch_size, seq_length, rnn_dim * 2)
        # hc_n = (h_n,c_n); h_n = tensor(2, batch_size, rnn_dim)
        rnn_out, hc_n = self.hidden_layer(pack_input)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        if self.attention:
            a_hidden = F.relu(self.att_hidden(rnn_out))
            # a_alpha = tensor(batch_size, seq_length, 1)
            a_alpha = F.relu(self.att_alpha(a_hidden))
            # a_alpha = tensor(batch_size, seq_length)
            a_alpha.squeeze_()
            # alpha = tensor(batch_size, seq_length)
            alpha = self.att_norm(a_alpha)
            # att_out = tensor(batch_size, seq_length, input_dim)
            att_out = rnn_out*alpha.view(batch_size,seq_length,1)
            # att_h = tensor(batch_size, input_dim)
            att_h = att_out.sum(1)    
            return att_h
        else:
            # concatenate forward and backward h_n; h_n = tensor(batch_size, rnn_dim*2)
            if type(hc_n) == tuple:
                h_n = torch.cat([hc_n[0][0, :, :], hc_n[0][1, :, :]], -1)
            else:
                h_n = torch.cat([hc_n[0, :, :], hc_n[1, :, :]], -1)
            return h_n

    def random_embedding(self, size, dim):
        pre_embs = np.empty([size, dim])
        scale = np.sqrt(3.0 / dim)
        for index in range(size):
            pre_embs[index,:] = np.random.uniform(-scale, scale, [1, dim])
        return pre_embs
        
    def set_zeros(self,idx):
        for i in idx:
            self.embeddings.weight.data[i].fill_(0)

class CNNLayer(nn.Module):
    def __init__(self, filter_size, embed_size = 32, use_batchnorm = False, out_channels = 32,
    dropout = 0.2, number_of_channels = 1, strides=(1, 1), activation=nn.ReLU, max_pool=True):
        super(CNNLayer, self).__init__()
        self.filter_size = filter_size
        self.out_channels = out_channels
        self.use_batchnorm = use_batchnorm
        self.activation = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d(self.out_channels)

        self.conv = [nn.Conv2d(in_channels=1, out_channels=self.out_channels,
        kernel_size=(_filter, embed_size), stride=(1, 1),
        padding=(_filter//2, 0), dilation=1, bias=False) for _, _filter in enumerate(filter_size)]
        self.drop = nn.Dropout(dropout)
        
    
    def forward(self, x):
        pool_out = []
        if self.use_batchnorm:
            for conv in self.conv:
                pool = F.tanh(conv(x))
                # pool = F.relu(conv(x)).squeeze(3)
                pool = torch.mean(pool, -1)
                pool_out.append(pool)
        else:
            for conv in self.conv:
                pool = F.relu(conv(x))
                pool = self.batch_norm(pool)                    
                # pool = F.relu(conv(x)).squeeze(3)
                pool = torch.mean(pool, -1)
                pool_out.append(pool)


        max = []
        for i in pool_out:
            pool = F.max_pool1d(i, i.size(2))
            # pool = F.max_pool1d(i, i.size(2)).squeeze(2)
            pool =  torch.mean(pool, -1)
            max.append(pool)
        out_cnn = torch.cat(max, 1)
        out_cnn = self.drop(out_cnn)
        return out_cnn


class BiRNN(nn.Module):
    """
    This module take embedding inputs (characters or words) feeding to an RNN layer before adding a softmax function for classification
    """
    def __init__(self, word_HPs=None, fc_drop = 0.1, filter_size = [2,3,4,5], out_channels = 32, use_batchnorm = False, num_labels=None):
        super(BiRNN, self).__init__()
        [nnmode, word_size, word_dim, wd_embeddings, word_hidden_dim,
         word_dropout, word_layers, word_bidirect, zero_padding, word_att] = word_HPs
        self.zero_padding = zero_padding
        self.num_labels = num_labels
        self.rnn = Embs(word_HPs)
        self.filter_size = filter_size
        self.use_batchnorm = use_batchnorm
        self.dropfinal = nn.Dropout(word_dropout)
        self.dropout_fc = nn.Dropout(fc_drop)
        hidden_dim = word_hidden_dim

        self.cnn = CNNLayer(filter_size = self.filter_size,
            use_batchnorm = self.use_batchnorm)

        self.activation = nn.ReLU()
        fc_in = len(self.filter_size) * out_channels
        self.fc1 = nn.Linear(fc_in, 300, bias=True)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 50)
        self.fc4 = nn.Linear(50, 1)

        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm1d(300)
            self.bn2 = nn.BatchNorm1d(200)
            self.bn3 = nn.BatchNorm1d(50)
        

        if num_labels > 2:
            self.hidden2tag = nn.Linear(hidden_dim, num_labels)
            self.lossF = nn.CrossEntropyLoss()
        else:
            self.hidden2tag = nn.Linear(hidden_dim, 1)
            self.lossF = nn.BCEWithLogitsLoss()            
    
    def forward(self, word_tensor, word_lengths):
        word_h_n = self.rnn(word_tensor, word_lengths)
        label_score = self.dropfinal(word_h_n)
        y = torch.unsqueeze(label_score, 1)
        y = torch.unsqueeze(y, 1) # input for cnn is [32, 1, 1, 32]
        
        out_cnn = self.cnn(y)
        out_cnn = self.linear_batchnormalize(out_cnn)
        print("out cnn, ", out_cnn.size())
        return out_cnn

    # fully connected layer use batch normalize
    def linear_batchnormalize(self, pool_out):
        # not use batch_norm
        if self.use_batchnorm is False:
            # pool_out = self.dropfinal(pool_out)
            out_cnn = self.fc1(pool_out)
            out_cnn = self.activation(out_cnn)
            pool_out = self.dropout_fc(out_cnn)
            out_cnn = self.fc2(out_cnn)
            out_cnn = self.activation(out_cnn)
            out_cnn = self.dropout_fc(out_cnn)
            out_cnn = self.fc3(out_cnn)
            out_cnn = self.activation(out_cnn)
            out_cnn = self.dropout_fc(out_cnn)
            out_cnn = self.fc4(out_cnn)
            return out_cnn
        
        # use batch_norm instead of fropout: linear->activation->batchnorm
        out_cnn = self.fc1(pool_out)        
        out_cnn = self.activation(out_cnn)
        out_cnn = self.bn1(out_cnn)
        out_cnn = self.fc2(out_cnn)
        out_cnn = self.activation(out_cnn)
        out_cnn = self.bn2(out_cnn)
        out_cnn = self.fc3(out_cnn)
        out_cnn = self.activation(out_cnn)
        out_cnn = self.bn3(out_cnn)
        out_cnn = self.fc4(out_cnn)
        return out_cnn

    def NLL_loss(self, label_score, label_tensor):
        if self.num_labels > 2:
            batch_loss = self.lossF(label_score, label_tensor)
        else:
            batch_loss = self.lossF(label_score, label_tensor.float().view(-1,1))
        return batch_loss  

    def inference(self, label_score, k=1):
        if self.num_labels > 2:
            label_prob = F.softmax(label_score, dim=-1)
            label_prob, label_pred = label_prob.data.topk(k)
        else:
            label_prob = torch.sigmoid(label_score.squeeze())
            label_pred = (label_prob >= 0.5).data.long()
        return label_prob, label_pred


if __name__ == "__main__":
    from data_utils import Data2tensor, Vocab, seqPAD, Txtfile
    filename = "../data/train.txt"
    vocab = Vocab(wl_th=None, cutoff=2)
    vocab.build([filename], firstline=False)
    word2idx = vocab.wd2idx(vocab.w2i)
    tag2idx = vocab.tag2idx(vocab.l2i)
    train_data = Txtfile(filename, firstline=False, word2idx=word2idx, tag2idx=tag2idx)

    train_iters = Vocab.minibatches(train_data, batch_size=4)
    data = []
    label_ids = []
    for words, labels in train_iters:
        data.append(words)
        label_ids.append(labels)
        word_ids, sequence_lengths = seqPAD.pad_sequences(words, pad_tok=0, wthres=1024)

    w_tensor = Data2tensor.idx2tensor(word_ids)
    y_tensor = Data2tensor.idx2tensor(labels)

    data_tensors = Data2tensor.sort_tensors(labels, word_ids, sequence_lengths)
    label_tensor, word_tensor, sequence_lengths, word_seq_recover = data_tensors
