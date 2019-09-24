#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 03:38:32 2018

@author: duytinvo
"""
import os
import sys
import time
import random
import argparse
import numpy as np

import torch
from sklearn import metrics
import torch.optim as optim

from utils.core_nns import BiRNN as fNN
from utils.other_utils import Progbar, Timer, SaveloadHP
from utils.data_utils import Vocab, Data2tensor, Txtfile, seqPAD, Embeddings
seed_num = 12345
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


class Classifier(object):
    def __init__(self, args=None):
        
        self.args = args  
        self.device = torch.device("cuda:0" if self.args.use_cuda else "cpu")
        # word_layers = 1
        word_bidirect = True        
        word_HPs = [self.args.word_nnmode, len(self.args.vocab.w2i), self.args.word_dim,
                    self.args.word_pred_embs, self.args.word_hidden_dim, self.args.dropout,
                    self.args.word_layers, word_bidirect, self.args.zero_padding, self.args.word_att]
        
        self.model = fNN(word_HPs=word_HPs, use_batchnorm = self.args.use_batchnorm, num_labels=len(self.args.vocab.l2i)).to(self.device)

        if args.optimizer.lower() == "adamax":
            self.optimizer = optim.Adamax(self.model.parameters(), lr=self.args.lr)
        elif args.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif args.optimizer.lower() == "adadelta":
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.args.lr)
        elif args.optimizer.lower() == "adagrad":
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.args.lr)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9)
        
        self.word2idx = self.args.vocab.wd2idx(vocab_words=self.args.vocab.w2i, allow_unk=True, start_end=self.args.start_end)
        self.tag2idx = self.args.vocab.tag2idx(vocab_tags=self.args.vocab.l2i)

    def evaluate_batch(self, eva_data):
        with torch.no_grad():
            wl = self.args.vocab.wl
            batch_size = self.args.batch_size  
             ## set model in eval model
            self.model.eval()
            start = time.time()
            y_true = Data2tensor.idx2tensor([], self.device)
            y_pred = Data2tensor.idx2tensor([], self.device)
            for i,(words, label_ids) in enumerate(self.args.vocab.minibatches(eva_data, batch_size=batch_size)):
                word_ids, sequence_lengths = seqPAD.pad_sequences(words, pad_tok=0, wthres=wl)
        
                data_tensors = Data2tensor.sort_tensors(label_ids, word_ids,sequence_lengths, self.device)
                label_tensor, word_tensor, sequence_lengths, word_seq_recover = data_tensors

                y_true = torch.cat([y_true,label_tensor])
                label_score = self.model(word_tensor, sequence_lengths)
                label_prob, label_pred = self.model.inference(label_score, k=1)
                
                y_pred = torch.cat([y_pred, label_pred])
            #measures = Classifier.class_metrics(y_true, y_pred.squeeze())
            measures = Classifier.class_metrics(y_true.data.cpu().numpy(), y_pred.squeeze().data.cpu().numpy())

            end = time.time() - start
            speed = len(y_true)/end
        return measures, speed

    def train_batch(self,train_data):
        wl = self.args.vocab.wl
        clip_rate = self.args.clip
        
        batch_size = self.args.batch_size
        num_train = len(train_data)
        total_batch = num_train//batch_size+1
        prog = Progbar(target=total_batch)
        ## set model in train model
        self.model.train()
        train_loss = []
        for i,(words, label_ids) in enumerate(self.args.vocab.minibatches(train_data, batch_size=batch_size)):
            word_ids, sequence_lengths = seqPAD.pad_sequences(words, pad_tok=0, wthres=wl)

            data_tensors = Data2tensor.sort_tensors(label_ids, word_ids,sequence_lengths,self.device)
            label_tensor, word_tensor, sequence_lengths, word_seq_recover = data_tensors

            self.model.zero_grad()
            label_score = self.model(word_tensor, sequence_lengths)
            batch_loss = self.model.NLL_loss(label_score, label_tensor)
            train_loss.append(batch_loss.item())
            
            batch_loss.backward()
            
            if clip_rate>0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_rate)
                
            self.optimizer.step()
            
            prog.update(i + 1, [("Train loss", batch_loss.item())])
        return np.mean(train_loss)

    def lr_decay(self, epoch):
        lr = self.args.lr/(1+self.args.decay_rate*epoch)
        print("INFO: - Learning rate is setted as: %f"%lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):            
        train_data = Txtfile(self.args.train_file, firstline=False, word2idx=self.word2idx, tag2idx=self.tag2idx)
        dev_data = Txtfile(self.args.dev_file, firstline=False, word2idx=self.word2idx, tag2idx=self.tag2idx)
        test_data = Txtfile(self.args.test_file, firstline=False, word2idx=self.word2idx, tag2idx=self.tag2idx)

        max_epochs = self.args.max_epochs
        saved_epoch = 0
        best_dev = -1
        best_metrics = {}

        nepoch_no_imprv = 0
        epoch_start = time.time()
        for epoch in range(max_epochs):
            if self.args.decay_rate>0: 
                self.lr_decay(epoch)
            print("Epoch: %s/%s" %(epoch,max_epochs))
            train_loss = self.train_batch(train_data)
            # evaluate on developing data
            dev_metrics, dev_speed = self.evaluate_batch(dev_data)
            dev_metric_standard = dev_metrics["prf_macro"][2]
            if dev_metric_standard > best_dev:
                nepoch_no_imprv = 0
                saved_epoch = epoch
                best_dev = dev_metric_standard
                best_metrics = dev_metrics
                print("UPDATES: - New improvement")  
                print("         - Train loss: %.4f"%train_loss)
                print("         - Dev acc: %.2f(%%); Dev P: %.2f(%%); Dev R: %.2f(%%);Dev F1: %.2f(%%); Dev speed: %.2f(sent/s)"%(100*dev_metrics["acc"],
                      100*dev_metrics["prf_macro"][0], 100*dev_metrics["prf_macro"][1], 100*dev_metrics["prf_macro"][2], dev_speed))
                print("         - Save the model to %s at epoch %d"%(self.args.model_name,saved_epoch))
                # Conver model to CPU to avoid out of GPU memory
                self.model.to("cpu")
                torch.save(self.model.state_dict(), self.args.model_name)
                self.model.to(self.device)
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.args.patience:
                    self.model.load_state_dict(torch.load(self.args.model_name))
                    self.model.to(self.device)
                    test_metrics, test_speed = self.evaluate_batch(test_data)
                    print("\nSUMMARY: - Early stopping after %d epochs without improvements"%(nepoch_no_imprv))
                    print("         - Dev acc: %.2f(%%); Dev P: %.2f(%%); Dev R: %.2f(%%);Dev F1: %.2f(%%)"%(100*best_metrics["acc"],
                          100*best_metrics["prf_macro"][0], 100*best_metrics["prf_macro"][1], 100*best_metrics["prf_macro"][2]))
                    print("         - Load the best model from: %s at epoch %d"%(self.args.model_name,saved_epoch))                    
                    print("         - Test acc: %.2f(%%); Test P: %.2f(%%); Test R: %.2f(%%);Test F1: %.2f(%%); "
                          "Test speed: %.2f(sent/s)"%(100*test_metrics["acc"], 100*test_metrics["prf_macro"][0],
                                                      100*test_metrics["prf_macro"][1],
                                                      100*test_metrics["prf_macro"][2], test_speed))
                
                    return

            epoch_finish = Timer.timeEst(epoch_start,(epoch+1)/max_epochs)
            print("\nINFO: - Trained time(Remained time for %d epochs: %s"%(max_epochs, epoch_finish))
        
        self.model.load_state_dict(torch.load(self.args.model_name))
        self.model.to(self.device)
        test_metrics, test_speed = self.evaluate_batch(test_data)
        print("\nSUMMARY: - Completed %d epoches"%(max_epochs))
        print("         - Dev acc: %.2f(%%); Dev P: %.2f(%%); Dev R: %.2f(%%);Dev F1: %.2f(%%)"%(100*best_metrics["acc"],
              100*best_metrics["prf_macro"][0], 100*best_metrics["prf_macro"][1], 100*best_metrics["prf_macro"][2]))
        print("         - Load the best model from: %s at epoch %d"%(self.args.model_name,saved_epoch))
        print("         - Test acc: %.2f(%%); Test P: %.2f (%%); Test R: %.2f(%%);Test F1: %.2f(%%); Test speed: %.2f(sent/s)"%(100*test_metrics["acc"],
              100*test_metrics["prf_macro"][0], 100*test_metrics["prf_macro"][1], 100*test_metrics["prf_macro"][2], test_speed))
        return 

    def predict(self, sent, k=1):
        """

        :param sent: processed sentence
        :param asp: an aspect mentioned inside sent
        :param k: int
        :return: top k predictions
        """
        wl = self.args.vocab.wl
         ## set model in eval model
        self.model.eval()
        
        fake_label = [0]        
        words = self.word2idx(sent)
        word_ids, sequence_lengths = seqPAD.pad_sequences([words], pad_tok=0, wthres=wl)
    
        data_tensors = Data2tensor.sort_tensors(fake_label, word_ids, sequence_lengths, self.device)
        fake_label_tensor, word_tensor, sequence_lengths, word_seq_recover = data_tensors

        label_score = self.model(word_tensor, sequence_lengths)
        label_prob, label_pred = self.model.inference(label_score, k)
        return label_prob, label_pred 
    
    @staticmethod
    def class_metrics(y_true, y_pred):
        acc = metrics.accuracy_score(y_true, y_pred)  
        f1_ma = metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')    
        f1_we = metrics.precision_recall_fscore_support(y_true, y_pred, average='weighted') 
        f1_no = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)  
        measures = {"acc":acc, "prf_macro":f1_ma, "prf_weighted":f1_we, "prf_individual":f1_no}
        return measures


def build_data(args):    
    print("Building dataset...")
    model_dir, _ = os.path.split(args.model_args)
    if not os.path.exists(model_dir): 
        os.mkdir(model_dir)

    vocab = Vocab(wl_th=args.word_thres, cutoff=args.cutoff)
    vocab.build([args.train_file, args.dev_file, args.test_file], firstline=False)
    args.vocab = vocab
    if args.emb_file != "":
        args.word_pred_embs = Embeddings.get_W(args.emb_file,wsize=args.word_dim,vocabx=vocab.w2i)
    else:
        args.word_pred_embs = None
    SaveloadHP.save(args, args.model_args)
    return args


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument('--train_file', help='Trained file', default="./data/train.txt", type=str)

    argparser.add_argument('--dev_file', help='Developed file', default="./data/val.txt", type=str)

    argparser.add_argument('--test_file', help='Tested file', default="./data/test.txt", type=str)
	
    argparser.add_argument('--emb_file', help='Embedding file', default="", type=str)
                        
    argparser.add_argument("--cutoff", type=int, default=2, help="prune words occurring <= cutoff")
    
    argparser.add_argument("--word_thres", type=int, default=None, help="word length threshold")
                
    argparser.add_argument("--word_att", action='store_true', default=False, help="word attentional mechanism flag")
    
    argparser.add_argument("--word_nnmode", type=str, default="lstm", help="word-level neural network")

    argparser.add_argument("--word_layers", type=int, default=1, help="number of layers")

    argparser.add_argument("--start_end", action='store_true', default=False, help="add start-end paddings")
    
    argparser.add_argument("--zero_padding", action='store_true', default=False, help="set all padding tokens to zero "
                                                                                      "during training")

    argparser.add_argument("--word_dim", type=int, default=16, help="word_embedding vector size")

    argparser.add_argument("--word_hidden_dim", type=int, default=32, help="LSTM word_hidden layers")
	
    argparser.add_argument("--dropout", type=float, default=0.4, help="dropout rate")
    
    argparser.add_argument("--patience", type=int, default=8, help="early stopping")
            
    argparser.add_argument("--optimizer", type=str, default="ADAM", help="learning method (adagrad, sgd, ...)")
    
    argparser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    
    argparser.add_argument("--decay_rate", type=float, default=0.05, help="decay learning rate")
        
    argparser.add_argument("--max_epochs", type=int, default=32, help="maximum # of epochs")
    
    argparser.add_argument("--batch_size", type=int, default=32, help="batch size")
    
    argparser.add_argument('--clip', type=int, default=5, help='Clipping value')
        
    argparser.add_argument('--model_name', type=str, default="./data/classifier.m", help='Model name')
    
    argparser.add_argument('--model_args', type=str, default="./data/classifier.args", help='Model arguments')
    
    argparser.add_argument("--use_cuda", action='store_true', default=False, help="GPUs Flag (default True)")

    argparser.add_argument("--use_batchnorm", action='store_true', default=True, help="Enable batchnormalize in network")
    
    args = argparser.parse_args()
    
    args = build_data(args)
    
    classifier = Classifier(args)

    classifier.train()
