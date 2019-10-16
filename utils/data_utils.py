#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 17:31:21 2018

@author: dtvo
"""
import re
import torch
import itertools
import numpy as np
from collections import Counter
# ----------------------
#    Word symbols
# ----------------------
PADw = u"<PADw>"
UNKw = u"<UNKw>"
SOw = u"<sw>"
EOw = u"</sw>"


class Vocab(object):
    def __init__(self, wl_th=None, cutoff=1):
        self.w2i = {}
        self.l2i = {}
        self.wl = wl_th
        self.cutoff = cutoff
                        
    def build(self, files, firstline=False, limit=-1):
        """
        Read a list of file names, return vocabulary
        :param files: list of file names
        :param firstline: ignore first line flag
        :param limit: read number of lines
        """
        lcnt = Counter()
        wcnt = Counter()
        print("Extracting vocabulary:")
        wl = 0
        count = 0
        for fname in files:
            raw = Txtfile(fname, firstline=firstline, limit=limit)
            for sent, label in raw:
                sent = sent.split()
                wcnt.update(sent)
                wl = max(wl, len(sent))
                if label != "":
                    lcnt.update([label])
                count += 1
        print("\t%d total samples, %d total tokens, %d total labels" % (count, sum(wcnt.values()), sum(lcnt.values())))
        wlst = [x for x, y in wcnt.items() if y >= self.cutoff]
        wlst = [PADw, UNKw, SOw, EOw] + wlst
        wvocab = dict([(y, x) for x, y in enumerate(wlst)])
        lvocab = dict([(y, x) for x, y in enumerate(lcnt.keys())])
        print("\t%d unique tokens, %d unique labels" % (len(wcnt), len(lcnt)))
        print("\t%d unique tokens appearing at least %d times" % (len(wvocab)-4, self.cutoff))
        self.w2i = wvocab
        self.l2i = lvocab 
        if self.wl is None:
            self.wl = wl
        else:
            self.wl = min(wl, self.wl)

    def wd2idx(self, vocab_words=None, allow_unk=True, start_end=False):
        """
        Return a function to convert tag2idx or word/word2idx
        :param vocab_words:
        :param allow_unk:
        :param start_end:
        """
        def f(sent):
            if vocab_words is not None:
                word_ids = []
                sent = sent.split()
                for word in sent:
                    # ignore words out of vocabulary
                    if word in vocab_words:
                        word_ids += [vocab_words[word]]
                    else:
                        if allow_unk:
                            word_ids += [vocab_words[UNKw]]
                        else:
                            raise Exception("Unknown key is not allowed. Check that your vocab (tags?) is correct")
                if start_end:
                    # SOc,EOc words for  EOW
                    word_ids = [vocab_words[SOw]] + word_ids + [vocab_words[EOw]]
            return word_ids
        return f
    
    @staticmethod
    def tag2idx(vocab_tags=None):
        def f(tags): 
            if tags in vocab_tags:
                tag_ids = vocab_tags[tags]
            else:
                raise Exception("Unknown key is not allowed. Check that your vocab (tags?) is correct")
            return tag_ids
        return f
        
    @staticmethod
    def minibatches(data, batch_size):
        """
        :param data:
        :param batch_size:
        :return:
        """
        """
        Args:
            data: generator of (sentence, tags) tuples
            minibatch_size: (int)
    
        Yields:
            list of tuples
    
        """
        x_batch, y_batch = [], []
        for (x, y) in data:
            if len(x_batch) == batch_size:
                yield x_batch, y_batch
                x_batch, y_batch = [], []
            x_batch += [x]
            y_batch += [y]
    
        if len(x_batch) != 0:
            yield x_batch, y_batch


class Txtfile(object):
    """
    Read cvs file
    """
    def __init__(self, fname, word2idx=None, tag2idx=None, firstline=True, limit=-1):
        self.fname = fname
        self.firstline = firstline
        if limit < 0:
            self.limit = None
        else:
            self.limit = limit
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.length = None
        
    def __iter__(self):
        with open(self.fname, newline='', encoding='utf-8') as f:
            f.seek(0)
            if self.firstline:
                # Skip the header
                next(f)
            for line in itertools.islice(f, self.limit):
                line = line.strip().split(",")
                if len(line[0].split()) != 0:
                    sent = line[0]
                else:
                    sent = "tốt"
                tag = line[1]
                if self.word2idx is not None:
                    sent = self.word2idx(sent)
                if self.tag2idx is not None:
                    tag = self.tag2idx(tag)
                # yield a tuple (words, tag)
                yield sent, tag
                
    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length

    @staticmethod
    def process_sent(sent):
        sent = re.sub('[^0-9a-zA-Z ]+', '', sent)
        return sent.lower()


class seqPAD:
    @staticmethod
    def _pad_sequences(sequences, pad_tok, max_length):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the word to pad with
    
        Returns:
            a list of list where each sublist has same length
        """
        sequence_padded, sequence_length = [], []
    
        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
            sequence_padded += [seq_]
            sequence_length += [min(len(seq), max_length)]
    
        return sequence_padded, sequence_length

    @staticmethod
    def pad_sequences(sequences, pad_tok, nlevels=1, wthres=1024, cthres=32):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the word to pad with
            nlevels: "depth" of padding, for the case where we have word ids
    
        Returns:
            a list of list where each sublist has same length
    
        """
        if nlevels == 1:
            max_length = max(map(lambda x: len(x), sequences))
            max_length = min(wthres, max_length)
            sequence_padded, sequence_length = seqPAD._pad_sequences(sequences, pad_tok, max_length)
    
        elif nlevels == 2:
            max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
            max_length_word = min(cthres, max_length_word)
            sequence_padded, sequence_length = [], []
            for seq in sequences:
                # pad the word-level first to make the word length being the same
                sp, sl = seqPAD._pad_sequences(seq, pad_tok, max_length_word)
                sequence_padded += [sp]
                sequence_length += [sl]
            # pad the word-level to make the sequence length being the same
            max_length_sentence = max(map(lambda x: len(x), sequences))
            max_length_sentence = min(wthres, max_length_sentence)
            sequence_padded, _ = seqPAD._pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
            # set sequence length to 1 by inserting padding 
            sequence_length, _ = seqPAD._pad_sequences(sequence_length, 1, max_length_sentence)
    
        return sequence_padded, sequence_length


class Embeddings:
    @staticmethod
    def load_embs(fname):
        embs = dict()
        s = 0
        V = 0
        with open(fname, 'r', encoding="utf-8") as f:
            for line in f: 
                p = line.strip().split()
                if len(p) == 2:
                    V = int(p[0])     # Vocabulary
                    s = int(p[1])     # embeddings size
                else:
                    # assert len(p)== s+1
                    w = "".join(p[0])
                    # print(p)
                    e = [float(i) for i in p[1:]]
                    embs[w] = np.array(e, dtype="float32")
#        assert len(embs) == V
        return embs 
    
    @staticmethod
    def get_W(emb_file, wsize, vocabx, scale=0.25):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        print("Extracting pretrained embeddings:")
        word_vecs = Embeddings.load_embs(emb_file)
        print('\t%d pre-trained word embeddings'% (len(word_vecs)))
        print('Mapping to vocabulary:')
        unk = 0
        part = 0
        W = np.zeros(shape=(len(vocabx), wsize), dtype="float32")
        for word, idx in vocabx.items():
            if idx == 0:
                continue
            if word_vecs.get(word) is not None:
                W[idx] = word_vecs.get(word)
            else:
                if word_vecs.get(word.lower()) is not None:
                    W[idx] = word_vecs.get(word.lower())
                    part += 1
                else:
                    unk += 1
                    rvector = np.asarray(np.random.uniform(-scale, scale, (1, wsize)), dtype="float32")
                    W[idx] = rvector
        print('\t%d randomly word vectors;' % unk)
        print('\t%d partially word vectors;'% part)
        print('\t%d pre-trained embeddings.'% (len(vocabx)-unk-part))
        return W

    @staticmethod
    def init_W(wsize, vocabx, scale=0.25):
        """
        Randomly initial word vectors between [-scale, scale]
        """
        W = np.zeros(shape=(len(vocabx), wsize), dtype="float32")
        for word, idx in vocabx.iteritems():
            if idx == 0:
                continue
            rvector = np.asarray(np.random.uniform(-scale, scale, (1, wsize)), dtype="float32")
            W[idx] = rvector
        return W


class Data2tensor:
    @staticmethod
    def idx2tensor(indexes, device=torch.device("cpu")):
        vec = torch.tensor(indexes, dtype=torch.long, device=device)
        return vec

    @staticmethod
    def sort_tensors(label_ids, word_ids, sequence_lengths, device=torch.device("cpu")):
        label_tensor = Data2tensor.idx2tensor(label_ids, device)
        word_tensor = Data2tensor.idx2tensor(word_ids, device)
        sequence_lengths = Data2tensor.idx2tensor(sequence_lengths, device)
        sequence_lengths, word_perm_idx = sequence_lengths.sort(0, descending=True)
        word_tensor = word_tensor[word_perm_idx]
        label_tensor = label_tensor[word_perm_idx]
        _, word_seq_recover = word_perm_idx.sort(0, descending=False)
        return label_tensor, word_tensor, sequence_lengths, word_seq_recover


if __name__ == "__main__":
    filename = "../data/test.csv"
    vocab = Vocab(wl_th=None, cutoff=2)
    vocab.build([filename], firstline=False)
    word2idx = vocab.wd2idx(vocab.w2i)
    tag2idx = vocab.tag2idx(vocab.l2i)
    train_data = Txtfile(filename, firstline=False, word2idx=word2idx, tag2idx=tag2idx)

    train_iters = Vocab.minibatches(train_data, batch_size=4)
    data = []
    label_ids = []
    c = 0
    for words, labels in train_iters:
        c += 1
        if len(words) == 0:
            print(words, labels, c)
        data.append(words)
        label_ids.append(labels)
        word_ids, sequence_lengths = seqPAD.pad_sequences(words, pad_tok=0, wthres=1024)

    w_tensor = Data2tensor.idx2tensor(word_ids)
    y_tensor = Data2tensor.idx2tensor(labels)

    data_tensors = Data2tensor.sort_tensors(labels, word_ids, sequence_lengths)
    label_tensor, word_tensor, sequence_lengths, word_seq_recover = data_tensors


