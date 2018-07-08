

from collections import Counter
from torch.utils.data import Dataset
import numpy as np
import torch


from random import shuffle
from utils import parse


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0


    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1


    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]


    def __len__(self):
        return len(self.word2idx)


class Corpus:
    def __init__(self):
        super().__init__()
        self.corpus = [parse(i) for i in range(1, 121)]
        self.sentences = []
        for idx, chapter in enumerate(self.corpus):
            for sentence in chapter:
                self.sentences.append((sentence, idx + 1))
        self.vocab = self._build_vocab()
        self.train_size = len(self.sentences) * 9 // 10
        self.test_size = len(self.sentences) - self.train_size
        self.train = self.sentences[:self.train_size]
        self.test = self.sentences[self.train_size:]
        self.sen_len = 0
        for chapter in self.corpus:
            for sentence in chapter:
                self.sen_len = max(self.sen_len, len(sentence))

    def _build_vocab(self):
        cnt = Counter()
        for chapter in self.corpus:
            for sentence in chapter:
                for word in sentence:
                    cnt[word] += 1
        words = [word for word, cnt in cnt.items() if cnt >= 5]

        vocab = Vocabulary()
        vocab.add_word('<pad>')
        vocab.add_word('<unk>')
        for word in words:
            vocab.add_word(word)
        return vocab

class TextDataset(Dataset):
    def __init__(self, corpus, train=True):
        self.corpus = corpus
        if train:
            self.sentences = corpus.train
        else:
            self.sentences = corpus.test


    def __getitem__(self, index):
        txt = torch.LongTensor(np.zeros(self.corpus.sen_len, dtype=np.int64))
        for idx, word in enumerate(self.sentences[index][0]):
            txt[idx] = self.corpus.vocab(word)
        label = torch.LongTensor([(self.sentences[index][1] - 1) // 40])
        sen_len = len(self.sentences[index][0])
        return txt, label, sen_len


    def __len__(self):
        return len(self.sentences)


class TextDataLoader:
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.samples = [i for i in dataset]
        self.n_samples = len(self.samples)
        self.n_batches = self.n_samples // self.batch_size


    def _shuffle_indices(self):
        self.indices = np.random.permutation(self.n_samples)
        self.index = 0
        self.batch_index = 0


    def _create_batch(self):
        batch_input = []
        batch_target = []
        batch_len = []
        n = 0
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch_input.append(self.samples[_index][0])
            batch_target.append(self.samples[_index][1])
            batch_len.append(self.samples[_index][1])
            self.index += 1
            n += 1
        self.batch_index += 1
        return batch_input, batch_target, batch_len


    def __iter__(self):
        self._shuffle_indices()
        for i in range(self.n_batches):
            if self.batch_index == self.n_batches:
                raise StopIteration()
            yield self._create_batch()
