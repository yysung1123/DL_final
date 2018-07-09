

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
    def __init__(self, chapters):
        super().__init__()
        self.corpus = [parse(i) for i in range(1, 121)]
        self.chapters = chapters
        self.chapter2label = {}
        for idx, chapter in enumerate(self.chapters):
            self.chapter2label[chapter] = idx
        self.sentences = []
        for idx, chapter in enumerate(self.corpus):
            if idx // 10 not in self.chapters:
                continue
            for sentence in chapter:
                self.sentences.append((sentence, self.chapter2label[idx // 10]))
        shuffle(self.sentences)
        self.vocab = self._build_vocab()
        self.train_size = len(self.sentences) * 9 // 10
        self.test_size = len(self.sentences) - self.train_size
        self.train = self.sentences[:self.train_size]
        self.test = self.sentences[self.train_size:]
        self.sen_len = 64

    def _build_vocab(self):
        cnt = Counter()
        for chapter in self.corpus:
            for sentence in chapter:
                for word in sentence:
                    cnt[word] += 1
        words = [word for word, cnt in cnt.items() if cnt >= 3]

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
            if idx >= self.corpus.sen_len:
                break
            txt[idx] = self.corpus.vocab(word)
        label = torch.LongTensor([self.sentences[index][1]])
        sen_len = min(self.corpus.sen_len, len(self.sentences[index][0]))
        return txt, label, sen_len


    def __len__(self):
        return len(self.sentences)

def choose_chapters1():
    chapters = list(range(8))
    shuffle(chapters)
    for idx in range(4, 8):
        chapters[idx] = idx + 4
    print_chapter(chapters)
    return chapters


def choose_chapters2():
    chapters = list(range(8))
    shuffle(chapters)
    print_chapter(chapters)
    return chapters

def print_chapter(chapters):
    print("choose chapters:")
    for label in chapters:
        print("{}~{}".format(label * 10 + 1, (label + 1) * 10))
