#!/usr/bin/env python
import os
import re
import torch

class Bayes(torch.nn.Module):
    def __init__(self, dims, HSK_LEVEL_COUNT):
        super().__init__()
        self.HSK_LEVEL_COUNT = HSK_LEVEL_COUNT
        self.L = torch.nn.Parameter(torch.empty(dims))
        self.total_words = sum([HSK_LEVEL_COUNT[i] for i in range(7)])
        unk_Pwk = sum([1/HSK_LEVEL_COUNT[i] for i in range(7)])
        self.unk_Pwk = torch.tensor(unk_Pwk)
        sum_k = 0
        for level in range(7):
            sum_k += HSK_LEVEL_COUNT[level]/self.total_words
        self.sum_k = sum_k

    def calculatePd(self, words, HSK_DICT):
        prod_w = torch.tensor(1)
        for w in words:
            if w in HSK_DICT: # P(w | k) = P(t | k) P(w | t)
                prod_w = torch.logaddexp(prod_w, torch.tensor(1/self.HSK_LEVEL_COUNT[HSK_DICT[w]]))
            else: # P(w | k) = \sum_t P(t | k) P(w | t)
                prod_w = torch.logaddexp(prod_w, self.unk_Pwk)

        Pd = self.sum_k * prod_w
        return Pd

    def predict(self, words, HSK_DICT):
        Pd = self.calculatePd(words, HSK_DICT)
        return Pd
        

def hsk_list():
    hsk_data = torch.load('data/HSK/all_vocab')
    HSK_DICT = {}
    HSK_LEVEL_COUNTS = {
        0:0,
        1:0,
        2:0,
        3:0,
        4:0,
        5:0,
        6:0
    }
    for entry in hsk_data:
        word, level = entry
        HSK_DICT[word] = level
        HSK_LEVEL_COUNTS[level] += 1
    torch.save(HSK_DICT, 'data/bayes_params/HSK_DICT')
    torch.save(HSK_LEVEL_COUNTS, 'data/bayes_params/HSK_LEVEL_COUNTS')

def count_hsk(m, HSK_DICT, HSK_LEVEL_COUNT):
    m.train()
    path = "data/segmented_THUCNews"
    for dir in os.listdir(path):
        if dir.startswith('.') or dir.__contains__('annotations'):
            continue
        dir_path = path + '/' + dir
        for file in os.listdir(dir_path):
            filename = dir_path + '/' + file
            text = open(filename, 'r').read().replace('\u3000', '')
            words = text.split(' ')

            Pd = m.calculatePd(words, HSK_DICT)
            m.L += torch.log(Pd)
    test(m, HSK_DICT)


def main():
    HSK_DICT = torch.load('data/bayes_params/HSK_DICT')
    HSK_LEVEL_COUNT = torch.load('data/bayes_params/HSK_LEVEL_COUNTS')

    model = Bayes(100, HSK_LEVEL_COUNT)

    count_hsk(model, HSK_DICT, HSK_LEVEL_COUNT)

if __name__ == "__main__":
    main()