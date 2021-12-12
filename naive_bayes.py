#!/usr/bin/env python
import os
import re
import torch

alpha = 0.1

'''
create hsk dict for easy parsing
'''
def hsk_list():
    HSK_DICT = {}
    path = "data/HSK"
    for file in os.listdir(path):
        if file == 'characters' or file == 'all_vocab':
            continue
        file_path = path + '/' + file
        level = int(file.split('HSK')[1].split('.txt')[0])
        words = open(file_path, 'r').read().split('\n')

        for word in words:
            HSK_DICT[word] = level
    torch.save(HSK_DICT, 'data/bayes_params/HSK_DICT')

'''
create naive bayes model using HSK data
HSK words as 'documents' and characters as bag of words
'''
def NB(HSK):
    PK = {}
    WK_IND_COUNTS = {}
    WK_COUNTS = {}
    total_lines = 0
    for s in HSK.keys():
        total_lines += 1
        k = HSK[s]
        if k not in PK:
            PK[k] = 0
        PK[k] += 1
    for k in PK.keys():
        count = PK[k]
        PK[k] = count/total_lines

    for s in HSK.keys():
        k = HSK[s]
        for c in s:
            if c not in WK_IND_COUNTS:
                WK_IND_COUNTS[c] = {}
            if k not in WK_IND_COUNTS[c]:
                WK_IND_COUNTS[c][k] = alpha
            WK_IND_COUNTS[c][k] += 1

            if k not in WK_COUNTS:
                WK_COUNTS[k] = 0
            WK_COUNTS[k] += 1

    torch.save((PK, WK_IND_COUNTS, WK_COUNTS), 'data/bayes_params/NB')

'''
test NB with input word
'''
def test_word(word):
    PK, WK_IND_COUNTS, WK_COUNTS = torch.load('data/bayes_params/NB')
    PROBS = {}
    for k in PK:
        pk = PK[k]
        prod_w = 1
        for c in word:
            try:
                ckw = WK_IND_COUNTS[c][k]
            except:
                ckw = alpha
            ckwp = WK_COUNTS[k]
            prod_w *= ckw/ckwp
        pkd = pk*prod_w
        PROBS[k] = pkd
    sol = max(PROBS, key=PROBS.get)
    return sol

'''
compute f1 for NB
'''
def test():
    total_f1 = 0
    path = "data/test/segmented_text"
    file_count = 0
    for file in os.listdir(path):
        file_count += 1
        file_path = path + '/' + file
        text = open(file_path, 'r').read()
        segments = text.split(' ')
        found = []
        for s in segments:
            advanced = test_word(s)
            if advanced > 5:
                found.append(s)
    
        real = open("data/test/vocab/" + file, 'r').read().split('\n')
        false_pos = 0
        true_pos = 0
        false_neg = 0
        for word in set(found):
            if word not in real:
                false_pos += 1
            if word in real:
                true_pos += 1
        for word in real:
            if word not in found:
                false_neg += 1
        f_score = true_pos / (true_pos + 0.5*(false_pos + false_neg))
        total_f1 += f_score
        print(f'{file} F1: {f_score}')
        print(f'fp: {false_pos}, fn: {false_neg}, tp: {true_pos}')
    print(total_f1/file_count)
    return total_f1/file_count

def main():
    HSK_DICT = torch.load('data/bayes_params/HSK_DICT')

    NB(HSK_DICT)
    test()

if __name__ == "__main__":
    main()