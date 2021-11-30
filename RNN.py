#!/usr/bin/env python

import collections, time, math, random, os, torch
import ANN

def parse_top10(top10, HSK_DICT):
    advanced_words = []
    for key in top10:
        text = open(key, 'r').read().split(' ')
        for word in text:
            if ANN.num_or_eng(word):
                continue
            if word not in HSK_DICT:
                advanced_words.append(word)
    print(advanced_words)
    torch.save(advanced_words, 'results/advanced_words')

def find_top10():
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

    top_10 = dict()
    prev_advanced_files = torch.load('results/advanced_filenames')

    path = "data/segmented_THUCNews"
    for dir in os.listdir(path):
        if dir.startswith('.') or dir.__contains__('annotations'):
            continue
        dir_path = path + '/' + dir
        for file in os.listdir(dir_path):
            filename = dir_path + '/' + file
            if filename in prev_advanced_files:
                continue
            text = open(filename, 'r').read().split(' ')
            word_count = 0
            unk_count = 0
            HSK_count = {0:0,1:0,2:0,3:0,4:0,5:0,6:0}
            for word in text:
                if ANN.num_or_eng(word):
                    continue
                word_count += 1
                if word in HSK_DICT:
                    HSK_count[HSK_DICT[word]] += 1
                else:
                    unk_count += 1

            max_prob = 0
            max_level = None
            for key in HSK_count:
                prob = HSK_count[key]/word_count
                if prob > max_prob:
                    max_prob = prob
                    max_level = level
            
            if max_level == 6:
                if len(top_10) < 10:
                    top_10[filename] = max_prob
                else:
                    temp = min(top_10.values())
                    lowest_files = [key for key in top_10 if top_10[key] == temp][0]
                    del top_10[lowest_files]
                    top_10[filename] = max_prob

    for file in top_10:
        prev_advanced_files.append(file)
    torch.save(prev_advanced_files, 'results/advanced_filenames')
    parse_top10(top_10, HSK_DICT)

def main():
    for epoch in range(10):
        find_top10()

        # add new advanced words to word/char vocabs
        new_advanced_words = torch.load('results/advanced_words')
        all_vocab = torch.load('data/HSK/all_vocab')
        all_characters = list(torch.load('data/HSK/characters'))
        for word in new_advanced_words:
            all_vocab.append([word, 6])
            for c in word:
                all_characters.append(c)
        torch.save(all_vocab, 'data/HSK/all_vocab')
        torch.save(set(all_characters), 'data/HSK/characters')

        # retrain with new words
        vocab = torch.load('data/HSK/all_vocab')
        ANN.train(vocab)

if __name__ == "__main__":
    main()