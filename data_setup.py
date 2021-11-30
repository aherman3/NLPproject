#!/usr/bin/env python

import spacy, torch
import os
import pickle
import math
from cjklib import characterlookup
import re

nlp = spacy.load("zh_core_web_sm")

def segment_thucnews():
    path = "data/cleaned_THUCNews"
    for dir in os.listdir(path):
        if dir.startswith('.'):
            continue
        dir_path = path + '/' + dir
        for file in os.listdir(dir_path):
            filename = dir_path + '/' + file
            print(filename)
            write_to = open("data/segmented_THUCNews" + '/' + dir + '/' + file, 'w')
            try:
                text = open(filename, 'r').read().replace('\u3000', '')
            except:
                continue
            doc = nlp(text)
            for token in doc:
                write_to.write(token.text + ' ')

def segment_test():
    path = "data/test/text"
    for file in os.listdir(path):
        file_path = path + '/' + file
        write_to = open("data/test/segmented_text/" + file, 'w')
        text = open(file_path, 'r').read().replace('\u3000', '')
        doc = nlp(text)
        for token in doc:
            write_to.write(token.text + ' ')

def count_frequencies():
    FREQUENCY_DICT = {}
    CHAR_FREQUENCY_DICT = {}
    path = "data/segmented_THUCNews"
    for dir in os.listdir(path):
        if dir.startswith('.') or dir.__contains__('annotations'):
            continue
        dir_path = path + '/' + dir
        for file in os.listdir(dir_path):
            filename = dir_path + '/' + file
            text = open(filename, 'r').read().replace('\n', '')
            segments = text.split(' ')
            for s in segments:
                if s in FREQUENCY_DICT:
                    FREQUENCY_DICT[s] += 1
                else:
                    FREQUENCY_DICT[s] = 1
                for c in s:
                    if c in CHAR_FREQUENCY_DICT:
                        CHAR_FREQUENCY_DICT[c] += 1
                    else:
                        CHAR_FREQUENCY_DICT[c] = 1
    torch.save(FREQUENCY_DICT, 'data/frequency_dict')
    torch.save(CHAR_FREQUENCY_DICT, 'data/char_frequency_dict')

def calculate_standard_dev(d):
    total = 0
    for key in d:
        total += d[key]
    average = total/len(d) # = 34, jieba=65
    print(average)

    total = 0
    for k in d:
        s = pow(d[key] - average, 2)
        total += s
    variance = total/len(d)

    sd = math.sqrt(variance)
    print(sd)
    return sd # = 33, jieba=64

def main():
    segment_thucnews()
    count_frequencies()

if __name__ == "__main__":
    main()