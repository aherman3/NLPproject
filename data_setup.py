#!/usr/bin/env python

'''
segment data using spacy
create frequency dict from THUCnews articles
'''

import spacy, torch
import os
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


'''
create and save dict of words from THUCnews w word frequencies
'''
def count_frequencies():
    FREQUENCY_DICT = {}
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
    torch.save(FREQUENCY_DICT, 'data/frequency_dict')

def main():
    count_frequencies()

if __name__ == "__main__":
    main()