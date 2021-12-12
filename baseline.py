#!/usr/bin/env python

'''
baseline:
- use frequency dict to find freq of each word in test files
- if frequency below certain limit, considered advanced
- write study guide with advanced words underlined and translated with googletrans
'''

import os
from googletrans import Translator
import re
from tqdm import tqdm
import sys
import spacy
import torch
from cjklib import characterlookup

nlp = spacy.load("zh_core_web_sm")
cjk = characterlookup.CharacterLookup('T')

S = '\033[4m'
E = '\033[0m'

'''
get gooogletrans English translation of Chinese word
'''
def translate(s):
    translator = Translator()
    source_lan = "zh-cn"
    translated_to = "en"
    try:
        translation = translator.translate(s, src=source_lan, dest = translated_to)
    except:
        return ""
    return "(" + translation.text + ")"

'''
if word contains numbers, English letters, or punctuation return False
'''
def num_or_eng(s):
    chinese_nums = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
    if len(s) == 0:
        return True
    if any(i in chinese_nums for i in s):
        return True
    if any(i.isdigit() for i in s):
        return True
    if re.search(r'[a-zA-Z]', s):
        return True
    my_punct = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '.', '…', '（', '）', '·', '',
           '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '、', '！', '—', '\xa0',
           '`', '{', '|', '}', '~', '»', '«', '“', '”', '\n', '。', '，', '《', '》', '：', '；', '【', '】', '？', '-']
    if any(i in my_punct for i in s):
        return True
    return False

def evaluate(file, found):
    real = open("data/test/vocab/" + file, 'r').read().split('\n')
    false_pos = 0
    true_pos = 0
    false_neg = 0
    for word in found:
        if word not in real:
            false_pos += 1
        if word in real:
            true_pos += 1
    for word in real:
        if word not in found:
            false_neg += 1
    f_score = true_pos / (true_pos + 0.5*(false_pos + false_neg))
    print(f'fp: {false_pos}, fn: {false_neg}, tp: {true_pos}')
    return f_score

'''
annotate all test files
input: frequency dict
'''
def write_study_guide(d):
    path = "data/test/segmented_text"
    for file in os.listdir(path):
        file_path = path + '/' + file
        text = open(file_path, 'r').read()
        segments = text.split(' ')
        found = []
        outfile = open("results/" + file, "w")
        for s in tqdm(segments):
            advanced = check_advanced(s, d)
            if advanced:
                found.append(s)
                outfile.write(S + s + E + translate(s))
            else:
                outfile.write(s)
        outfile.close()


'''
annotate input sentence or file
'''
def write_study_guide_demo(d, t, type):
    found = []
    if type == 't':
        path = "results/demo_temp.txt"
    else:
        path = "results/" + type 
    outfile = open(path, "w")

    segments = nlp(t)
    for s in segments:
        s = s.text
        advanced = check_advanced(s, d)
        if advanced:
            found.append(s)
            outfile.write(S + s + E + translate(s))
        else:
            outfile.write(s)
    outfile.close()
    cat = open(path, 'r').read()
    print(cat)


def stroke_count(w):
    total = 0
    if len(w) == 0:
        return 0
    for c in w:
        try:
            total += cjk.getStrokeCount(c)
        except:
            return 0
    return total/len(w) # average stroke count of word

'''
frequencies above mean + 4*standard dev considered advanced
'''
def check_advanced(s, d):
    # mean & standard deviation calculated in data_setup.py calculate_standard_dev()
    mean = 34
    sd = 33
    min_freq = mean + 4*sd
    if num_or_eng(s): # skip numbers and english words
        return False
    if s in d:
        if d[s] < min_freq: # low freq, advanced
            strokes = stroke_count(s)
            if strokes > 5:
                return True
        return False
    return True # word never seen, advanced

'''
test using word frequency dict
'''
def analyze_frequency(d):
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
            advanced = check_advanced(s, d)
            if advanced:
                found.append(s)
        f_score = evaluate(file, found)
        total_f1 += f_score
        print(f'{file} F1: {f_score}')
    print(f'Total F1: {total_f1/file_count}')


def main():
    FREQUENCY_DICT = torch.load('data/frequency_dict')
    analyze_frequency(FREQUENCY_DICT)

    # demo
    if len(sys.argv) > 1:
        if sys.argv[1] == '-t': # demo with text input
            input = ' '.join(sys.argv[2::])
            write_study_guide_demo(FREQUENCY_DICT, input, 't')
        if sys.argv[1] == '-f': # demo with file input
            file = open(sys.argv[2], 'r').read()
            filename = sys.argv[2].split('/')[-1]
            write_study_guide_demo(FREQUENCY_DICT, file, filename)

if __name__ == "__main__":
    main()