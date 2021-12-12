#!/usr/bin/env python

import collections, time, math, random, os
import torch, re, sys
import torch.nn.functional as F
import numpy as np
from sklearn.neural_network import MLPClassifier
from googletrans import Translator
from tqdm import tqdm
import baseline
import naive_bayes as NB

FREQUENCY_DICT = torch.load('data/frequency_dict')
class Net(torch.nn.Module):
    def  __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2, 480),    
            torch.nn.ReLU(),
            torch.nn.Linear(480, 240),      
            torch.nn.ReLU(),
            torch.nn.Linear(240, 120),      
            torch.nn.ReLU(),
            torch.nn.Linear(120, 64), 
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 7),
            torch.nn.Softmax(dim=0)
        )

    def forward(self, x):
        return self.layers(x)

'''
split hsk data into train and dev sets
train = 5/6th of data, dev = 1/6th
'''
def hsk_data():
    X = []
    train = []
    dev = []
    vocab = []
    path = "data/HSK"
    for file in os.listdir(path):
        if file == 'characters':
            continue
        file_path = path + '/' + file
        level = int(file.split('HSK')[1].split('.txt')[0])
        words = open(file_path, 'r').read().split('\n')
        random.shuffle(words)
        size = len(words)
        for i, word in enumerate(words):
            X.append([word, level])
            if i < 5*size/6:
                train.append([word, level])
            else:
                dev.append([word, level])
            for c in word:
                vocab.append(c)
    torch.save(X, 'data/HSK/all_vocab')
    torch.save(set(vocab), 'data/HSK/characters')
    torch.save(train, 'data/train')
    torch.save(dev, 'data/dev')
    return X

'''
annotate input file with list of advanced words
'''
def write_study_guide(advanced_words, file):
    advanced_words_set = set(advanced_words)
    S = '\033[4m'
    E = '\033[0m'
    text = open("data/test/segmented_text/" + file, 'r').read()
    segments = text.split(' ')
    outfile = open("results/" + file, "w")
    for s in tqdm(segments):
        if s in advanced_words_set:
            outfile.write(S + s + E + baseline.translate(s))
            advanced_words_set.remove(s) #dont annotate twice
        else:
            outfile.write(s)
    outfile.close()

def encode(word):
    if word not in FREQUENCY_DICT:
        word_freq = 0
    else:
        word_freq = FREQUENCY_DICT[word]
    NB_result = NB.test_word(word)
    return [word_freq, NB_result]

def test_net(m):
    m.eval()
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
            if baseline.num_or_eng(s):
                continue
            x = encode(s)
            x = torch.tensor(x, dtype=torch.float)
            pred = m(x)
            level_pred = torch.argmax(pred).item()
            if level_pred >= 5:
                found.append(s)

        #write_study_guide(found, file)

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
    print(f'Total F1: {total_f1/file_count}')

'''
encode training and dev data
'''
def train_encode():
    train = torch.load('data/train')
    dev = torch.load('data/dev')

    train_encoding = []
    for entry in train:
        word, level = entry
        if baseline.num_or_eng(word):
            continue
        encoding = encode(word)
        encoding = torch.tensor(encoding, dtype=torch.float)
        train_encoding.append([encoding, level])

    dev_encoding = []
    for entry in dev:
        word, level = entry
        if baseline.num_or_eng(word):
            continue
        encoding = encode(word)
        encoding = torch.tensor(encoding, dtype=torch.float)
        dev_encoding.append([encoding, level])

    torch.save(train_encoding, 'data/train.encoded')
    torch.save(dev_encoding, 'data/dev.encoded')

def train_net(m):
    train_encoding = torch.load('data/train.encoded')
    dev_encoding = torch.load('data/dev.encoded')

    optimizer = torch.optim.SGD(m.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    prev_dev_acc = None
    best_dev_acc = 0
    for epoch in range(100):
        m.train()
        random.shuffle(train_encoding)
        for entry in tqdm(train_encoding):
            e, level = entry
            pred = m(e)
            outputs = pred.unsqueeze(dim=0)
            level_pred = torch.argmax(pred).item()
            loss = criterion(outputs, torch.tensor([level]))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)            
            optimizer.step()
        
        dev_loss = 0.
        dev_words = dev_correct = 0
        m.eval()
        random.shuffle(dev_encoding)
        for entry in dev_encoding:
            dev_words += 1
            e, level = entry
            pred = m(e)
            outputs = pred.unsqueeze(dim=0)
            level_pred = torch.argmax(pred).item()
            loss = criterion(outputs, torch.tensor([level]))
            dev_loss -= loss
            if level_pred == level:
                dev_correct += 1

        dev_acc = dev_correct/dev_words

        print(f'epoch={epoch+1} dev_loss={dev_loss} dev_acc={dev_acc}')

        if prev_dev_acc is not None and dev_acc <= prev_dev_acc:
            optimizer.param_groups[0]['lr'] *= 0.5
            print(f"lr={optimizer.param_groups[0]['lr']}")

        if dev_acc > best_dev_acc:
            torch.save(m, 'model')
            best_dev_acc = dev_acc

        prev_dev_acc = dev_acc
        if epoch % 10 == 0:
            test_net(m)
        

def main():
    # HSK ANN
    net = Net()
    train_encode()
    #train_net(net)
    #m = torch.load('model')
    #test_net(m)

    #train()

if __name__ == "__main__":
    main()