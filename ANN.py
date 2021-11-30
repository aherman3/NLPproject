#!/usr/bin/env python

import collections, time, math, random, os
from pickle import encode_long

from evaluate import E, S, evaluate
import torch, re
import torch.nn.functional as F
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from googletrans import Translator
from tqdm import tqdm
import evaluate

FREQUENCY_DICT = torch.load('data/frequency_dict')
CHAR_FREQUENCY_DICT = torch.load('data/char_frequency_dict')

class Net(torch.nn.Module):
    def  __init__(self, d_in, d_out, parameters=None):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=d_in,
                             out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, 
                             out_features=84)
        self.fc3 = torch.nn.Linear(in_features=84,  
                             out_features=d_out)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        return torch.sigmoid(self.fc3(X))

def translate(s):
    translator = Translator()
    source_lan = "zh-cn"
    translated_to = "en"
    try:
        translation = translator.translate(s, src=source_lan, dest = translated_to)
    except:
        return ""
    return "(" + translation.text + ")"

class Vocab:
    def __init__(self, words):
        self.num_to_word = list(words) + ['UNK']
        self.word_to_num = {word:num for num, word in enumerate(self.num_to_word)}

    def len(self):
        return len(self.num_to_word)

    def words(self):
        return self.num_to_word

    def numberize(self, word):
        if word in self.word_to_num:
            return self.word_to_num[word]
        else: 
            return self.word_to_num['UNK']

    def denumberize(self, num):
        return self.num_to_word[num]

def hsk_data():
    X = []
    train = []
    dev = []
    vocab = []
    path = "data/HSK"
    for file in os.listdir(path):
        if file == 'all_vocab' or file == 'characters':
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

def encode_chars(word, characters):
    encoding = [0 for x in range(10)]
    for i, c in enumerate(word):
        e = characters.numberize(c)
        encoding[i] = e 
    return encoding

def encode(word):
    if word not in FREQUENCY_DICT:
        word_freq = 0
    else:
        word_freq = FREQUENCY_DICT[word]
    return [word_freq]

def further_encode(word, characters):
    if word not in FREQUENCY_DICT:
        word_freq = 0
    else:
        word_freq = FREQUENCY_DICT[word]
    char_freq = [0 for x in range(10)]
    for i, c in enumerate(word):
        if c not in CHAR_FREQUENCY_DICT:
            char_freq[i] = 0
        else:
            char_freq[i] = CHAR_FREQUENCY_DICT[c]
    encoding = encode_chars(word, characters)
    return [word_freq] #+ char_freq + encoding + [len(word)] + [evaluate.stroke_count(word)]

def train():
    vocab = torch.load('data/HSK/all_vocab')
    characters = torch.load('data/HSK/characters')
    characters = Vocab(characters)

    X_train = []
    X_train2 = []
    Y = []
    random.shuffle(vocab)
    for entry in vocab:
        word, level = entry
        if evaluate.num_or_eng(word):
                continue
        encoding = encode(word)
        #further_encoding = further_encode(word, characters)
        X_train.append(encoding)
        #X_train2.append(further_encoding)
        Y.append(level)

    clf1 = MLPClassifier(random_state=1, max_iter=300, activation='logistic', learning_rate='adaptive', early_stopping=True, hidden_layer_sizes=(5,))
    clf1 = clf1.fit(X_train, Y)

    #clf2 = RandomForestClassifier(max_depth=2, random_state=0)
    #clf2 = clf2.fit(X_train2, Y)

    model_f1 = test(clf1, clf1, characters)
    print(model_f1)

def test(model1, model2, characters):
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
            if evaluate.num_or_eng(s):
                continue
            advanced = model1.predict([encode(s)])
            if advanced > 5:
                found.append(s)
        
        '''
        updated_found = []
        for word in found:
            advanced = model2.predict([further_encode(word, characters)])
            if advanced > 5:
                updated_found.append(word)
        found = updated_found
        '''

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
    return total_f1/file_count

def test_net(m):
    characters = torch.load('data/HSK/characters')
    characters = Vocab(characters)
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
            if evaluate.num_or_eng(s):
                continue
            x = further_encode(s, characters)
            x = torch.tensor(x, dtype=torch.float)
            pred = m(x)
            level_pred = torch.argmax(pred).item()
            if level_pred > 5:
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
    print(f'Total F1: {total_f1/file_count}')


def write_study_guide(advanced_words, file):
    advanced_words_set = set(advanced_words)
    S = '\033[4m'
    E = '\033[0m'
    text = open("data/test/segmented_text/" + file, 'r').read()
    segments = text.split(' ')
    outfile = open("results/" + file, "w")
    for s in tqdm(segments):
        if s in advanced_words_set:
            outfile.write(S + s + E + translate(s))
            advanced_words_set.remove(s) #dont annotate twice
        else:
            outfile.write(s)
    outfile.close()

def train_net(m):
    vocab = torch.load('data/HSK/all_vocab')
    train = torch.load('data/train')
    dev = torch.load('data/dev')
    characters = torch.load('data/HSK/characters')
    characters = Vocab(characters)

    train_encoding = []
    for entry in train:
        word, level = entry
        if evaluate.num_or_eng(word):
            continue
        further_encoding = further_encode(word, characters)
        further_encoding = torch.tensor(further_encoding, dtype=torch.float)
        train_encoding.append([further_encoding, level])

    dev_encoding = []
    for entry in dev:
        word, level = entry
        if evaluate.num_or_eng(word):
            continue
        further_encoding = further_encode(word, characters)
        further_encoding = torch.tensor(further_encoding, dtype=torch.float)
        dev_encoding.append([further_encoding, level])

    optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
    prev_dev_acc = None
    for epoch1 in range(10):
        m.train()
        for epoch2 in tqdm(range(100)):
            random.shuffle(train_encoding)
            loss = 0.
            for entry in train_encoding:
                e, level = entry
                pred = m(e)
                level_pred = torch.argmax(pred).item()
                loss -= pred[level_pred]

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
            level_pred = torch.argmax(pred).item()
            print(level, level_pred)
            dev_loss -= pred[level_pred]
            if level_pred == level:
                dev_correct += 1

        dev_acc = dev_correct/dev_words

        print(f'epoch={epoch1+1} dev_ppl={math.exp(dev_loss/dev_words)} dev_acc={dev_acc}')

        if prev_dev_acc is not None and dev_acc <= prev_dev_acc:
            optimizer.param_groups[0]['lr'] *= 0.5
            print(f"lr={optimizer.param_groups[0]['lr']}")

        test_net(m)
        

def main():
    # HSK ANN
    net = Net(1, 7)
    train_net(net)

if __name__ == "__main__":
    main()