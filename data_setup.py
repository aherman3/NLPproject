import spacy
import os
import pickle
import math

nlp = spacy.load("zh_core_web_sm")

def segment_thucnews():
    path = "data/THUCNews"
    for dir in os.listdir(path):
        if dir.startswith('.'):
            continue
        dir_path = path + '/' + dir
        for file in os.listdir(dir_path):
            filename = dir_path + '/' + file
            write_to = open("data/segmented_THUCNews" + '/' + dir + '/' + file, 'w')
            text = open(filename, 'r').read().replace('\u3000', '')
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
    outfile = open("data/frequency_dict.pkl", 'wb')
    path = "data/segmented_THUCNews"
    for dir in os.listdir(path):
        if dir.startswith('.'):
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
    pickle.dump(FREQUENCY_DICT, outfile)

def load_frequencies():
    infile = open("data/frequency_dict.pkl", 'rb')
    FREQUENCY_DICT = pickle.load(infile)
    infile.close()
    return FREQUENCY_DICT

def calculate_standard_dev(d):
    total = 0
    for key in d:
        total += d[key]
    average = total/len(d) # = 34

    total = 0
    for k in d:
        s = pow(d[key] - average, 2)
        total += s
    variance = total/len(d)

    sd = math.sqrt(variance)
    return sd # = 33

def main():
    FREQUENCY_DICT = load_frequencies()
    print(FREQUENCY_DICT['就'])

if __name__ == "__main__":
    main()