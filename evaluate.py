import pickle
import os
import googletrans
from googletrans import Translator
import re
from tqdm import tqdm

def load_frequencies():
    infile = open("data/frequency_dict.pkl", 'rb')
    FREQUENCY_DICT = pickle.load(infile)
    infile.close()
    return FREQUENCY_DICT

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
    return f_score

def write_study_guide(d):
    # mean & standard deviation calculated in data_setup.py calculate_standard_dev()
    mean = 34
    sd = 33
    min_freq = mean + 2*sd

    path = "data/test/segmented_text"
    for file in os.listdir(path):
        file_path = path + '/' + file
        text = open(file_path, 'r').read()
        segments = text.split(' ')
        found = []
        outfile = open("results/" + file, "w")
        for s in tqdm(segments):
            if num_or_eng(s): # skip numbers and english words
                outfile.write(s)
                continue
            if s in d:
                if d[s] < min_freq:
                    found.append(s)
                    outfile.write(s + translate(s))
                else:
                    outfile.write(s)
            else:
                found.append(s)
                outfile.write(s + translate(s))

def analyze_frequency(d):
    # mean & standard deviation calculated in data_setup.py calculate_standard_dev()
    mean = 34
    sd = 33
    min_freq = mean + 2*sd

    path = "data/test/segmented_text"
    for file in os.listdir(path):
        file_path = path + '/' + file
        text = open(file_path, 'r').read()
        segments = text.split(' ')
        found = []
        for s in segments:
            if num_or_eng(s): # skip numbers and english words
                continue
            if s in d:
                if d[s] < min_freq:
                    found.append(s)
            else:
                found.append(s)
        f_score = evaluate(file, found)
        print(f'{file} F1: {f_score}')
        
def translate(s):
    translator = Translator()
    source_lan = "zh-cn"
    translated_to = "en"
    try:
        translation = translator.translate(s, src=source_lan, dest = translated_to)
    except:
        return ""
    return "(" + translation.text + ")"


def num_or_eng(s):
    chinese_nums = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
    if any(i in chinese_nums for i in s):
        return True
    if any(i.isdigit() for i in s):
        return True
    if re.search(r'[a-zA-Z]', s):
        return True

def main():
    FREQUENCY_DICT = load_frequencies()
    analyze_frequency(FREQUENCY_DICT)
    #write_study_guide(FREQUENCY_DICT)

if __name__ == "__main__":
    main()