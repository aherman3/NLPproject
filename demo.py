#!/usr/bin/env python

'''
apply final model to input sentence or text file
'''
import sys, torch, spacy
from neural_net import *
from baseline import *

nlp = spacy.load("zh_core_web_sm")

'''
annotate input sentence or file
'''
def write_study_guide_demo(m, t, type):
    found = []
    if type == 't':
        path = "results/demo_temp.txt"
    else:
        path = "results/" + type 
    outfile = open(path, "w")

    segments = nlp(t)
    for s in segments:
        s = s.text
        encoding = torch.tensor(encode(s), dtype=torch.float)
        pred = m(encoding)
        level_pred = torch.argmax(pred).item()
        if level_pred >= 5:
            found.append(s)
            outfile.write(S + s + E + translate(s))
        else:
            outfile.write(s)
    outfile.close()
    cat = open(path, 'r').read()
    print(cat)

def main():
    m = torch.load('model')

    # demo
    if len(sys.argv) > 1:
        if sys.argv[1] == '-t': # demo with text input
            input = ' '.join(sys.argv[2::])
            write_study_guide_demo(m, input, 't')
        if sys.argv[1] == '-f': # demo with file input
            file = open(sys.argv[2], 'r').read()
            filename = sys.argv[2].split('/')[-1]
            write_study_guide_demo(m, file, filename)

if __name__ == "__main__":
    main()