from __future__ import print_function
from math import ceil
import argparse
import numpy as np
import sys
import pdb
import os
import random
import pickle
import json

from nltk.tokenize import RegexpTokenizer
import torch
import torch.optim as optim
import torch.nn as nn

import generator
import discriminator
import helpers

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

CUDA = True
VOCAB_SIZE = 1457
MAX_SEQ_LEN = 9
START_LETTER = 0
MLE_TRAIN_EPOCHS = 100
ADV_TRAIN_EPOCHS = 20
POS_NEG_SAMPLES = 30000

GEN_EMBEDDING_DIM = 300
GEN_HIDDEN_DIM = 300
DIS_EMBEDDING_DIM = 300
DIS_HIDDEN_DIM = 300

class words_dict():
    def __init__(self):
        self.word_count = {}
        self.id_to_word = {0: '_sos_', 1: '_eos_', 2: '_unk_'}
        self.word_to_id = {'_sos_': 0, '_eos_': 1, '_unk_': 2}
        self.n_words = 3
        self.remain_id = []
        self.max_len = 200

def main(dictionary, input_sam):
    gen_path = f'model/ADV_gen_MLEtrain_EMBDIM_300_HIDDENDIM300_VOCAB1457_MAXSEQLEN9_19_06new'

    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED) 
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(0)

    with open('data/glove_id_to_emb.pkl', 'rb') as f:
        emb_dict = pickle.load(f)

    input_sam = torch.tensor([0]+[dictionary.word_to_id[c] for c in input_sam]+[1]).reshape(1,-1)
    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA, emb_dict=emb_dict)

    if CUDA:
        gen = gen.cuda()

    # LOAD GENERATOR 
    gen.load_state_dict(torch.load(gen_path))
    # valid_samples = torch.load(testing_samples_path).type(torch.LongTensor)
    # valid_id = torch.load(testing_id_path)
    if CUDA:
        gen = gen.cuda()
        input_sam = input_sam.cuda()
    with torch.no_grad():
        pred = gen.predict(input_sam, dictionary)
        pred = ''.join(pred[0][1:8])

    print (pred)

if __name__=='__main__':
    dictionary = words_dict()
    with open('./data/train_dict_cut.pkl', 'rb') as f:
        dictionary = pickle.load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument("first_scroll")
    args = parser.parse_args()
    main(dictionary, args.first_scroll)