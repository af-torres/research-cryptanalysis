import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
import pickle
import random
from model import LSTM_model, get_loss
from datasets import load_dataset
import string
import time

if torch.cuda.is_available():
    dev = "cuda"
    print('Running on CUDA')
else: 
    dev = "cpu"
    print('Running on CPU')

device = torch.device(dev)

noise_std = np.array([0.])#np.array(range(0, 45, 5)) / 100
use_positional_enc = False
alphabet = string.printable

DATASETS = [
    {
        "DATASET_ENC": './data/random/substitutionCipherArr-encryptedRandomCharSeq.csv',
        "DATASET_ORI": './data/random/arr-decryptedRandomCharSeq.csv',
        "RESULTS_FILE": "./results/random/lstm-substitutionCipher.pkl",
    },
    {
        "DATASET_ENC": './data/random/transpositionCipherArr-encryptedRandomCharSeq.csv',
        "DATASET_ORI": './data/random/arr-decryptedRandomCharSeq.csv',
        "RESULTS_FILE": "./results/random/lstm-transpositionCipher.pkl",
    },
    {
        "DATASET_ENC": './data/random/productCipherArr-encryptedRandomCharSeq.csv',
        "DATASET_ORI": './data/random/arr-decryptedRandomCharSeq.csv',
        "RESULTS_FILE": "./results/random/lstm-productCipher.pkl",
    },
    {
        "DATASET_ENC": './data/eng_sentences/substitutionCipherArr-encryptedEngSeq.csv',
        "DATASET_ORI": './data/eng_sentences/arr-decryptedEngSeq.csv',
        "RESULTS_FILE": "./results/eng_sentences/lstm-substitutionCipher.pkl",
    },
    {
        "DATASET_ENC": './data/eng_sentences/transpositionCipherArr-encryptedEngSeq.csv',
        "DATASET_ORI": './data/eng_sentences/arr-decryptedEngSeq.csv',
        "RESULTS_FILE": "./results/eng_sentences/lstm-transpositionCipher.pkl",
    },
    {
        "DATASET_ENC": './data/eng_sentences/productCipherArr-encryptedEngSeq.csv',
        "DATASET_ORI": './data/eng_sentences/arr-decryptedEngSeq.csv',
        "RESULTS_FILE": "./results/eng_sentences/lstm-productCipher.pkl",
    },
]

def build_model(noise_std, num_classes, seq_len):
    input_size = 1 # (fixed for the data) input dimesion
    output_size = num_classes # (fixed for the data) output dimesion
    PAD_INDEX = 0 # number used to pad sequences
    num_layers = 1 # number of stacked lstm layers
    hidden_size = 2000 # (fixed for the data) number of features in hidden state
    padding_idx = 0
    learning_rate = 1e-3 # lr
    weight_decay = 1e-2

    model = LSTM_model(output_size, input_size, hidden_size, num_layers)

    lossFunc = nn.CrossEntropyLoss(ignore_index=padding_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return {
        "model": model,
        "lossFunc": lossFunc,
        "optimizer": optimizer,
    }