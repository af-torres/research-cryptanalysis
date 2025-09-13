import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
import random
from model import AutoEncoder_model, get_loss
from datasets import load_dataset
import string
import time
from utils import load_results, get_accuracy, plot_acc_line

if torch.cuda.is_available():
    dev = "cuda"
    print('Running on CUDA')
else: 
    dev = "cpu"
    print('Running on CPU')

device = torch.device(dev)

noise_std = np.array(range(0, 45, 5)) / 100
alphabet = string.printable

DATASETS = [
    {
        "DATASET_ENC": '../cryptanalysis_old/datasets/dtEnc10k.csv',
        "DATASET_ORI": '../cryptanalysis_old/datasets/dtOri10k.csv',
        "RESULTS_FILE": "./results/random/substitutionCipher-old-data.pkl",
        "NAME": "OLD-DATASET",
    },
#    {
#        "DATASET_ENC": './data/random/substitutionCipherArr-encryptedRandomCharSeq.csv',
#        "DATASET_ORI": './data/random/arr-decryptedRandomCharSeq.csv',
#        "RESULTS_FILE": "./results/random/substitutionCipher.pkl",
#    },
#    {
#        "DATASET_ENC": './data/random/transpositionCipherArr-encryptedRandomCharSeq.csv',
#        "DATASET_ORI": './data/random/arr-decryptedRandomCharSeq.csv',
#        "RESULTS_FILE": "./results/random/transpositionCipher.pkl",
#    },
#    {
#        "DATASET_ENC": './data/random/productCipherArr-encryptedRandomCharSeq.csv',
#        "DATASET_ORI": './data/random/arr-decryptedRandomCharSeq.csv',
#        "RESULTS_FILE": "./results/random/productCipher.pkl",
#    },
#    {
#        "DATASET_ENC": './data/eng_sentences/substitutionCipherArr-encryptedEngSeq.csv',
#        "DATASET_ORI": './data/eng_sentences/arr-decryptedEngSeq.csv',
#        "RESULTS_FILE": "./results/eng_sentences/substitutionCipher.pkl",
#    },
#    {
#        "DATASET_ENC": './data/eng_sentences/transpositionCipherArr-encryptedEngSeq.csv',
#        "DATASET_ORI": './data/eng_sentences/arr-decryptedEngSeq.csv',
#        "RESULTS_FILE": "./results/eng_sentences/transpositionCipher.pkl",
#    },
#    {
#        "DATASET_ENC": './data/eng_sentences/productCipherArr-encryptedEngSeq.csv',
#        "DATASET_ORI": './data/eng_sentences/arr-decryptedEngSeq.csv',
#        "RESULTS_FILE": "./results/eng_sentences/productCipher.pkl",
#    },
]

def build_model(noise_std, num_classes, seq_len):
    embedding_dim = 64
    hidden_dim = 200
    padding_idx = 0
    learning_rate = 1e-3 # lr
    weight_decay = 1e-2

    model = AutoEncoder_model(
        num_classes=num_classes,
        seq_len=seq_len,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        noise_std=noise_std,
        padding_idx=padding_idx,
    )

    lossFunc = nn.CrossEntropyLoss(ignore_index=padding_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return {
        "model": model,
        "lossFunc": lossFunc,
        "optimizer": optimizer,
    }


for j, d in enumerate(DATASETS):
    torch.manual_seed(1234)
    random.seed(1234)

    train, validation, test = load_dataset(d.get("DATASET_ORI"), d.get("DATASET_ENC"))
    X_tr, Y_tr = train
    X_vl, Y_vl = validation
    X_ts, Y_ts = test

    num_classes = len(torch.unique(X_tr))#len(alphabet) + 1
    seq_len = X_tr.shape[1]

    results = load_results(d.get("RESULTS_FILE"))
    assert len(results) == len(noise_std)

    for i, s in enumerate(noise_std):

        build = build_model(s, num_classes, seq_len)
        m = build.get("model")
        weights = results[i].get("weights", None)
        assert weights is not None

        m.load_state_dict(weights)
        m.to(device)
         
        tr_row_acc = get_accuracy(m, X_tr, Y_tr, axis=1, sort=True)
        tr_col_acc = get_accuracy(m, X_tr, Y_tr, axis=0)
        plot_acc_line(1-tr_row_acc, 1-tr_col_acc, output_filename=f"results/pictures/auto_encoder-tr_{d.get('NAME', j)}-sd_{i}.png", subtitle="Row and Column Miss-classification", metric_type="Miss-classification")
        
        vl_row_acc = get_accuracy(m, X_vl, Y_vl, axis=1, sort=True)
        vl_col_acc = get_accuracy(m, X_vl, Y_vl, axis=0)      
        plot_acc_line(1-vl_row_acc, 1-vl_col_acc, output_filename=f"results/pictures/auto_encoder-vl_{d.get('NAME', j)}-sd_{i}", subtitle="Row and Column Miss-classification", metric_type="Miss-classification")
        
        ts_row_acc = get_accuracy(m, X_ts, Y_ts, axis=1, sort=True)
        ts_col_acc = get_accuracy(m, X_ts, Y_ts, axis=0)
        plot_acc_line(1-ts_row_acc, 1-ts_col_acc, output_filename=f"results/pictures/auto_encoder-ts_{d.get('NAME', j)}-sd_{i}", subtitle="Row and Column Miss-classification", metric_type="Miss-classification")
        

