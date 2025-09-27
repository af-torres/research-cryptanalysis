import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
import random
from model import AutoEncoder_factory, get_loss
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

alphabet = string.printable
model_version = "simple"
train_id = "0324bc21b65b4472985be86cf5a0f56d"

DATASETS = [
#    {
#        "DATASET_ENC": '../cryptanalysis_old/datasets/dtEnc10k.csv',
#        "DATASET_ORI": '../cryptanalysis_old/datasets/dtOri10k.csv',
#        "RESULTS_FILE": "./results/random/substitutionCipher-old-data.pkl",
#        "NAME": "OLD-DATASET",
#    },
    {
        "DATASET_ENC": './data/random/substitutionCipherArr-encryptedRandomCharSeq.csv',
        "DATASET_ORI": './data/random/arr-decryptedRandomCharSeq.csv',
        "RESULTS_DIR": "./results/random/",
        "NAME": "random-substitution",
    },
    {
        "DATASET_ENC": './data/random/transpositionCipherArr-encryptedRandomCharSeq.csv',
        "DATASET_ORI": './data/random/arr-decryptedRandomCharSeq.csv',
        "RESULTS_DIR": "./results/random/",
        "NAME": "random-transposition",
    },
    {
        "DATASET_ENC": './data/random/productCipherArr-encryptedRandomCharSeq.csv',
        "DATASET_ORI": './data/random/arr-decryptedRandomCharSeq.csv',
        "RESULTS_DIR": "./results/random/",
        "NAME": "random-product",
    },
    {
        "DATASET_ENC": './data/eng_sentences/substitutionCipherArr-encryptedEngSeq.csv',
        "DATASET_ORI": './data/eng_sentences/arr-decryptedEngSeq.csv',
        "RESULTS_DIR": "./results/eng_sentences/",
        "NAME": "eng_sentences-substitution",
    },
    {
        "DATASET_ENC": './data/eng_sentences/transpositionCipherArr-encryptedEngSeq.csv',
        "DATASET_ORI": './data/eng_sentences/arr-decryptedEngSeq.csv',
        "RESULTS_DIR": "./results/eng_sentences/",
        "NAME": "eng_sentences-transposition",
    },
    {
        "DATASET_ENC": './data/eng_sentences/productCipherArr-encryptedEngSeq.csv',
        "DATASET_ORI": './data/eng_sentences/arr-decryptedEngSeq.csv',
        "RESULTS_DIR": "./results/eng_sentences/",
        "NAME": "eng_sentences-product",
    },
]

def build_model(
    num_classes, seq_len,
    noise_std = 0.45,
    dropout = 0.5,
    hidden_dim = 150,
    learning_rate = 1e-3,
    weight_decay = 1e-2,
):
    padding_idx = 0

    model = AutoEncoder_factory.get_model(
        model_version,
        num_classes=num_classes,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        noise_std=noise_std,
        dropout=dropout,
        padding_idx=padding_idx,
    )

    lossFunc = nn.CrossEntropyLoss(ignore_index=padding_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return {
        "model": model,
        "lossFunc": lossFunc,
        "optimizer": optimizer,
    }


for i, d in enumerate(DATASETS):
    torch.manual_seed(1234)
    random.seed(1234)

    train, validation, test = load_dataset(d.get("DATASET_ORI"), d.get("DATASET_ENC"))
    X_tr, Y_tr = train
    X_vl, Y_vl = validation
    X_ts, Y_ts = test

    # num_classes = len(torch.unique(X_tr))
    num_classes = len(alphabet) + 1

    seq_len = X_tr.shape[1]
    
    results_dir = d.get("RESULTS_DIR", "")
    dataset_name = d.get("NAME")
    results_file = results_dir + f"{model_version}/"  + dataset_name + "-autoencoder-" + train_id + ".pkl"
    results = load_results(results_file)
    hyper_params = results.get("hyper-params")
    assert hyper_params is not None
    weights = results.get("weights", None)
    assert weights is not None

    with open(f"results/reports/auto_encoder-{dataset_name}-{train_id}.md", "w") as report_file:
        build = build_model(
            num_classes, seq_len,
            **hyper_params
        )
        m = build.get("model")

        m.load_state_dict(weights)
        m.to(device)
         
        tr_row_acc = get_accuracy(m, X_tr, Y_tr, axis=1, sort=True)
        tr_col_acc = get_accuracy(m, X_tr, Y_tr, axis=0)
        plot_acc_line(1-tr_row_acc, 1-tr_col_acc,
            output_filename=f"results/pictures/auto_encoder_{model_version}-tr_{dataset_name}-{train_id}",
            subtitle="Row and Column Miss-classification", metric_type="Miss-classification")
        
        vl_row_acc = get_accuracy(m, X_vl, Y_vl, axis=1, sort=True)
        vl_col_acc = get_accuracy(m, X_vl, Y_vl, axis=0)      
        plot_acc_line(1-vl_row_acc, 1-vl_col_acc,
            output_filename=f"results/pictures/auto_encoder_{model_version}-vl_{dataset_name}-{train_id}",
            subtitle="Row and Column Miss-classification", metric_type="Miss-classification")
        
        ts_row_acc = get_accuracy(m, X_ts, Y_ts, axis=1, sort=True)
        ts_col_acc = get_accuracy(m, X_ts, Y_ts, axis=0)
        plot_acc_line(1-ts_row_acc, 1-ts_col_acc, 
            output_filename=f"results/pictures/auto_encoder_{model_version}-ts_{dataset_name}-{train_id}", 
            subtitle="Row and Column Miss-classification", metric_type="Miss-classification")
        
        tr_row_acc = get_accuracy(m, X_tr, Y_tr)
        vl_row_acc = get_accuracy(m, X_vl, Y_vl)
        ts_row_acc = get_accuracy(m, X_ts, Y_ts)
        
        report_line = f"Accuracy acchieved:\ntraining: {tr_row_acc}\nvalidation: {vl_row_acc}\ntesting: {ts_row_acc}"
        report_line += "\n#####################################\n"
        report_file.write(report_line)

