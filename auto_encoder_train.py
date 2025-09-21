import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
import pickle
import random
from model import AutoEncoder_model, get_loss
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

noise_std = np.array(range(0, 45, 5)) / 100
use_positional_enc = False
alphabet = string.printable

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
        "RESULTS_FILE": "./results/eng_sentences/",
        "NAME": "eng_sentences-product",
    },
]

def build_model(
    num_classes, seq_len,
    noise_std = 0.45,
    dropout = 0.5,
    embedding_dim = 500,
    hidden_dim = 150,
    learning_rate = 1e-3,
    weight_decay = 1e-2,
):
    padding_idx = 0

    model = AutoEncoder_model(
        num_classes=num_classes,
        seq_len=seq_len,
        embedding_dim=embedding_dim,
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

def train_model(
    model, lossFunc, optimizer,
    X_tr, Y_tr, X_vl, Y_vl,
    num_epochs, batch_size=250, verbose=False,
):
    print("training model...")
    
    train_dataset = TensorDataset(X_tr, Y_tr)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    weights = None # used to store best performing model weights (as of valdiation loss)
    train_loss = []
    val_loss = []

    model.to(device) # assigning model to GPU
    for epoch in range(num_epochs):
        model.train()
        for (x, y) in train_dataloader:
            optimizer.zero_grad() # reset gradients on each batch
            outputs = model(x) # forward pass

            # obtain the loss function
            _batch_size = outputs.shape[0] * outputs.shape[1]
            loss = lossFunc(
                outputs.view(_batch_size, -1), # same as (batch_size * seq_len, num_classes)
                y.view(-1) # same as (batch_size * seq_len)
            )
            loss.backward() # calculates the loss of the loss function
            optimizer.step() # improve from loss, i.e backprop

        # Compute loss over training and validation datasets
        epoch_train_loss = get_loss(model, lossFunc, X_tr, Y_tr)
        epoch_validation_loss = get_loss(model, lossFunc, X_vl, Y_vl)

        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_validation_loss)

        if weights is None or np.min(val_loss) == epoch_validation_loss:
            # this way of assigning the weights variable must be done to avoid a shallow copy of
            # the model as we are still updating it in the training loop
            weights = {k: v.clone().to(torch.device("cpu")) for k, v in model.state_dict().items()}

        if verbose and (epoch % 100 == 0 or epoch + 1 == num_epochs):
                print(f"Epoch: {epoch + 1}, loss: {epoch_train_loss:.5f}, val_loss: {epoch_validation_loss:.5f}")

    print(f"model performance:\nmin_loss: {np.min(train_loss):.5f}, min_val_loss: {np.min(val_loss):.5f}")
    return {
        "weights": weights,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }


num_epochs = 5_00

for d in DATASETS:
    torch.manual_seed(1234)
    random.seed(1234)

    train, validation, _ = load_dataset(d.get("DATASET_ORI"), d.get("DATASET_ENC"))
    X_tr, Y_tr = train
    X_vl, Y_vl = validation

    # num_classes = len(torch.unique(X_tr))
    num_classes = len(alphabet) + 1
    seq_len = X_tr.shape[1]

    models_setup = []
    for s in noise_std:
        m = build_model(s, num_classes, seq_len)
        models_setup.append(m)

    start = time.time()
    last = start
    training_results = []
    for config in models_setup:
        m = config.get("model")
        l = config.get("lossFunc")
        o = config.get("optimizer")
        
        r = train_model(
            m, l, o,
            X_tr, Y_tr, X_vl, Y_vl,
            num_epochs, verbose=True
        )
        training_results.append(r)
        
        end = time.time()
        print(f"training completed in {end - last}")
        last = end
        
    end = time.time()
    print(f"all models for one dataset completed in {end - start}")

    # write weights and training results file
    results_dir = d.get("RESULTS_DIR", "")
    dataset_name = d.get("NAME")
    results_file = results_dir + dataset_name
    with open(results_file, "wb") as file:
        pickle.dump(training_results, file)
        print(f"wrote results file {results_file}")

