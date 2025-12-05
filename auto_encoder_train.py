import numpy as np
import argparse

import torch
from torch.utils.data import TensorDataset, DataLoader

from mango import scheduler, Tuner

import pickle
import random
import uuid

from model import build_auto_encoder as build_model, AutoEncoder_factory, get_loss
from datasets import load_dataset

import string
import time
import os

parser = argparse.ArgumentParser(
    description="Script that trains a specified model with defined training configuration."
)
parser.add_argument(
    "-m",
    "--model_version",
    type=str,
    choices=[
        "one_hot",
        "simple",
        "embedding",
        "reduced_embedding",
        "reduced_by_char_embedding",
        "by_char_embedding",
        "lstm"
    ],
    required=True,
    help="Specify the model type. Must be one of: 'reduced' or 'simple'."
)
parser.add_argument(
    "--input_noise",
    type=float,
    default=0.0,
    help="Specify the percentage of noise perturbation to add to inputs (0-1)."
)
parser.add_argument(
    "--device",
    type=int,
    default=0,
    help="Specify the device number (0–4)."
)
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    choices=[
        "eng_sentences",
        "random"
    ],
    required=True
)
args = parser.parse_args()


if torch.cuda.is_available():
    dev = f"cuda:{args.device}"
    print('Running on CUDA')
else: 
    dev = "cpu"
    print('Running on CPU')
device = torch.device(dev)

#alphabet = string.printable
alphabet = string.ascii_lowercase + " "
model_version = args.model_version
input_noise_p = args.input_noise
dataset = args.dataset

num_classes = 0
seq_len = 0
num_epochs = 5_00
train_id = uuid.uuid4().hex

datasets_map = dict(
    eng_sentences = [
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
    ],
    random = [
        {
            "DATASET_ENC": './data/random/substitutionCipherArr-encryptedRandomAsciiCharSeq.csv',
            "DATASET_ORI": './data/random/arr-decryptedRandomAsciiCharSeq.csv',
            "RESULTS_DIR": "./results/random/",
            "NAME": "random-substitution",
        },
        {
            "DATASET_ENC": './data/random/transpositionCipherArr-encryptedRandomAsciiCharSeq.csv',
            "DATASET_ORI": './data/random/arr-decryptedRandomAsciiCharSeq.csv',
            "RESULTS_DIR": "./results/random/",
            "NAME": "random-transposition",
        },
        {
            "DATASET_ENC": './data/random/productCipherArr-encryptedRandomAsciiCharSeq.csv',
            "DATASET_ORI": './data/random/arr-decryptedRandomAsciiCharSeq.csv',
            "RESULTS_DIR": "./results/random/",
            "NAME": "random-product",
        },
    ]
)

DATASETS = datasets_map.get(dataset)
assert DATASETS

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

@scheduler.serial
def objective(
    **build_kwargs
):
    config = build_model(
        model_version,
        num_classes, seq_len,
        **build_kwargs
    )
    m = config.get("model")
    l = config.get("lossFunc")
    o = config.get("optimizer")
    
    r = train_model(
        m, l, o,
        X_tr, Y_tr, X_vl, Y_vl,
        num_epochs, verbose=True
    )
    loss = r.get("val_loss")

    return np.min(loss) # type: ignore


for d in DATASETS:
    torch.manual_seed(1234)
    random.seed(1234)

    train, validation, _ = load_dataset(d.get("DATASET_ORI"), d.get("DATASET_ENC"), device, input_noise_p)
    X_tr, Y_tr = train
    X_vl, Y_vl = validation

    # num_classes = len(torch.unique(X_tr))
    num_classes = len(alphabet) + 1
    seq_len = X_tr.shape[1]

    # Search for hyper-parameters
    start = time.time()
    conf_dict = dict(num_iteration=40, domain_size=5000, initial_random=5)
    param_space = AutoEncoder_factory.get_hyper_param_space(model_version)
    tuner = Tuner(param_space, objective, conf_dict)
    results = tuner.minimize()        
    end = time.time()
    print(f"training completed in {end - start}")
    
    # Train the model and save results with best found parameters 
    params = results["best_params"]
    config = build_model(
        model_version,
        num_classes, seq_len, **params
    )
    m = config.get("model")
    l = config.get("lossFunc")
    o = config.get("optimizer")
    training_results = train_model(
        m, l, o,
        X_tr, Y_tr, X_vl, Y_vl,
        num_epochs, verbose=True
    ) | {"hyper-params": params}

    # Write weights and training results file
    results_dir = d.get("RESULTS_DIR", "")
    dataset_name = d.get("NAME")
    results_file = results_dir + f"{model_version}/"  + dataset_name + "-autoencoder-" + train_id + ".pkl" # type: ignore
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "wb") as file:
        pickle.dump(training_results, file)
        print(f"wrote results file {results_file}")

with open("training_log", "a") as file:
    file.write(f"{train_id}: model={model_version}; input_noise={input_noise_p}\n")
