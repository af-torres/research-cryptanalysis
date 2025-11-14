import pandas as pd
import random
import torch
import numpy as np

def add_input_noise(X, input_noise=0, padding_val=0):
    replace = np.random.binomial(1, input_noise, size=X.shape) & (X != 0)
    unique = np.unique(X)
    
    random_samples = np.random.choice(unique, size=X.shape, replace=True)
    X_noised = np.where(replace, random_samples, X)
    return X_noised

def load_dataset(
    decrypted_arr_file, encrypted_arr_file, 
    device="cuda" if torch.cuda.is_available() else "cpu",
    input_noise=0
):
    X = pd.read_csv(encrypted_arr_file, header=None).to_numpy()
    X = add_input_noise(X, input_noise)
    Y = pd.read_csv(decrypted_arr_file, header=None).to_numpy()
    n = X.shape[0]

    reorder_idx = random.sample(range(0, n), n)
    X = X[reorder_idx]
    Y = Y[reorder_idx]

    p_tr = .8
    p_vl = .1

    n_tr = int(p_tr * n)
    n_vl = int(p_vl * n)
    n_ts = n - (n_tr + n_vl)

    # Training Data
    X_tr = X[0:n_tr, :]
    X_tr_ten = torch.tensor(X_tr, dtype=torch.long)
    X_tr_ten = X_tr_ten.to(device) # assining the tensor to GPU

    Y_tr = Y[0:n_tr, :]
    Y_tr_ten = torch.tensor(Y_tr, dtype=torch.long)
    Y_tr_ten = Y_tr_ten.to(device) # assining the tensor to GPU

    # Validation Data
    X_vl = X[n_tr:n_tr + n_vl, :]
    X_vl_ten = torch.tensor(X_vl, dtype=torch.long)
    X_vl_ten = X_vl_ten.to(device) # assining the tensor to GPU

    Y_vl = Y[n_tr:n_tr + n_vl, :]
    Y_vl_ten = torch.tensor(Y_vl, dtype=torch.long)
    Y_vl_ten = Y_vl_ten.to(device) # assining the tensor to GPU

    # Test Data
    X_ts = X[n_tr + n_vl:, :]
    X_ts_ten = torch.tensor(X_ts, dtype=torch.long)
    X_ts_ten = X_ts_ten.to(device) # assining the tensor to GPU

    Y_ts = Y[n_tr + n_vl:, :]
    Y_ts_ten = torch.tensor(Y_ts, dtype=torch.long)
    Y_ts_ten = Y_ts_ten.to(device) # assining the tensor to GPU

    # Sanity check to ensure data is well distributed and we get samples of every character during training
    unique_x_tr = np.unique(X_tr)
    unique_x_vl = np.unique(X_vl)
    unique_x_ts = np.unique(X_ts)

    print("characters not in tr but in vl", [x for x in unique_x_vl if x not in unique_x_tr])
    print("characters not in tr but in ts", [x for x in unique_x_ts if x not in unique_x_tr])

    return (X_tr_ten, Y_tr_ten), (X_vl_ten, Y_vl_ten), (X_ts_ten, Y_ts_ten)
