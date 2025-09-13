import csv
import torch
import numpy as np
import string
import pickle
import matplotlib.pyplot as plt


DATA_DIR = "./data/"

def seq_to_numpy(sequences, alphabet=string.printable):
    max_seq_len = 0
    for s in sequences:
        max_seq_len = max(len(s), max_seq_len)

    alphabet_map = dict(zip(alphabet, range(1, len(alphabet) + 1)))
    arr = np.zeros((len(sequences), max_seq_len))
    for i, s in enumerate(sequences):
        arr[i, 0:len(s)] = np.array([alphabet_map.get(x) for x in s])

    return arr

def encrypt_sequences(sequences, encrypt, **kwargs):
    encrypted_sequences = []
    
    for s in sequences:
        encrypted_sequences.append(encrypt(s, **kwargs))

    return encrypted_sequences

def write_sequence(sequences, file_name):
    with open(DATA_DIR+file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        
        for item in sequences:
            writer.writerow([item])

def write_sequence_arr(arr, file_name):
    np.savetxt(DATA_DIR+file_name, arr, delimiter=',')

def write_key(key, file_name):
    with open(DATA_DIR+file_name, 'w') as file:
        file.write(repr(key))

def load_results(file_name):
    with open(file_name, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


################################################################
# Analysis
################################################################

def _predictions(pred, axis=2):
    return np.argmax(pred, axis)

def _rm_padding(pred, y):
    # remove padded values
    is_padding = y == 0
    pred[is_padding] = 0
    return pred, is_padding

def _get_predictions(model, x):
    model.eval()
    with torch.no_grad():
        output = model(x)
        output = output.cpu().numpy()

    return _predictions(output)

def _correctly_classified(model, x, y):
    y = y.squeeze().cpu().numpy()
    pred = _get_predictions(model, x)
    pred, is_padding = _rm_padding(pred, y)
    correct = pred == y
    correct[is_padding] = 1
    return correct, is_padding

def get_accuracy(model, x, y, axis=None, sort=False):
    correctly_classified, is_padding = _correctly_classified(model, x, y)
    valid_mask = ~is_padding

    # Count number of correctly classified non-padding elements
    masked_correct = correctly_classified * valid_mask
    valid_count = np.sum(valid_mask, axis=axis)
    total_correctly_classified = np.sum(masked_correct, axis=axis)

    if sort:
        # Sorting by ascending sequence length (descending padding)
        padding_count = np.sum(is_padding, axis=axis)
        order_index = np.flip(np.argsort(padding_count))
        total_correctly_classified = total_correctly_classified[order_index]
        valid_count = valid_count[order_index]

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        accuracy = np.true_divide(total_correctly_classified, valid_count)
        accuracy = np.nan_to_num(accuracy)  # replace NaNs with 0

    return accuracy


def plot_acc_line(row_acc, col_acc, output_filename=None, subtitle="Row and Column Accuracy", metric_type="Accuracy"):
    fig, axes = plt.subplots(2, 1, figsize=(12, 5))
    
    # Calculate the trend line (linear regression)
    x = np.arange(row_acc.size)
    z = np.polyfit(x, row_acc, 1) # 1 for linear, 2 for quadratic, etc.
    p = np.poly1d(z)
    
    axes[0].plot(x, row_acc)
    axes[0].plot(x, p(x), "r--", label='Trend Line')
    axes[0].set_title(f"Row {metric_type}")
    axes[0].set_xlabel("Row")
    axes[0].set_ylabel(metric_type)
    
    # Calculate the trend line (linear regression)
    x = np.arange(col_acc.size)
    z = np.polyfit(x, col_acc, 1) # 1 for linear, 2 for quadratic, etc.
    p = np.poly1d(z)
    
    axes[1].plot(x, col_acc)
    axes[1].plot(x, p(x), "r--", label='Trend Line')
    axes[1].set_title(f"Column {metric_type}")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel(metric_type)
    
    plt.suptitle(subtitle, fontsize=16)
    plt.tight_layout()
    if output_filename is not None:
        plt.savefig(output_filename)
        plt.close()
    else: plt.show()

