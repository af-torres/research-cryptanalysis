import csv
import numpy as np
import string
import pickle

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
