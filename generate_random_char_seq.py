import substitution_cipher
import transposition_cipher
import random
import string
from utils import encrypt_sequences, write_sequence, \
    write_key, seq_to_numpy, write_sequence_arr

ECN_FILE_NAME = "encryptedRandomAsciiCharSeq.csv"
DEC_FILE_NAME = "decryptedRandomAsciiCharSeq.csv"
KEY_FILE_NAME = "keyAscii10k.txt"

#alphabet = string.printable
alphabet = string.ascii_lowercase + " "
min_seq_len = 3
max_seq_len = 73
n = 10000

def generate_random_data():
    sequences = []
    options = range(len(alphabet))
    
    for _ in range(n):
        length = random.randint(min_seq_len, max_seq_len)
        line = random.choices(options, k = length)
        sequences.append("".join([alphabet[i] for i in line]))

    return sequences    

if __name__ == "__main__":
    random.seed(1234)

    decrypted_data = generate_random_data()
    decrypted_data_arr = seq_to_numpy(decrypted_data, alphabet)

    substitution_key = substitution_cipher.generate_key(alphabet)
    transposition_key = transposition_cipher.generate_key(1, min_seq_len)

    substitution_enc = encrypt_sequences(
        decrypted_data,
        substitution_cipher.encrypt,
        key=substitution_key,
        alphabet=alphabet,
    )
    substitution_enc_arr = seq_to_numpy(substitution_enc, alphabet)

    transposition_enc = encrypt_sequences(
        decrypted_data,
        transposition_cipher.encrypt, 
        key=transposition_key
    )
    transposition_enc_arr = seq_to_numpy(transposition_enc, alphabet)

    product_enc = encrypt_sequences(
        substitution_enc,
        transposition_cipher.encrypt,
        key=transposition_key
    )
    product_enc_arr = seq_to_numpy(product_enc, alphabet)

    write_sequence(decrypted_data, DEC_FILE_NAME)
    write_sequence_arr(decrypted_data_arr, "arr-" + DEC_FILE_NAME)
    
    write_sequence(substitution_enc, "substitutionCipher-" + ECN_FILE_NAME)
    write_sequence_arr(substitution_enc_arr, "substitutionCipherArr-" + ECN_FILE_NAME)

    write_sequence(transposition_enc, "transpositionCipher-" + ECN_FILE_NAME)
    write_sequence_arr(transposition_enc_arr, "transpositionCipherArr-" + ECN_FILE_NAME)
    
    write_sequence(product_enc, "productCipher-" + ECN_FILE_NAME)
    write_sequence_arr(product_enc_arr, "productCipherArr-" + ECN_FILE_NAME)

    write_key(substitution_key, "substitution-" + KEY_FILE_NAME)
    write_key(transposition_key, "transposition-" + KEY_FILE_NAME)
