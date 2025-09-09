import string
import random

def generate_key(alphabet = string.printable):
    k = len(alphabet)
    order = random.sample(range(k), k = k)

    return ''.join([alphabet[i] for i in order])

def encrypt(plaintext, key, alphabet=string.printable):
    table = str.maketrans(alphabet, key)
    return plaintext.translate(table)

def decrypt(ciphertext, key, alphabet=string.printable):
    table = str.maketrans(key, alphabet)
    return ciphertext.translate(table)

# Example usage:
if __name__ == "__main__":
    key = generate_key()
    plaintext = "hello world"
    ciphertext = encrypt(plaintext, key)
    decrypted = decrypt(ciphertext, key)

    print(f"Key:        {key}")
    print(f"Plaintext:  {plaintext}")
    print(f"Ciphertext: {ciphertext}")
    print(f"Decrypted:  {decrypted}")
