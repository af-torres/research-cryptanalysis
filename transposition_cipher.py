import random
import math

def generate_key(min_key = 5, max_key = 20):
    return random.randint(min_key, max_key)

def encrypt(plaintext, key):
    ciphertext = [''] * key

    for column in range(key):
        pointer = column
        while pointer < len(plaintext):
            ciphertext[column] += plaintext[pointer]
            pointer += key

    return ''.join(ciphertext)

def decrypt(ciphertext, key):
    num_columns = math.ceil(len(ciphertext) / key)
    num_rows = key
    num_shaded_boxes = (num_columns * num_rows) - len(ciphertext)

    plaintext = [''] * num_columns
    column = 0
    row = 0

    for symbol in ciphertext:
        plaintext[column] += symbol
        column += 1

        # Skip to next row if we hit end of column, or are at the shaded area
        if (column == num_columns) or (column == num_columns - 1 and row >= num_rows - num_shaded_boxes):
            column = 0
            row += 1

    return ''.join(plaintext)

# Example usage
if __name__ == '__main__':
    message = "WE ARE DISCOVERED FLEE AT ONCE"
    key = 5

    encrypted = encrypt(message, key)
    print("Encrypted:", encrypted)

    decrypted = decrypt(encrypted, key)
    print("Decrypted:", decrypted)
