import torch
import numpy as np
import torch.nn as nn

def get_loss(model, lossFunc, x, y):
    model.eval() # deactivate things like dropout and other training configurations
    with torch.no_grad():
        outputs = model(x)
        y = y.reshape(-1)
        loss = lossFunc(
            outputs.view(-1, outputs.size(2)),
            y
        )
        return loss.item()

def get_positional_encoding(sequence_length, embedding_size):
    """
    Generates sinusoidal positional encodings.

    Args:
        sequence_length (int): Length of the input sequence.
        embedding_size (int): Dimensionality of embeddings (d_model).

    Returns:
        numpy.ndarray of shape (sequence_length, embedding_size)
    """
    position = np.arange(sequence_length)[:, np.newaxis]         # Shape: (seq_len, 1)
    div_term = np.exp(np.arange(0, embedding_size, 2) * (-np.log(10000.0) / embedding_size))  # Shape: (embedding_size/2,)

    pe = np.zeros((sequence_length, embedding_size))
    pe[:, 0::2] = np.sin(position * div_term)  # Even indices
    pe[:, 1::2] = np.cos(position * div_term)  # Odd indices

    return torch.Tensor(pe)

class HadamardNoise(nn.Module):
    def __init__(self, noise_std=0.1):
        super(HadamardNoise, self).__init__()
        self.noise_std = noise_std

    def forward(self, x):
        if self.training:  # only apply noise during training
            noise = torch.randn_like(x) * self.noise_std
            return x * (1 + noise)
        else:
            return x  # no noise during evaluation

class AutoEncoder_model(nn.Module):
    def __init__(
        self, num_classes,
        seq_len=150, embedding_dim=32, hidden_dim=64,
        noise_std=0.1, padding_idx=0,
        use_positional_enc=False
    ):
        super(AutoEncoder_model, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim, padding_idx=padding_idx)

        self.encoder = nn.Sequential(
            nn.Linear(seq_len * embedding_dim, hidden_dim), # (2336 -> 64)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.noise = HadamardNoise(noise_std=noise_std)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * num_classes) # (64 -> 2190) (actual X)
        )

        self.num_classes = num_classes
        self.seq_len = seq_len

        pos = None
        if use_positional_enc:
            pos = get_positional_encoding(seq_len, embedding_dim)
        else:
            pos = torch.zeros((seq_len, embedding_dim))
        self.register_buffer("positional_encoding", pos)


    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded + self.positional_encoding
        embedded = embedded.view(x.size(0), -1)
        encoded = self.encoder(embedded)

        noised = self.noise(encoded) # this works as a regulizer
        decoded = self.decoder(noised)
        decoded = decoded.view(-1, self.seq_len, self.num_classes)

        return decoded

class LSTM_model(nn.Module):
    def __init__(self, output_size, 
                 input_size, hidden_size, num_layers,
                 noise_std=0.0,
                 bidirectional=False):
        super(LSTM_model, self).__init__()
        self.output_size = output_size # n umber of classes (number of outputs)
        self.num_layers = num_layers # number of layers
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # hidden state
        
        D = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=bidirectional
        ) # lstm
        self.noise = HadamardNoise(noise_std=noise_std)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * D, hidden_size * D),
            nn.ReLU(),
            nn.Linear(hidden_size * D, output_size),
        )
        
    def forward(self, x):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) # hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) # internal state

        # Assigning hidden and intenal states to GPU
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        
        # Propagate input through LSTM
        out, (hn, cn) = self.lstm(x, (h_0, c_0)) # lstm with input, hidden, and internal state
        noised = self.noise(out)
        
        return self.head(noised) # Final Output
    