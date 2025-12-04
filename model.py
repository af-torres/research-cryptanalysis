import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mango.domain.distribution import loguniform
from scipy.stats import uniform

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

class AutoEncoderSimple_model(nn.Module):
    def __init__(
        self, num_classes, seq_len,
        hidden_dim=64,
        noise_std=0.1, dropout=0.2, padding_idx=0,
    ):
        super(AutoEncoderSimple_model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.Dropout(p=dropout), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.noise = HadamardNoise(noise_std=noise_std)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout), 
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * num_classes) 
        )

        self.num_classes = num_classes
        self.seq_len = seq_len

    def forward(self, x):
        encoded = self.encoder(x.float())
    
        noised = self.noise(encoded) # this works as a regularizer
       
        decoded = self.decoder(noised)
        decoded = decoded.view(-1, self.seq_len, self.num_classes)
        
        return decoded

class AutoEncoderEmbeddingReduced_model(nn.Module):
    def __init__(
        self, num_classes, seq_len,
        embedding_dim=120,
        hidden_dim=64,
        noise_std=0.1, dropout=0.2,
        padding_idx=0,
    ):
        super(AutoEncoderEmbeddingReduced_model, self).__init__()

        self.embedding = nn.Embedding(num_classes, embedding_dim, padding_idx)
        self.encoder = nn.Sequential(
            nn.Linear(seq_len * embedding_dim, hidden_dim),
            nn.Dropout(p=dropout), 
            nn.ReLU(),
        )

        self.noise = HadamardNoise(noise_std=noise_std)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, seq_len * num_classes),
            nn.Dropout(p=dropout), 
            nn.ReLU(),
        )

        self.num_classes = num_classes
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

    def forward(self, x):
        embedded = self.embedding(x)
        encoded = self.encoder(embedded.view((-1, self.seq_len * self.embedding_dim)))
    
        noised = self.noise(encoded) # this works as a regularizer
       
        decoded = self.decoder(noised)
        decoded = decoded.view(-1, self.seq_len, self.num_classes)
        
        return decoded

class AutoEncoderEmbeddingReducedByChar_model(nn.Module):
    def __init__(
        self, num_classes, seq_len,
        embedding_dim=120,
        hidden_dim=64,
        noise_std=0.1, dropout=0.2,
        padding_idx=0,
    ):
        super(AutoEncoderEmbeddingReducedByChar_model, self).__init__()

        self.embedding = nn.Embedding(num_classes, embedding_dim, padding_idx)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Dropout(p=dropout), 
            nn.ReLU(),
        )

        self.noise = HadamardNoise(noise_std=noise_std)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.Dropout(p=dropout), 
            nn.ReLU(),
        )

        self.num_classes = num_classes
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

    def forward(self, x):
        embedded = self.embedding(x)
        encoded = self.encoder(embedded.view((-1, self.embedding_dim)))
    
        noised = self.noise(encoded) # this works as a regularizer
       
        decoded = self.decoder(noised)
        decoded = decoded.view(-1, self.seq_len, self.num_classes)
        
        return decoded

class AutoEncoderEmbeddingByChar_model(nn.Module):
    def __init__(
        self, num_classes, seq_len,
        embedding_dim=120,
        hidden_dim=64,
        noise_std=0.1, dropout=0.2,
        padding_idx=0,
    ):
        super(AutoEncoderEmbeddingByChar_model, self).__init__()

        self.embedding = nn.Embedding(num_classes, embedding_dim, padding_idx)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Dropout(p=dropout), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.noise = HadamardNoise(noise_std=noise_std)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout), 
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes) 
        )

        self.num_classes = num_classes
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

    def forward(self, x):
        embedded = self.embedding(x)
        encoded = self.encoder(embedded.view((-1, self.embedding_dim)))
    
        noised = self.noise(encoded) # this works as a regularizer
       
        decoded = self.decoder(noised)
        decoded = decoded.view(-1, self.seq_len, self.num_classes)
        
        return decoded

class AutoEncoderEmbedding_model(nn.Module):
    def __init__(
        self, num_classes, seq_len,
        embedding_dim=120,
        hidden_dim=64,
        noise_std=0.1, dropout=0.2,
        padding_idx=0,
    ):
        super(AutoEncoderEmbedding_model, self).__init__()

        self.embedding = nn.Embedding(num_classes, embedding_dim, padding_idx)
        self.encoder = nn.Sequential(
            nn.Linear(seq_len * embedding_dim, hidden_dim),
            nn.Dropout(p=dropout), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.noise = HadamardNoise(noise_std=noise_std)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout), 
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * num_classes) 
        )

        self.num_classes = num_classes
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

    def forward(self, x):
        embedded = self.embedding(x)
        encoded = self.encoder(embedded.view((-1, self.seq_len * self.embedding_dim)))
    
        noised = self.noise(encoded) # this works as a regularizer
       
        decoded = self.decoder(noised)
        decoded = decoded.view(-1, self.seq_len, self.num_classes)
        
        return decoded

class AutoEncoderOneHot_model(nn.Module):
    def __init__(
        self, num_classes, seq_len,
        hidden_dim=64,
        noise_std=0.1, dropout=0.2, padding_idx=0,
    ):
        super(AutoEncoderOneHot_model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(seq_len * num_classes, hidden_dim), # (2336 -> 64)
            nn.Dropout(p=dropout), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.noise = HadamardNoise(noise_std=noise_std)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout), 
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * num_classes) # (64 -> 2190) (actual X)
        )

        self.num_classes = num_classes
        self.seq_len = seq_len

    def forward(self, x):
        one_hot = F.one_hot(x, num_classes=self.num_classes)
        one_hot = one_hot.view((len(x), -1)).float()
        encoded = self.encoder(one_hot)
        
        noised = self.noise(encoded) # this works as a regularizer
       
        decoded = self.decoder(noised)
        decoded = decoded.view(-1, self.seq_len, self.num_classes)
        
        return decoded

class LSTM_model(nn.Module):
    def __init__(
            self, num_classes, seq_len,
            hidden_dim, num_layers,
            noise_std=0.0,
            dropout=0.5,
            bidirectional=False,
            padding_idx=0
        ):
        super(LSTM_model, self).__init__()
        self.output_size = num_classes # n umber of classes (number of outputs)
        self.num_layers = num_layers # number of layers
        self.input_size = 1 # input size
        self.hidden_size = hidden_dim # hidden state
        
        D = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        self.noise = HadamardNoise(noise_std=noise_std)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size * D * seq_len, self.hidden_size * D * seq_len),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(self.hidden_size * D * seq_len, self.output_size * seq_len),
        )
        
    def forward(
            self, x: torch.Tensor, 
        ):
        device = x.device
        x = x.unsqueeze(2).to(dtype=torch.float32)
        
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) # hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) # internal state

        # Assigning hidden and intenal states to GPU
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        
        # Propagate input through LSTM
        out, (hn, cn) = self.lstm(x, (h_0, c_0)) # lstm with input, hidden, and internal state
        flatten = torch.flatten(out, start_dim=1)
        noised = self.noise(flatten)
       
        return self.head(noised).view((x.shape[0], x.shape[1], -1))

class AutoEncoder_factory:
    __versions = {
        "one_hot": AutoEncoderOneHot_model,
        "simple": AutoEncoderSimple_model,
        "embedding": AutoEncoderEmbedding_model,
        "reduced_embedding": AutoEncoderEmbeddingReduced_model,
        "reduced_by_char_embedding": AutoEncoderEmbeddingReducedByChar_model,
        "by_char_embedding": AutoEncoderEmbeddingByChar_model,
        "lstm": LSTM_model,
    } 

    __model_hyper_params = {
        "one_hot": dict(
            noise_std = uniform(0.35, 0.35),
            dropout = uniform(0.25, 0.25),
            hidden_dim = range(5, 1000),
        ),
        "simple": dict(
            noise_std = np.arange(0, 0.75, 0.05),
            dropout = np.arange(0, 0.75, 0.05),
            hidden_dim = range(5, 1000),
        ),
        "embedding": dict(
            noise_std = np.arange(0, 0.75, 0.05),
            dropout = np.arange(0, 0.75, 0.05),
            hidden_dim = range(5, 1000),
            embedding_dim = range(1, 500),
        ),
        "reduced_embedding": dict(
            noise_std = np.arange(0, 0.75, 0.05),
            dropout = np.arange(0, 0.75, 0.05),
            hidden_dim = range(5, 1000),
            embedding_dim = range(1, 500),
        ),
        "reduced_by_char_embedding": dict(
            noise_std = np.arange(0, 0.75, 0.05),
            dropout = np.arange(0, 0.75, 0.05),
            hidden_dim = range(5, 1000),
            embedding_dim = range(1, 500),
        ),
        "by_char_embedding": dict(
            noise_std = np.arange(0, 0.75, 0.05),
            dropout = np.arange(0, 0.75, 0.05),
            hidden_dim = range(5, 1000),
            embedding_dim = range(1, 500),
        ),
        "lstm": dict(
            noise_std = np.arange(0, 0.75, 0.05),
            dropout = np.arange(0, 0.75, 0.05),
            hidden_dim = range(5, 250),
            num_layers = range(1, 3)
        ),
    }

    __loss_hyper_params = {
        "one_hot": dict(
            lr = loguniform(-3, 2),
            weight_decay = loguniform(-2, 2),
        ),
        "simple": dict(
            lr = loguniform(-3, 2),
            weight_decay = loguniform(-2, 2),
        ),
        "embedding": dict(
            lr = loguniform(-3, 2),
            weight_decay = loguniform(-2, 2),
        ),
        "reduced_embedding": dict(
            lr = loguniform(-3, 2),
            weight_decay = loguniform(-2, 2),
        ),
        "default": dict(
            lr = loguniform(-3, 2),
            weight_decay = loguniform(-2, 2),
        ),
    }

    @classmethod
    def get_model(cls, version, *args, **kargs):
        print(f"building auto_encoder with args={args} and kwargs={kargs}")
        model = cls.__versions[version](*args, **kargs)
        return model

    @classmethod
    def get_hyper_param_space(cls, version):
        return cls.__model_hyper_params[version] | (cls.__loss_hyper_params["default"] if cls.__loss_hyper_params.get(version, "") == "" else cls.__loss_hyper_params[version])

    @classmethod
    def get_model_param_names(cls, version):
        return cls.__model_hyper_params[version].keys()

    @classmethod
    def get_loss_param_names(cls, version):
        if cls.__loss_hyper_params.get(version, "") == "": return cls.__loss_hyper_params["default"].keys()
        return cls.__loss_hyper_params[version].keys()

def get_config_subset_values(config, keys):
    values = [config[k] for k in keys]
    return dict(zip(keys, values))

def build_auto_encoder(
    model_version,
    num_classes, seq_len,
    padding_idx = 0,
    **config
):
    model_config = get_config_subset_values(
        config,
        AutoEncoder_factory.get_model_param_names(model_version)
    )
    model = AutoEncoder_factory.get_model(
        model_version,
        num_classes=num_classes,
        seq_len=seq_len,
        padding_idx=padding_idx,
        **model_config,
    )
    
    loss_config = get_config_subset_values(
        config,
        AutoEncoder_factory.get_loss_param_names(model_version)
    )
    lossFunc = nn.CrossEntropyLoss(ignore_index=padding_idx)
    optimizer = torch.optim.AdamW(model.parameters(), **loss_config)

    return {
        "model": model,
        "lossFunc": lossFunc,
        "optimizer": optimizer,
    }
