import mindspore.nn as nn
from mindspore.nn.layer import activation
import mindspore.ops as P

class RNNEncoder(nn.Cell):
    def __init__(self, input_size, hidden_size, bidirectional, num_layers, dropout=0.0, embeddings=None, use_bridge=False):
        super().__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.bidirectional = bidirectional
        self.embeddings = embeddings
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self.total_hidden_dim = hidden_size * num_layers
            self.bridge = nn.Dense(self.total_hidden_dim, self.total_hidden_dim, activation='relu')
    
    def construct(self, inputs, seq_length):
        emb = self.embeddings(inputs)
        memory_bank, encoder_final = self.rnn(emb, seq_length=seq_length)

        if self.use_bridge:
            shape = encoder_final.shape
            encoder_final = self.bridge(encoder_final.view(-1, self.total_hidden_dim)).view(shape)
        
        if self.bidirectional:
            batch_size = encoder_final.shape[1]
            encoder_final = encoder_final.view(self.num_layers, 2, batch_size, self.hidden_size) \
                .swapaxes(1, 2).view(self.num_layers, batch_size, self.hidden_size * 2)
        
        return encoder_final, memory_bank
