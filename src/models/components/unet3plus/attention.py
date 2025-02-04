import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        N, seq_length, d_model = x.shape

        # Split the embedding into self.num_heads different pieces
        queries = self.query(x).view(N, seq_length, self.num_heads, self.head_dim)
        keys = self.key(x).view(N, seq_length, self.num_heads, self.head_dim)
        values = self.value(x).view(N, seq_length, self.num_heads, self.head_dim)

        # Transpose to get dimensions (N, num_heads, seq_length, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Scaled dot-product attention
        energy = torch.matmul(queries, keys.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attention = torch.softmax(energy, dim=-1)

        out = torch.matmul(attention, values)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(N, seq_length, d_model)

        # Final linear layer
        out = self.fc_out(out)

        return out