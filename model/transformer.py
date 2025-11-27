import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    pass


class EncoderLayer(nn.Module):
    pass


class DecoderLayer(nn.Module):
    pass


class MultiHeadAttention(nn.Module):
    
    def __init__(self, input_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.linear_q = nn.Linear(input_dim, input_dim)
        self.linear_k = nn.Linear(input_dim, input_dim)
        self.linear_v = nn.Linear(input_dim, input_dim)

    def forward(self, q, k, v, mask=None):
        q = self.linear_q(q) # (batch_size, seq_len, input_dim)
        k = self.linear_k(k)
        v = self.linear_v(v)
        
        q = q.view(q.size(0), q.size(1), self.n_heads, q.size(-1) // self.n_heads).transpose(1, 2) # (batch_size, n_heads, seq_len, head_dim)
        k = k.view(k.size(0), k.size(1), self.n_heads, k.size(-1) // self.n_heads).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.n_heads, v.size(-1) // self.n_heads).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)/ (k.size(-1) ** 0.5)) # (batch_size, n_heads, seq_len, seq_len)
        
        if mask is not None:
            mask = torch.triu(torch.ones(attn.size(-2), attn.size(-1)), 1).to(attn.device)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)
        output = torch.matmul(attn, v) # (batch_size, n_heads, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous()
        output = output.view(output.size(0), output.size(1), -1) # (batch_size, seq_len, input_dim)

        return output

class PositionalEncoding(nn.Module):
    pass


class FeedForwardNetwork(nn.Module):
    pass