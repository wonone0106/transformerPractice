'''
    파일명 : transformer.py
    설명 : PyTorch로 Transformer 모델 구현
    작성자 : 박승일
    작성일 : 2025-11-27
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, src_dim, tgt_dim, embed_dim, n_heads, num_encoder_layers, num_decoder_layers):
        super().__init__()
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.embed_src = nn.Embedding(src_dim, embed_dim)
        self.embed_tgt = nn.Embedding(tgt_dim, embed_dim)
        pass

class EncoderLayer(nn.Module):
    pass


class DecoderLayer(nn.Module):
    pass


class MultiHeadAttention(nn.Module):
    """
    다중 헤드 어텐션 메커니즘 구현
    """
    
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, mask=None):
        q = self.linear_q(q) # (batch_size, seq_len, embed_dim)
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
        output = output.view(output.size(0), output.size(1), -1) # (batch_size, seq_len, embed_dim)

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsequeeze(1).float()
        div_term = torch.arange(0, embed_dim, 2).float() ** (torch.tensor(10000.0) / embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class FeedForwardNetwork(nn.Module):
    pass