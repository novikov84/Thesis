import torch
from torchsummary import summary
import numpy as np
import json
import math

from typing import List, Dict, Tuple
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

specs_vocab_size = 300
SPECS_BOS_TOKEN = specs_vocab_size
SPECS_EOS_TOKEN = specs_vocab_size + 1
SPECS_PAD_TOKEN = specs_vocab_size + 2

# Correct
frags_vocab_size = 5113
FRAGS_BOS_TOKEN = frags_vocab_size
FRAGS_EOS_TOKEN = frags_vocab_size + 1
FRAGS_PAD_TOKEN = frags_vocab_size + 2


class Spec2FragsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data : List[Dict[str, List[int]]]
    ) -> None:
        
        self.spectrums = []
        self.frags = []
        
        for cur in tqdm(data):
            self.spectrums.append(
                torch.cat([
                    torch.tensor([SPECS_BOS_TOKEN], dtype=torch.int64),
                    self._get_labels_from_spectrum(
                        torch.tensor(
                            cur['spectrum'],
                            dtype=torch.int64
                        )
                    ),
                    torch.tensor([SPECS_EOS_TOKEN], dtype=torch.int64)
                ])
            )
            self.frags.append(
                torch.cat([
                    torch.tensor([FRAGS_BOS_TOKEN], dtype=torch.int64),
                    torch.tensor(
                        cur['frags'],
                        dtype=torch.int64
                    ),
                    torch.tensor([FRAGS_EOS_TOKEN], dtype=torch.int64)
                ])
            )


    
    def _get_labels_from_spectrum(
        self,
        spectrum : torch.Tensor
    ) -> torch.Tensor:
        res = []
        for idx in (spectrum > 0).nonzero().flatten():
            res.extend([idx] * spectrum[idx])
        return torch.stack(res)

    def __len__(
        self
    ) -> int:
        return len(self.spectrums)

    def __getitem__(
        self,
        idx
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.spectrums[idx].to(device), self.frags[idx].to(device)

class Spec2Gen(torch.utils.data.Dataset):
    def __init__(
        self,
        data : List[List[float]]
    ) -> None:
        
        self.spectrums = []
        
        for cur in tqdm(data):
            self.spectrums.append(
                torch.cat([
                    torch.tensor([SPECS_BOS_TOKEN], dtype=torch.int64),
                    self._get_labels_from_spectrum(
                        torch.tensor(
                            self._get_input_vect(cur),
                            dtype=torch.int64
                        )
                    ),
                    torch.tensor([SPECS_EOS_TOKEN], dtype=torch.int64)
                ])
            )
            
    def _get_input_vect(
        self,
        signals
    ):
        result = [0]*300
        for signal in signals:
            position = int(signal- (-50))
            result[position] += 1
        
        return result
    
    def _get_labels_from_spectrum(
        self,
        spectrum : torch.Tensor
    ) -> torch.Tensor:
        res = []
        for idx in (spectrum > 0).nonzero().flatten():
            res.extend([idx] * spectrum[idx])
        return torch.stack(res)

    def __len__(
        self
    ) -> int:
        return len(self.spectrums)

    def __getitem__(
        self,
        idx
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.spectrums[idx].to(device)

def collate_fn(
    data : List[Tuple[torch.Tensor, torch.Tensor]]
):
    specs = []
    frags = []
    for cur in data:
        specs.append(cur[0])
        frags.append(cur[1])
    return (
        torch.nn.utils.rnn.pad_sequence(
            specs, 
            batch_first=True, 
            padding_value=SPECS_PAD_TOKEN
        ),
        torch.nn.utils.rnn.pad_sequence(
            frags, 
            batch_first=True, 
            padding_value=FRAGS_PAD_TOKEN
        )
    )


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = torch.nn.Linear(d_model, d_model).to(device)
        self.W_k = torch.nn.Linear(d_model, d_model).to(device)
        self.W_v = torch.nn.Linear(d_model, d_model).to(device)
        self.W_o = torch.nn.Linear(d_model, d_model).to(device)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output.to(device)
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = torch.nn.Linear(d_model, d_ff).to(device)
        self.fc2 = torch.nn.Linear(d_ff, d_model).to(device)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads).to(device)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff).to(device)
        self.norm1 = torch.nn.LayerNorm(d_model).to(device)
        self.norm2 = torch.nn.LayerNorm(d_model).to(device)
        self.dropout = torch.nn.Dropout(dropout).to(device)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads).to(device)
        self.cross_attn = MultiHeadAttention(d_model, num_heads).to(device)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff).to(device)
        self.norm1 = torch.nn.LayerNorm(d_model).to(device)
        self.norm2 = torch.nn.LayerNorm(d_model).to(device)
        self.norm3 = torch.nn.LayerNorm(d_model).to(device)
        self.dropout = torch.nn.Dropout(dropout).to(device)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(torch.nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = torch.nn.Embedding(src_vocab_size, d_model).to(device)
        self.decoder_embedding = torch.nn.Embedding(tgt_vocab_size, d_model).to(device)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = torch.nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = torch.nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = torch.nn.Linear(d_model, tgt_vocab_size).to(device)
        self.dropout = torch.nn.Dropout(dropout).to(device)

    def generate_mask(self, src, tgt):
        src_mask = (src != SPECS_PAD_TOKEN).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != FRAGS_PAD_TOKEN).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask.to(device) & nopeak_mask.to(device)
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        #src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        #tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        src_embedded = self.dropout(self.encoder_embedding(src))
        tgt_embedded = self.dropout(self.decoder_embedding(tgt))
                                    
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

def get_num_of_params(
    model : torch.nn.Module
) -> int:
    return sum([reduce(lambda x, y: x * y, cur.shape) for cur in model.parameters()])

@torch.inference_mode
def generate(
    src : torch.Tensor,
    model
) -> torch.Tensor:
    tokens = [FRAGS_BOS_TOKEN]
    while len(tokens) < 100 and tokens[-1] != FRAGS_EOS_TOKEN:
        tokens.append(
            model(
                src=src.unsqueeze(0).to(device),
                tgt=torch.tensor([tokens], dtype=torch.int64).to(device)
            )[0, -1, :].argmax().item()
        )
    return torch.tensor(tokens[1:-1], dtype=torch.int64)

