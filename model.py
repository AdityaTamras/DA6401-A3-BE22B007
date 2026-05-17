import math
import copy
import os
import gdown
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k=Q.size(-1)
    raw_scores=torch.matmul(Q, K.transpose(-2, -1))/torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        raw_scores=raw_scores.masked_fill(mask, -float('inf'))
    attn_weights=F.softmax(raw_scores, dim=-1)
    outputs=torch.matmul(attn_weights, V)
    return outputs, attn_weights

def make_src_mask(src, pad_idx=1):
    mask=(src==pad_idx)
    mask=mask.unsqueeze(1).unsqueeze(2)
    return mask

def make_tgt_mask(tgt, pad_idx=1):
    tgt_len=tgt.size(-1)
    padding_mask=(tgt==pad_idx)
    padding_mask=padding_mask.unsqueeze(1).unsqueeze(1)
    causal_mask=~torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device))
    tgt_mask = padding_mask | causal_mask
    return tgt_mask

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        self.W_q=nn.Linear(d_model, d_model)
        self.W_k=nn.Linear(d_model, d_model)
        self.W_v=nn.Linear(d_model, d_model)
        self.W_o=nn.Linear(d_model, d_model)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size=query.size(0)
        q=self.W_q(query)
        k=self.W_k(key)
        v=self.W_v(value)

        q=q.view(batch_size, -1, self.num_heads, self.d_k)
        k=k.view(batch_size, -1, self.num_heads, self.d_k)
        v=v.view(batch_size, -1, self.num_heads, self.d_k)

        q=q.transpose(1, 2)
        k=k.transpose(1, 2)
        v=v.transpose(1, 2)

        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        attn_output=attn_output.transpose(1, 2)
        attn_output=attn_output.contiguous().view(batch_size, -1, self.d_model)
        attn_output=self.W_o(attn_output)
        return attn_output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        pe=torch.zeros(max_len, d_model)
        pos=torch.arange(0, max_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
        pe[:, 0::2]=torch.sin(pos*div_term)
        pe[:, 1::2]=torch.cos(pos*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1=nn.Linear(d_model, d_ff)
        self.linear2=nn.Linear(d_ff, d_model)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self, x):
        x=self.linear1(x)
        x=F.relu(x)
        x=self.dropout(x)
        x=self.linear2(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.dropout=nn.Dropout(p=dropout)
        self.self_attn=MultiHeadAttention(self.d_model, self.num_heads, dropout)
        self.positionwise_ff=PositionwiseFeedForward(self.d_model, self.d_ff, dropout)
        self.layer_norm1=nn.LayerNorm(self.d_model)
        self.layer_norm2=nn.LayerNorm(self.d_model)
        
    def forward(self, x, src_mask):
        attn_output=self.self_attn.forward(x, x, x, src_mask)
        x=self.layer_norm1(x+self.dropout(attn_output))
        ffn_output=self.positionwise_ff.forward(x)
        x=self.layer_norm2(x+self.dropout(ffn_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.dropout=nn.Dropout(p=dropout)
        self.self_attn=MultiHeadAttention(self.d_model, self.num_heads, dropout)
        self.cross_attn=MultiHeadAttention(self.d_model, self.num_heads, dropout)
        self.positionwise_ff=PositionwiseFeedForward(self.d_model, self.d_ff, dropout)
        self.layer_norm1=nn.LayerNorm(self.d_model)
        self.layer_norm2=nn.LayerNorm(self.d_model)
        self.layer_norm3=nn.LayerNorm(self.d_model)
        

    def forward(self, x, memory, src_mask, tgt_mask):
        x=self.layer_norm1(x+self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x=self.layer_norm2(x+self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x=self.layer_norm3(x+self.dropout(self.positionwise_ff(x)))
        return x
    
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers=nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm=nn.LayerNorm(layer.d_model)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x=layer(x, mask)
        x=self.norm(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers=nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm=nn.LayerNorm(layer.d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x=layer(x, memory, src_mask, tgt_mask)
        x=self.norm(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size=None, tgt_vocab_size=None, d_model=512, N=6, num_heads=8, d_ff=2048, dropout=0.1, checkpoint_path="checkpoint_epoch_2.pt", gdrive_id="1ExeDE2qKBKsj96hqi-Jkk_vLobCAKIl3"):
        super().__init__()
        import spacy
        import subprocess
        from dataset import Multi30kDataset

        try:
            self.spacy_de=spacy.load("de_core_news_sm")
        except OSError:
            print("Downloading de_core_news_sm...")
            subprocess.run(["python", "-m", "spacy", "download", "de_core_news_sm"], check=True)
            self.spacy_de=spacy.load("de_core_news_sm")

        dataset=Multi30kDataset(split="train")
        dataset.build_vocab(min_freq=3)
        self.src_vocab=dataset.src_vocab
        self.tgt_vocab=dataset.tgt_vocab
        self.idx_to_token={v: k for k, v in self.tgt_vocab.items()}

        src_vocab_size=len(self.src_vocab)
        tgt_vocab_size=len(self.tgt_vocab)

        self.d_model=d_model
        self.src_embedding=nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding=nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding=PositionalEncoding(d_model, dropout, max_len=5000)
        encoder_layer=EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder=Encoder(encoder_layer, N)
        decoder_layer=DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(decoder_layer, N)
        self.linear=nn.Linear(d_model, tgt_vocab_size)

        if not os.path.exists(checkpoint_path):
            print(f"Downloading checkpoint from Google Drive....")
            gdown.download(id=gdrive_id, output=checkpoint_path, quiet=False)

        checkpoint=torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded weights from {checkpoint_path}")

    def encode(self, src, src_mask):
        src_embeds=self.src_embedding(src)*math.sqrt(self.d_model)
        src_encodings=self.pos_encoding(src_embeds)
        return self.encoder(src_encodings, src_mask)
        
    def decode(self, memory, src_mask, tgt, tgt_mask):
        tgt_embeds=self.tgt_embedding(tgt)*math.sqrt(self.d_model)
        tgt_encodings=self.pos_encoding(tgt_embeds)
        return self.linear(self.decoder(tgt_encodings, memory, src_mask, tgt_mask))

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
    def infer(self, src_sentence, src_vocab, tgt_vocab, spacy_de, max_len=50):
        self.eval()

        tokens=[tok.text for tok in spacy_de(src_sentence)]

        unk_idx=src_vocab["<unk>"]
        sos_idx=src_vocab["<sos>"]
        eos_idx=src_vocab["<eos>"]
        src_indices = [sos_idx] + [src_vocab.get(tok, unk_idx) for tok in tokens] + [eos_idx]
        device=next(self.parameters()).device
        src = torch.LongTensor(src_indices).unsqueeze(0)

        with torch.no_grad():
            src_mask=make_src_mask(src)
            memory=self.encode(src, src_mask)
            tgt_sos_idx=tgt_vocab["<sos>"]
            tgt_eos_idx=tgt_vocab["<eos>"]
            tgt=torch.LongTensor([[tgt_sos_idx]])
            for _ in range(max_len):
                tgt_mask=make_tgt_mask(tgt)
                logits=self.decode(memory, src_mask, tgt, tgt_mask)
                next_token=logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if next_token.item()==tgt_eos_idx:
                    break
                tgt=torch.cat([tgt, next_token], dim=1)
        predicted_tokens=tgt.squeeze(0).tolist()[1:]
        translated = " ".join(self.idx_to_token.get(idx, "<unk>") for idx in predicted_tokens)
        return translated












    

    


    



