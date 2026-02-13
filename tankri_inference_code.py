import torch
import math
import re
import unicodedata
import pickle
#from your_model_module import TransformerSeq2Seq, PositionalEncoding  # wherever you defined them

import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # build [max_len × d_model] PE matrix
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=src_vocab["<pad>"])
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_vocab["<pad>"])
        self.pos_enc   = PositionalEncoding(d_model, max_len=MAX_LEN)
        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers,
            dim_feedforward, dropout, batch_first=True
        )
        self.fc_out    = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt_input,
                src_key_padding_mask, tgt_key_padding_mask, decoder_mask):
        # embed + pos-encode
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.transformer.d_model))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt_input) * math.sqrt(self.transformer.d_model))

        # transformer expects batch_first=True
        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=decoder_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        return self.fc_out(out)

# ─── 1. Settings ────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH   = "tankri_transformer_10eps.pkl"  # your saved state_dict
SRC_VOCAB_PKL= "src_vocab.pkl"
TGT_VOCAB_PKL= "tgt_vocab.pkl"
MAX_LEN      = 128

# ─── 2. Load vocabs ─────────────────────────────────────────────────────
# with open(SRC_VOCAB_PKL, "rb") as f:
#     src_vocab = pickle.load(f)
# with open(TGT_VOCAB_PKL, "rb") as f:
#     tgt_vocab = pickle.load(f)

src_vocab = torch.load(SRC_VOCAB_PKL, map_location=DEVICE)
tgt_vocab = torch.load(TGT_VOCAB_PKL, map_location=DEVICE)

inv_tgt = {i:t for t,i in tgt_vocab.items()}

# ─── 3. Re-build model & load weights ────────────────────────────────────
model = TransformerSeq2Seq(
    src_vocab_size     = len(src_vocab),
    tgt_vocab_size     = len(tgt_vocab),
    d_model            = 512,
    nhead              = 8,
    num_encoder_layers = 6,
    num_decoder_layers = 6,
    dim_feedforward    = 2048,
    dropout            = 0.1
).to(DEVICE)

# model = torch.load(MODEL_PATH, map_location=DEVICE)
import sys
sys.modules['__main__'] = sys.modules[__name__]

import torch.serialization
torch.serialization.add_safe_globals([__name__])
torch.serialization.add_safe_globals([TransformerSeq2Seq])

# now load the model
# model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
with torch.serialization.safe_globals([TransformerSeq2Seq]):
    model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

model.to(DEVICE).eval()

# ─── 4. Preprocessing + greedy decode ───────────────────────────────────
def clean(text):
    t = unicodedata.normalize("NFC", text).lower().strip()
    return re.sub(r"\s+", " ", t)

def greedy_decode(src_ids):
    """
    src_ids: Tensor shape [1, L]
    returns: list of token-ids
    """
    with torch.no_grad():
        src_key_mask = (src_ids == src_vocab["<pad>"])
        memory = model.transformer.encoder(
            model.pos_enc(model.src_embed(src_ids) * math.sqrt(model.transformer.d_model)),
            src_key_padding_mask=src_key_mask
        )
        ys = torch.ones(1, 1, device=DEVICE).fill_(tgt_vocab["<sos>"]).long()
        for _ in range(MAX_LEN-1):
            tgt_mask = model.transformer.generate_square_subsequent_mask(ys.size(1)).to(DEVICE)
            out = model.transformer.decoder(
                model.pos_enc(model.tgt_embed(ys) * math.sqrt(model.transformer.d_model)),
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_mask
            )
            prob   = model.fc_out(out[:, -1, :])           # [1, V]
            next_id= prob.argmax(dim=-1).item()
            ys     = torch.cat([ys, torch.tensor([[next_id]], device=DEVICE)], dim=1)
            if next_id == tgt_vocab["<eos>"]:
                break
    return ys.squeeze().tolist()

def translate(sentence: str) -> str:
    # 1. clean & tokenize
    clean_txt = clean(sentence)
    tokens    = clean_txt.split()  # same as your train script
    # 2. encode + add sos/eos
    ids = [src_vocab["<sos>"]] + [src_vocab.get(t, src_vocab["<unk>"]) for t in tokens] + [src_vocab["<eos>"]]
    src = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    # 3. decode
    out_ids = greedy_decode(src)
    # 4. detokenize (drop sos/eos)
    #    strip everything from <eos> onward
    if tgt_vocab["<eos>"] in out_ids:
        cut = out_ids.index(tgt_vocab["<eos>"])
        out_ids = out_ids[1:cut]
    else:
        out_ids = out_ids[1:]
    return " ".join(inv_tgt[i] for i in out_ids)

# ─── 5. Interactive inference ───────────────────────────────────────────
if __name__ == "__main__":
    print("Enter a sentence (or 'quit' to exit):")
    while True:
        line = input("> ")
        if not line or line.lower() == "quit":
            break
        print("⟶", translate(line))
    print("Goodbye.")
