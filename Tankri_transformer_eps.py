device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from collections import Counter
import pandas as pd

def load_data(file_path):
    # Load Excel file into a DataFrame
    df = pd.read_excel(file_path, engine="openpyxl")  

    # Assuming the first two columns contain the source and target texts
    src_texts = df.iloc[:, 0].astype(str).tolist()  # Convert to string and list
    tgt_texts = df.iloc[:, 2].astype(str).tolist()

    return src_texts, tgt_texts

# Load the data
src_texts, tgt_texts = load_data("C:/Users/IIT - MANDI/Downloads/new Takri data.xlsx")

# Print sample data
#print(src_texts[:5], tgt_texts[:5])  # Print first 5 entries

# Tokenization (You can use nltk or spaCy)
def tokenize(text):
    return text.split()

src_tokenized = [tokenize(sentence) for sentence in src_texts]
tgt_tokenized = [tokenize(sentence) for sentence in tgt_texts]

def build_vocab(tokenized_texts):
    counter = Counter(token for sentence in tokenized_texts for token in sentence)
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    vocab.update({word: i + 4 for i, (word, _) in enumerate(counter.most_common())})
    return vocab

src_vocab = build_vocab(src_tokenized)
tgt_vocab = build_vocab(tgt_tokenized)

# Convert words to indices
def encode_text(text, vocab):
    return [vocab["<sos>"]] + [vocab[token] for token in text] + [vocab["<eos>"]]

src_encoded = [encode_text(text, src_vocab) for text in src_tokenized]
tgt_encoded = [encode_text(text, tgt_vocab) for text in tgt_tokenized]

# Padding function
def pad_sequences(sequences, max_len, pad_idx):
    return [seq + [pad_idx] * (max_len - len(seq)) for seq in sequences]

max_src_len = max(len(seq) for seq in src_encoded)
max_tgt_len = max(len(seq) for seq in tgt_encoded)

src_encoded = pad_sequences(src_encoded, max_src_len, src_vocab["<pad>"])
tgt_encoded = pad_sequences(tgt_encoded, max_tgt_len, tgt_vocab["<pad>"])


from sklearn.model_selection import train_test_split

src_train, src_rem, tgt_train, tgt_rem = train_test_split(
    src_encoded, tgt_encoded, test_size=0.1, random_state=42
)
src_val, src_test, tgt_val, tgt_test = train_test_split(
    src_rem, tgt_rem, test_size=0.5, random_state=42
)


import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, src_seqs, tgt_seqs):
        self.src = [torch.tensor(s, dtype=torch.long) for s in src_seqs]
        self.tgt = [torch.tensor(t, dtype=torch.long) for t in tgt_seqs]
        self.pad = src_vocab["<pad>"]

    def __len__(self): 
        return len(self.src)

    def __getitem__(self, i):
        return {"src": self.src[i], "tgt": self.tgt[i]}


from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    src_batch = [item["src"] for item in batch]
    tgt_batch = [item["tgt"] for item in batch]

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=src_vocab["<pad>"])
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_vocab["<pad>"])

    # padding masks: True where token is pad
    src_key_padding_mask = (src_padded == src_vocab["<pad>"])
    tgt_key_padding_mask = (tgt_padded == tgt_vocab["<pad>"])

    # decoder input / output
    tgt_input  = tgt_padded[:, :-1]
    tgt_output = tgt_padded[:, 1:]

    # causal mask for decoder (square, with -inf above diagonal)
    seq_len = tgt_input.size(1)
    decoder_mask = torch.triu(torch.ones((seq_len, seq_len), 
                                         device=src_padded.device) * float('-inf'), diagonal=1)

    return {
      "src": src_padded,
      "tgt_input": tgt_input,
      "tgt_output": tgt_output,
      "src_key_padding_mask": src_key_padding_mask,
      "tgt_key_padding_mask": tgt_key_padding_mask[:, :-1],
      "decoder_mask": decoder_mask
    }


from torch.utils.data import DataLoader

train_ds = TranslationDataset(src_train, tgt_train)
val_ds   = TranslationDataset(src_val,   tgt_val)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, collate_fn=collate_fn)


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

import torch
import math
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu

def greedy_decode(model, src, src_key_padding_mask, max_len, start_symbol, eos_symbol, device):
    """
    src: [1, src_len]
    src_key_padding_mask: [1, src_len]
    """
    model.eval()
    memory = model.transformer.encoder(
        model.pos_enc(model.src_embed(src) * math.sqrt(model.transformer.d_model)),
        src_key_padding_mask=src_key_padding_mask
    )
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)  # [1,1]
    for i in range(max_len - 1):
        tgt_mask = model.transformer.generate_square_subsequent_mask(ys.size(1)).to(device)
        out = model.transformer.decoder(
            model.pos_enc(model.tgt_embed(ys) * math.sqrt(model.transformer.d_model)),
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        prob = model.fc_out(out[:, -1])        # [1, vocab_size]
        next_word = prob.argmax(dim=-1).item() # scalar
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word)], dim=1)
        if next_word == eos_symbol:
            break
    return ys.squeeze().tolist()

inv_src_vocab = {idx: tok for tok, idx in src_vocab.items()}
inv_tgt_vocab = {idx: tok for tok, idx in tgt_vocab.items()}

# 3. Hyperparameters
d_model            = 512
nhead              = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward    = 2048
dropout            = 0.1
MAX_LEN            = max_tgt_len  # or your chosen cutoff
num_epochs         = 10
lr                 = 1e-4

# 4. Instantiate your model (the class from the previous message)
model = TransformerSeq2Seq(
    src_vocab_size     = len(src_vocab),
    tgt_vocab_size     = len(tgt_vocab),
    d_model            = d_model,
    nhead              = nhead,
    num_encoder_layers = num_encoder_layers,
    num_decoder_layers = num_decoder_layers,
    dim_feedforward    = dim_feedforward,
    dropout            = dropout
).to(device)

# 5. Optimizer & loss
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<pad>"])

# 6. (Re)create your validation dataset reference
#    so you can grab raw tensors for inference printing:
val_dataset = TranslationDataset(src_val, tgt_val)

model.to(device)

train_losses = []
val_losses   = []
val_bleus    = []

for epoch in range(1, num_epochs + 1):
    # ─── Training Pass ─────────────────────────────────────────
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in train_bar:
        optimizer.zero_grad()
        out = model(
            batch["src"].to(device),
            batch["tgt_input"].to(device),
            batch["src_key_padding_mask"].to(device),
            batch["tgt_key_padding_mask"].to(device),
            batch["decoder_mask"].to(device)
        )
        loss = criterion(
            out.view(-1, out.size(-1)),
            batch["tgt_output"].to(device).contiguous().view(-1)
        )
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.set_postfix(train_loss=(running_loss / (train_bar.n + 1)))

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ─── Validation Pass ────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]  ", leave=False)
    with torch.no_grad():
        for batch in val_bar:
            out = model(
                batch["src"].to(device),
                batch["tgt_input"].to(device),
                batch["src_key_padding_mask"].to(device),
                batch["tgt_key_padding_mask"].to(device),
                batch["decoder_mask"].to(device)
            )
            loss = criterion(
                out.view(-1, out.size(-1)),
                batch["tgt_output"].to(device).contiguous().view(-1)
            )
            val_loss += loss.item()
            val_bar.set_postfix(val_loss=(val_loss / (val_bar.n + 1)))

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

    # Compute corpus-level BLEU on validation set
    model.eval()
    references = []
    hypotheses = []
    for sample in val_dataset:
        src = sample["src"].unsqueeze(0).to(device)
        src_mask = (src == src_vocab["<pad>"])
        pred_ids = greedy_decode(
            model, src, src_mask,
            max_len=MAX_LEN,
            start_symbol=tgt_vocab["<sos>"], eos_symbol=tgt_vocab["<eos>"],
            device=device
        )
        # Drop SOS/EOS
        pred_tokens = [inv_tgt_vocab[i] for i in pred_ids[1:-1]]
        ref_tokens  = [inv_tgt_vocab[i] for i in sample["tgt"].tolist()[1:-1]]
        hypotheses.append(pred_tokens)
        references.append([ref_tokens])
    bleu = corpus_bleu(references, hypotheses) * 100
    val_bleus.append(bleu)
    print(f"    → Validation BLEU: {bleu:.2f}%")

    # ─── Inference Samples ──────────────────────────────────────
    print("└ Sample outputs:")
    for i in range(3):  # show 3 samples
        sample = val_dataset[i]
        src = sample["src"].unsqueeze(0).to(device)  # [1, src_len]
        src_mask = (src == src_vocab["<pad>"])        # key_padding_mask
        pred_ids = greedy_decode(
            model, src, src_mask, max_len=MAX_LEN,
            start_symbol=tgt_vocab["<sos>"], eos_symbol=tgt_vocab["<eos>"],
            device=device
        )
        # convert IDs back to tokens
        pred_tokens = [inv_tgt_vocab[id] for id in pred_ids]
        src_tokens  = [inv_src_vocab[id] for id in sample["src"].tolist()]
        tgt_tokens  = [inv_tgt_vocab[id] for id in sample["tgt"].tolist()]
        print(f"   • SRC: {' '.join(src_tokens)}")
        print(f"     REF: {' '.join(tgt_tokens[1:-1])}")
        print(f"     PRED: {' '.join(pred_tokens[1:-1])}")

# Plot Loss and BLEU Curves
epochs = list(range(1, num_epochs + 1))
plt.figure()
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('CrossEntropy Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.savefig('loss_curve.png', dpi=300)
plt.show()

plt.figure()
plt.plot(epochs, val_bleus, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Corpus BLEU (%)')
plt.title('Validation BLEU over Epochs')
plt.savefig('bleu_curve.png', dpi=300)
plt.show()

torch.save(model, 'tankri_transformer_10eps.pkl')
torch.save(src_vocab, 'src_vocab.pkl')
torch.save(tgt_vocab, 'tgt_vocab.pkl')