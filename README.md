# Tankri Transformer Translation Model

This repository contains a Transformer-based sequence-to-sequence model implemented in PyTorch for machine translation between Tankri (Takri) script and English.
The project is part of an effort to support and revive the low-resource Tankri script using modern deep learning techniques.

## 📌 Features

-> End-to-end Transformer Seq2Seq model (Encoder–Decoder)

-> Custom tokenization, vocabulary building, and padding

-> Positional Encoding implementation from scratch
-> Training with teacher forcing
-> Evaluation using Corpus BLEU score
-> Greedy decoding for inference
-> Training/validation loss & BLEU visualization
-> Model and vocabulary checkpoint saving

## 🏗️ Model Architecture

| Component         | Value |
| ---------------   | ----- |
| Embedding Size    | 512   |
| Attention Heads   | 8     |
| Encoder Layers    | 6     |
| Decoder Layers    | 6     |
| Feedforward Dim   | 2048  |
| Dropout           | 0.1   |

## 🔄 Training Pipeline

-> Load and tokenize raw text
-> Build vocabularies (<pad>, <sos>, <eos>, <unk>)
-> Encode & pad sequences
-> Train / Validation / Test split
-> Train using teacher forcing
-> Evaluate using BLEU score
-> Save model & vocabularies

## 📈 Evaluation Metrics

=> Loss Function: CrossEntropyLoss (padding ignored)
=> Metric: Corpus BLEU Score
=> Inference: Greedy decoding

