"""v1's model, rewritten in PyTorch and trained on the new data, as a baseline.

Same design as v1/incident_report_generator.py: word tokens with v1's
normalisation (lowercase, numbers/IPs/timestamps masked, punctuation stripped),
a bidirectional LSTM encoder, an LSTM decoder with dot-product attention over the
encoder states, LayerNorm and a softmax over the output vocabulary.

Two changes so it can take part at all:
- It outputs a flat "status incident category network ..." sequence that is turned
  into the JSON report afterwards (v1's tokenizer strips the punctuation JSON needs).
- The "N:" line-number prefixes are kept as tokens (L1..L15) instead of being masked
  as numbers, otherwise it couldn't point at evidence lines at all.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import torch
from torch import nn

from .report import NORMAL, Report

PAD, SOS, EOS, UNK = "<pad>", "<start>", "<end>", "<oov>"
V1_FILTERS = re.compile(r'[!"#$%&()*+,\-./:;=?@\[\\\]^_`{|}~\t]')


def normalize_line(text: str) -> str:
    # v1's normalize_log
    text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", "<TIMESTAMP>", text)
    text = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "<IP>", text)
    text = re.sub(r"\b\d+\b", "<NUM>", text)
    text = text.lower().replace("<timestamp>", " TIMESTAMP ").replace("<ip>", " IP ") \
        .replace("<num>", " NUM ")
    return V1_FILTERS.sub(" ", text)


def input_tokens(log: str, max_len: int) -> list[str]:
    toks: list[str] = []
    for line in log.splitlines():
        no, _, text = line.partition(": ")
        toks.append(f"L{no}")
        toks.extend(normalize_line(text).split())
    return toks[:max_len]


def target_tokens(report: dict) -> list[str]:
    r = report
    toks = ["status", r["status"], "category", r["category"], "severity", r["severity"],
            "component", r["component"] or "null", "evidence"]
    toks += [str(i) for i in r["evidence"]]
    return toks


def tokens_to_report(tokens: list[str]) -> str:
    """Turn the flat sequence back into JSON. Returns the raw text if it doesn't fit."""
    try:
        t = tokens
        ev = t.index("evidence")
        data = {"status": t[t.index("status") + 1], "category": t[t.index("category") + 1],
                "severity": t[t.index("severity") + 1],
                "component": None if t[t.index("component") + 1] == "null"
                else t[t.index("component") + 1],
                "evidence": [int(x) for x in t[ev + 1:] if x.isdigit()], "summary": ""}
        return Report.model_validate(data).to_json()
    except Exception:  # missing key, bad value, invalid enum -> counts as invalid output
        return " ".join(tokens)


class Vocab:
    def __init__(self, sequences: list[list[str]], max_size: int = 5000):
        counts = Counter(t for s in sequences for t in s)
        self.itos = [PAD, SOS, EOS, UNK] + [t for t, _ in counts.most_common(max_size - 4)]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, toks: list[str]) -> list[int]:
        return [self.stoi.get(t, 3) for t in toks]

    def __len__(self) -> int:
        return len(self.itos)


class Seq2Seq(nn.Module):
    def __init__(self, in_vocab: int, out_vocab: int, embed: int = 128, units: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.enc_embed = nn.Embedding(in_vocab, embed, padding_idx=0)
        self.encoder = nn.LSTM(embed, units, batch_first=True, bidirectional=True)
        self.state_h = nn.Linear(2 * units, units)
        self.state_c = nn.Linear(2 * units, units)
        self.enc_proj = nn.Linear(2 * units, units)
        self.dec_embed = nn.Embedding(out_vocab, embed, padding_idx=0)
        self.decoder = nn.LSTM(embed, units, batch_first=True)
        self.norm = nn.LayerNorm(2 * units)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * units, out_vocab)

    def encode(self, x):
        mask = x != 0
        enc, (h, c) = self.encoder(self.drop(self.enc_embed(x)))
        h = torch.tanh(self.state_h(torch.cat([h[0], h[1]], -1))).unsqueeze(0)
        c = torch.tanh(self.state_c(torch.cat([c[0], c[1]], -1))).unsqueeze(0)
        return self.enc_proj(enc), mask, (h, c)

    def decode(self, y, enc, mask, state):
        dec, state = self.decoder(self.drop(self.dec_embed(y)), state)
        scores = dec @ enc.transpose(1, 2)  # dot-product attention, like keras.layers.Attention
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
        context = torch.softmax(scores, -1) @ enc
        return self.out(self.drop(self.norm(torch.cat([dec, context], -1)))), state

    def forward(self, x, y_in):
        enc, mask, state = self.encode(x)
        logits, _ = self.decode(y_in, enc, mask, state)
        return logits

    @torch.no_grad()
    def greedy(self, x, max_len: int = 60) -> list[list[int]]:
        enc, mask, state = self.encode(x)
        y = torch.full((x.size(0), 1), 1, dtype=torch.long)  # <start>
        outs = []
        for _ in range(max_len):
            logits, state = self.decode(y, enc, mask, state)
            y = logits[:, -1:].argmax(-1)
            outs.append(y)
        return torch.cat(outs, 1).tolist()


def pad(seqs: list[list[int]], length: int | None = None) -> torch.Tensor:
    length = length or max(len(s) for s in seqs)
    return torch.tensor([s[:length] + [0] * (length - len(s[:length])) for s in seqs])


def save(path, model: Seq2Seq, src: Vocab, tgt: Vocab, max_in: int) -> None:
    torch.save({"state": model.state_dict(), "src": src.itos, "tgt": tgt.itos, "max_in": max_in},
               path)


def normal_output() -> str:
    return json.dumps(NORMAL.model_dump())
