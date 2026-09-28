"""Train the v1-style LSTM baseline and write predictions/v1-lstm.jsonl. Runs on CPU.

    python scripts/train_lstm.py
"""

import json
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from logreport.dataset import read_jsonl
from logreport.lstm import (EOS, SOS, Seq2Seq, Vocab, input_tokens, pad, target_tokens,
                            tokens_to_report)

MAX_IN = 400
EPOCHS = 30
BATCH = 32


def main() -> None:
    torch.manual_seed(42)
    random.seed(42)
    train = read_jsonl(ROOT / "data" / "train.jsonl")
    test = read_jsonl(ROOT / "data" / "test.jsonl")
    random.shuffle(train)
    n_val = len(train) // 10
    val, train = train[:n_val], train[n_val:]

    src_seqs = [input_tokens(e["log"], MAX_IN) for e in train]
    tgt_seqs = [[SOS] + target_tokens(e["report"]) + [EOS] for e in train]
    src, tgt = Vocab(src_seqs), Vocab(tgt_seqs)
    print(f"train {len(train)}, val {len(val)}, input vocab {len(src)}, output vocab {len(tgt)}")

    def tensors(examples):
        x = pad([src.encode(input_tokens(e["log"], MAX_IN)) for e in examples])
        y = pad([tgt.encode([SOS] + target_tokens(e["report"]) + [EOS]) for e in examples])
        return x, y[:, :-1], y[:, 1:]

    model = Seq2Seq(len(src), len(tgt))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    xv, yv_in, yv_out = tensors(val)
    best, best_state, patience = float("inf"), None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        random.shuffle(train)
        t, total = time.time(), 0.0
        for i in range(0, len(train), BATCH):
            x, y_in, y_out = tensors(train[i:i + BATCH])
            logits = model(x, y_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_out.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # v1: clipnorm=5
            opt.step()
            total += loss.item() * len(x)
        model.eval()
        with torch.no_grad():
            lv = model(xv, yv_in)
            val_loss = loss_fn(lv.reshape(-1, lv.size(-1)), yv_out.reshape(-1)).item()
        print(f"epoch {epoch:>2}  train {total / len(train):.3f}  val {val_loss:.3f}  "
              f"({time.time() - t:.0f}s)", flush=True)
        if val_loss < best - 1e-3:
            best, patience = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 5:  # v1: EarlyStopping(patience=5, restore_best_weights=True)
                break
    model.load_state_dict(best_state)
    model.eval()

    preds = []
    for i in range(0, len(test), BATCH):
        batch = test[i:i + BATCH]
        x = pad([src.encode(input_tokens(e["log"], MAX_IN)) for e in batch])
        for e, ids in zip(batch, model.greedy(x), strict=True):
            toks = []
            for j in ids:
                if tgt.itos[j] == EOS:
                    break
                toks.append(tgt.itos[j])
            preds.append({"id": e["id"], "output": tokens_to_report(toks)})
    out = ROOT / "predictions" / "v1-lstm.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(json.dumps(p) + "\n" for p in preds))
    print(f"wrote {out.relative_to(ROOT)}, params {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
