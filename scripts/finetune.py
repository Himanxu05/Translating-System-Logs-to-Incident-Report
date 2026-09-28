"""Fine-tune a small instruct model with (Q)LoRA and write test predictions.

On a GPU (Colab T4) this is QLoRA: the base model is loaded in 4-bit and only small
LoRA adapters are trained. Without a GPU it falls back to plain LoRA in float32,
which is only useful as a quick check that everything runs:

    # Colab, T4
    python scripts/finetune.py --epochs 2
    # CPU smoke test with a tiny model
    python scripts/finetune.py --base Qwen/Qwen2.5-0.5B-Instruct --max-steps 3 --limit 4

Writes:
    outputs/<run>/adapter/                      LoRA weights
    outputs/<run>/train_info.json               loss curve, time, trainable params
    predictions/<run>.jsonl                     fine-tuned model on data/test.jsonl
    predictions/<base-name>-base.jsonl          same base model with no fine-tuning
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from logreport.dataset import read_jsonl
from logreport.prompt import messages
from logreport.report import Report


def prompt_text(tok, log: str) -> str:
    return tok.apply_chat_template(messages(log), tokenize=False, add_generation_prompt=True)


def encode_example(tok, ex: dict, max_len: int) -> dict:
    """Tokens for prompt + answer, with the prompt part masked out of the loss (-100),
    so the model is only trained to write the report, not to repeat the logs."""
    prompt = tok(prompt_text(tok, ex["log"]), add_special_tokens=False)["input_ids"]
    answer = Report.model_validate(ex["report"]).to_json() + tok.eos_token
    answer_ids = tok(answer, add_special_tokens=False)["input_ids"]
    prompt = prompt[: max_len - len(answer_ids)]  # rare very long windows: cut the logs
    ids = prompt + answer_ids
    return {"input_ids": ids, "labels": [-100] * len(prompt) + answer_ids}


def collate(batch, pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id)
    labels = torch.full((len(batch), n), -100)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        labels[i, :k] = torch.tensor(b["labels"])
        mask[i, :k] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


@torch.no_grad()
def generate(model, tok, test: list[dict], out_path: Path, batch_size: int) -> float:
    """Greedy decoding for every test window. Returns seconds per window."""
    model.eval()
    model.config.use_cache = True  # k-bit prep / checkpointing turn it off; generation needs it
    tok.padding_side = "left"
    rows, start = [], time.perf_counter()
    for i in range(0, len(test), batch_size):
        chunk = test[i:i + batch_size]
        enc = tok([prompt_text(tok, e["log"]) for e in chunk], return_tensors="pt",
                  padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=320, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for e, seq in zip(chunk, out[:, enc["input_ids"].shape[1]:], strict=True):
            rows.append({"id": e["id"], "output": tok.decode(seq, skip_special_tokens=True)})
        print(f"  generated {len(rows)}/{len(test)}", flush=True)
    per_window = (time.perf_counter() - start) / max(len(rows), 1)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return per_window


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--run", help="name for outputs/ and predictions/ (default from base)")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop early (smoke tests)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--limit", type=int, help="only use N train and N test examples")
    ap.add_argument("--gen-batch", type=int, default=4)
    ap.add_argument("--skip-base", action="store_true", help="don't evaluate the untuned base")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    gpu = torch.cuda.is_available()
    short = args.base.split("/")[-1].lower()
    run = args.run or f"{short}-{'qlora' if gpu else 'lora-cpu'}"
    out_dir = ROOT / "outputs" / run
    train = read_jsonl(ROOT / "data" / "train.jsonl")[: args.limit]
    test = read_jsonl(ROOT / "data" / "test.jsonl")[: args.limit]
    print(f"device: {torch.cuda.get_device_name(0) if gpu else 'cpu'}, run: {run}, "
          f"train {len(train)}, test {len(test)}")

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if gpu:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.float16)  # T4 has no bf16
        model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=quant,
                                                     device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)

    if not args.skip_base:
        print("predicting with the untuned base model...")
        base_s = generate(model, tok, test, ROOT / "predictions" / f"{short}-base.jsonl",
                          args.gen_batch)
        print(f"  base model: {base_s:.2f}s per window")

    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                      "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.config.use_cache = False  # incompatible with gradient checkpointing during training
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} of {total:,} ({trainable / total:.2%})")

    train_ds = [encode_example(tok, e, args.max_len) for e in train]
    tok.padding_side = "right"
    # warmup as steps, not warmup_ratio: that argument is gone in transformers 5
    steps = args.max_steps if args.max_steps > 0 else math.ceil(
        len(train_ds) / (args.batch * args.grad_accum) * args.epochs)
    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"), num_train_epochs=args.epochs,
        max_steps=args.max_steps, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * steps)), per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, logging_steps=5, save_strategy="no",
        fp16=gpu, gradient_checkpointing=gpu, report_to=[], remove_unused_columns=False,
        optim="paged_adamw_8bit" if gpu else "adamw_torch", seed=42,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id))
    start = time.perf_counter()
    result = trainer.train()
    minutes = (time.perf_counter() - start) / 60
    model.save_pretrained(out_dir / "adapter")

    print("predicting with the fine-tuned model...")
    tuned_s = generate(model, tok, test, ROOT / "predictions" / f"{run}.jsonl", args.gen_batch)

    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    info = {"base": args.base, "run": run, "device": torch.cuda.get_device_name(0) if gpu
            else "cpu", "quantized_4bit": gpu, "train_examples": len(train),
            "epochs": args.epochs, "max_steps": args.max_steps, "lr": args.lr,
            "lora_rank": args.rank, "trainable_params": trainable, "total_params": total,
            "train_minutes": round(minutes, 1), "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "train_loss_avg": round(result.training_loss, 4),
            "seconds_per_window_tuned": round(tuned_s, 2),
            "steps": trainer.state.global_step,
            "log_history": trainer.state.log_history}
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "log_history"}, indent=2))
    if losses and not all(math.isfinite(x) for x in losses):
        print("WARNING: loss went NaN/inf")


if __name__ == "__main__":
    main()
