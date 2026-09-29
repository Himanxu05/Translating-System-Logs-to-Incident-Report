# Log → Incident Report

Give it a window of raw log lines from a server; get back a structured incident report:

```
 1: Dec 10 10:54:54 LabSZ sshd[24898]: Failed password for root from 183.62.140.253 port 38375 ssh2
 2: Dec 10 10:54:54 LabSZ sshd[24898]: Received disconnect from 183.62.140.253: 11: Bye Bye [preauth]
 3: Dec 10 10:54:54 LabSZ sshd[24900]: pam_unix(sshd:auth): authentication failure; ... rhost=183.62.140.253 user=root
 ...
```
```json
{"status": "incident", "category": "authentication", "severity": "medium", "component": "sshd",
 "evidence": [1, 3, 4, 6, 7, 9, 10, 12, 13, 15],
 "summary": "10 authentication failures from sshd, e.g. \"Failed password for root from 183.62.140.253 ...\""}
```

This is version 2. The main question it answers: **does a small LLM fine-tuned with QLoRA handle logs from
systems it has never seen, where rules and my old model don't?**

## From v1 to v2

v1 (still in [`v1/`](v1/)) was a TensorFlow sequence-to-sequence model trained on synthetic logs. Looking back
at it, it had real problems:

- **The README didn't match the code.** It described a from-scratch Transformer with positional encoding and
  multi-head attention. The code is a BiLSTM encoder, an LSTM decoder and Keras' built-in attention layer.
- **It made up details.** It masked every number in the input logs but had to produce reports containing the
  real numbers and host names, so it guessed them.
- **The task was too easy to mean anything.** The data came from 6 templates, and the test set used the same
  templates, so the model only had to memorise them. The rule-based function in the same file already
  covered all 6 cases.
- **Evaluation:** a hand-written BLEU-1 on 5 samples.

v2 fixes each of these:

| | v1 | v2 |
|---|---|---|
| data | 6 synthetic templates | 12 real systems from [Loghub](https://github.com/logpai/loghub) |
| output | free text | JSON report with fields that can be checked exactly |
| test set | same templates as training | includes 4 systems never seen in training |
| model | BiLSTM seq2seq (TensorFlow) | Qwen2.5-1.5B-Instruct + QLoRA (PyTorch), compared against 4 baselines |
| metric | BLEU-1 on 5 samples | per-field accuracy on 168 windows, plus "did it invent a component?" |

## The task

Input: 15 consecutive raw log lines, numbered. Output: a JSON report.

| field | values |
|---|---|
| `status` | `incident` / `normal` |
| `category` | authentication, network, storage, hardware, service, permission, configuration, or none |
| `severity` | medium, high, critical, or none |
| `component` | the program that logged the problem, **copied exactly from the log** |
| `evidence` | line numbers of the problem lines |
| `summary` | one sentence |

The prompt, including the category and severity definitions, is in [`src/logreport/prompt.py`](src/logreport/prompt.py).
Every model gets the same one.

## Data

- **Source:** [Loghub](https://github.com/logpai/loghub)'s 2,000-line samples of 12 systems: Linux, OpenSSH,
  Apache, HDFS, Hadoop, Spark, BGL, Zookeeper, OpenStack, Windows, Thunderbird and Proxifier. Loghub is free
  for research and academic use; the logs aren't copied into this repo, `scripts/build_dataset.py` downloads
  them.
- **Labels:** Loghub parses every line into a template (e.g. `Failed password for <*> from <*> port <*> ssh2`).
  - I went through the templates of each system and marked the ones that describe a problem, with a
    category and severity. The decisions are listed one per template in
    [`src/logreport/labels.py`](src/logreport/labels.py).
  - BGL comes with expert anomaly labels, and those are used as they are. Thunderbird's sample is
    expert-labelled all normal.
  - A window's report is then built from its lines' labels. No LLM is involved in labelling, so the reference
    answers don't favour any model.
  - One catch: in BGL and Thunderbird the raw lines start with the expert label itself, so it is stripped
    from the input.
- **Split:**
  - train: 8 systems (Linux, HDFS, Hadoop, Spark, BGL, Windows, Thunderbird, Proxifier), windows from the
    first 80% of each file. That's 674 windows, half of them incidents after downsampling the normal ones.
  - test, seen systems: 48 windows from the last 20% of the same 8 files.
  - test, **unseen systems**: 120 windows from OpenSSH, Apache, Zookeeper and OpenStack, which never appear
    in training.

The unseen systems are mostly incidents. Nearly every 15-line OpenSSH window has a failed login, for
example, so status alone is easy there. Category, component and evidence are the harder parts.

## Results

`python scripts/score.py` scores every file in `predictions/`.
- **component:** counts if it names the same program (`sshd` matches `sshd(pam_unix)`).
- **invented component:** it named something that doesn't appear anywhere in the log.

| model | test set | status | category | severity | component | evidence F1 | invented component |
|---|---|---|---|---|---|---|---|
| keyword rules (v1-style) | seen systems | 100% | 85% | 81% | 0% | 84% | 0% |
| keyword rules (v1-style) | **unseen systems** | 48% | 33% | 39% | 0% | 35% | 0% |
| v1 LSTM, retrained in PyTorch | seen systems | 98% | 96% | 98% | 88% | 80% | 0% |
| v1 LSTM, retrained in PyTorch | **unseen systems** | 78% | 77% | 66% | 34% | 51% | 50% |
| Qwen2.5-1.5B, no fine-tuning | seen systems | 38% | 33% | 21% | 31% | 15% | 2% |
| Qwen2.5-1.5B, no fine-tuning | **unseen systems** | 72% | 61% | 13% | 69% | 51% | 14% |
| **Qwen2.5-1.5B + QLoRA** | seen systems | 98% | 98% | 98% | 94% | 95% | 0% |
| **Qwen2.5-1.5B + QLoRA** | **unseen systems** | **95%** | **92%** | 57% | **80%** | **82%** | **0%** |
| gpt-oss-120b, prompted | | _running_ | | | | | |

Valid JSON: 100% for rules and the LSTM (they can't produce anything else), 92% for the untuned Qwen, and
97% for the tuned one.

Fine-tuning: 674 examples, 2 epochs, 170 steps, 58 minutes on a free Colab T4. 18.5M trainable parameters
(LoRA), about 2% of the model. Training loss went from 0.65 to 0.004. Details are in
`outputs/qwen2.5-1.5b-instruct-qlora/train_info.json`, which the notebook writes.

What the fine-tuned model shows:
- **It generalises to new systems.** On the 4 systems it never saw, category accuracy is 92%, against 77% for
  the LSTM and 33% for rules. Evidence F1 is 82% against 51%.
- **It never invents a component.** It copies the component from the log every time, where the LSTM invents
  one in half the unseen windows. That was the main problem with v1.
- **Fine-tuning mostly teaches the format and the labelling conventions.** The untuned model already does
  reasonably on unseen systems (61% category), but it uses its own ideas of severity and sometimes writes
  invalid JSON.
- **Severity on unseen systems is the weak spot (57%).** Nearly all of it is Apache: all 29 incidents there
  are labelled `high` (I marked mod_jk worker failures as high), and the model says `medium` every time.
  On the same windows it gets the component right every time and the category 90% of the time. Severity depends on labelling conventions more
  than any other field, so it transfers the least.
- **Its 5 invalid outputs are all Zookeeper.** It wrote a severity that isn't allowed (`warn`, copied from
  the log level) or escaped quotes wrongly in the summary. Constrained decoding (only allowing valid JSON)
  would fix both.

What the baselines show:
- **Keyword rules** are fine on the systems their keywords were written for, and fall apart on new ones (85%
  → 33% category). They never identify the component. The keywords are v1's plus ones taken from the
  training systems only; nothing was added after looking at the unseen systems.
- **The v1 LSTM** is nearly perfect on familiar systems and noticeably worse on new ones. It invents the
  component in half of the unseen-system windows: it can only output program names it saw in training, so
  on OpenSSH it writes `sshd(pam_unix)` from the Linux logs. This is v1's "makes up details" problem again,
  now measured.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[train,llm,dev]"

python scripts/build_dataset.py      # data/train.jsonl, data/test.jsonl
python scripts/predict_rules.py      # keyword baseline
python scripts/train_lstm.py         # v1-style LSTM baseline (CPU, about 4 minutes)
python scripts/predict_llm.py --model openai/gpt-oss-120b   # prompted baseline (GROQ_API_KEY in .env)
python scripts/score.py
```

Fine-tuning needs an NVIDIA GPU. Open [`notebooks/finetune_colab.ipynb`](notebooks/finetune_colab.ipynb) in
Google Colab with a free T4 and run all cells (about an hour). It:
1. predicts with the untuned model
2. fine-tunes with QLoRA (4-bit base model, LoRA rank 16 on all attention and MLP projections, 2 epochs)
3. predicts with the tuned model
4. downloads the predictions and adapter

`scripts/finetune.py` also runs on a CPU as a smoke test with a smaller model:

```bash
python scripts/finetune.py --base Qwen/Qwen2.5-0.5B-Instruct --max-steps 2 --limit 4
```

## Layout

```
src/logreport/
  loghub.py     download + load the Loghub samples
  labels.py     per-template problem labels, component extraction
  report.py     report format, reference reports, parsing model output
  dataset.py    windows, train/test split, balancing
  prompt.py     the prompt every model gets
  rules.py      keyword baseline
  lstm.py       v1's architecture in PyTorch
  metrics.py    scoring
scripts/        build data, run each approach, score
notebooks/      Colab notebook for QLoRA fine-tuning
v1/             the original project, unchanged
tests/
```

## Limitations

- The labels come from one person's reading of the templates. Another person would draw some lines
  differently, for example whether a Hadoop "Address change detected" warning is an incident.
- Loghub's samples are only 2,000 lines per system, so the test set is small (168 windows), and a few systems
  have very few incidents.
- The windows are fixed 15-line chunks. Real incidents don't respect those boundaries.
- The summary field isn't scored.
