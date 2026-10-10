# Laya banking email QRC

Fine-tune [Laya](https://pypi.org/project/laya/) so it classifies a retail-bank customer email as a **query**, a **request**, or a **complaint**. Training starts from the English checkpoint `convaiinnovations/laya` and updates the encoder and the decision head.

A **query** only asks for information, such as a rate or a balance. A **request** asks the bank to do something and does not say anything went wrong. A **complaint** says the customer is unhappy about a charge, a payment, or the service. If they are unhappy and also ask for a fix, the label is still a complaint.

*You charged the monthly fee twice. Please reverse it.* That is a complaint: the reversal is a request, but the customer is unhappy about the charge.

Those three descriptions are the question the fine-tuned model answers. They live in [`scripts/qrc.py`](scripts/qrc.py) under `criteria`. Training copies that text onto every row, and the API loads the fine-tuned checkpoint and uses the same object, so the model is always scored against the same words.

## About Laya

[Laya](https://pypi.org/project/laya/) is an open decision model, System One category. It does not write a reply. You pass a state, here the customer email, and one or more typed questions. One forward pass returns a probability for every option.

This project starts from the English checkpoint `convaiinnovations/laya`, a ModernBERT-large model of about 421 million parameters. Fine-tuning updates that encoder and the decision head. `freeze_encoder` is false in [`scripts/finetune_qrc.py`](scripts/finetune_qrc.py).

Laya has three question types:

| Type | What you ask | What comes back |
| --- | --- | --- |
| `choice` | Pick one label from criteria you write | The chosen label and a probability for each label |
| `score` | Place the email on an ordered scale | A number on that scale, plus a probability per level |
| `noul` | A yes/no question | The probability that the answer is yes |

The QRC task is a `choice` question. The demo script also asks a `score` (urgency) and a `noul` (does the customer want a person to call) so a first run shows every type. The HTTP API asks only the QRC question.

## Layout

| Path | Role |
| --- | --- |
| [`data/label_policy.md`](data/label_policy.md) | How an email is labeled |
| [`data/train.jsonl`](data/train.jsonl) | 500 training emails |
| [`data/test.jsonl`](data/test.jsonl) | 24 held-out emails, used by fine-tuning |
| [`data/api_test.jsonl`](data/api_test.jsonl) | 30 fuller emails, 10 of each class, written differently from the one-line train and test sets |
| [`scripts/qrc.py`](scripts/qrc.py) | The QRC question used everywhere |
| [`scripts/setup.py`](scripts/setup.py) | Installs CPU PyTorch and Laya into the active Python |
| [`scripts/first_decision.py`](scripts/first_decision.py) | Scores one email in the terminal |
| [`scripts/finetune_qrc.py`](scripts/finetune_qrc.py) | Fine-tunes the encoder and the QRC decision head |
| [`service/main.py`](service/main.py) | HTTP API: email in, QRC tag and scores out |
| `checkpoints/qrc` | Fine-tuned weights, written by the training script |

`checkpoints/` and `.cache/` are gitignored. The English base checkpoint is about 800 MB and is downloaded on first use into `.cache/huggingface` inside this folder.

## Requirements

- Python 3.10 or newer, from [python.org](https://www.python.org/downloads/) or your system package manager
- About 2 GB of free disk for PyTorch and the English checkpoint
- Internet on the first install and the first model download

Training and inference run on CPU. No NVIDIA GPU is required.

## Setup

Create and activate a virtual environment with Python 3.10 or newer, then from the project folder:

```bash
python scripts/setup.py
```

The script installs the CPU build of PyTorch, then Laya from [`requirements.txt`](requirements.txt), into whichever Python is already active. It does not create the virtual environment. If Laya replaces that PyTorch with a CUDA build, the script installs the CPU build again.

Activate the environment before the commands below.

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
source .venv/bin/activate
```

## Score one email

```bash
python scripts/first_decision.py
```

With no argument, the script uses a sample email: a duplicate maintenance-fee debit, a request to reverse it, and a request for a call from the branch. Under the label policy that email is a **complaint**.

Pass your own text in quotes:

```bash
python scripts/first_decision.py "What is the interest rate on a 12-month fixed deposit?"
```

The first run downloads the English checkpoint, `convaiinnovations/laya`. Later runs use the copy in `.cache/huggingface`.

To score with the fine-tuned head instead of the base model:

```bash
python scripts/first_decision.py --checkpoint checkpoints/qrc
```

The script asks the QRC `choice` plus the urgency `score` and the follow-up `noul` described above.

Read `probabilities`, not only the winning label. A low `confidence` means the probability mass is spread across the three labels. The load may warn that this checkpoint's temperatures are outside the range Laya accepts, and that the affected confidence values are uncalibrated. The decision still runs.

## Labels

The full rule is [`data/label_policy.md`](data/label_policy.md).

- **query** — the customer only wants information, such as a rate, a balance, a limit, branch hours, or product terms.
- **request** — the customer asks the bank to do something and is not expressing dissatisfaction. Examples: issue a cheque book, change a mobile number, raise a limit, open a deposit, send a letter.
- **complaint** — the customer is unhappy about a charge, a failed or missing payment, a blocked card, or the service. If they are unhappy and also ask for a fix, the label is still **complaint**.

Each JSONL row is one email:

```json
{"state": "What is the interest rate on a 12-month fixed deposit?", "questions": {"qrc": {"type": "choice", "instructions": "...", "criteria": {"query": "...", "request": "...", "complaint": "..."}}}, "expected": {"qrc": "query"}}
```

`questions.qrc` is the object from `scripts/qrc.py`, copied onto every row so training sees the same criteria text as inference.

[`data/train.jsonl`](data/train.jsonl) has 500 emails: 167 query, 167 request, and 166 complaint. 500 does not divide evenly by three, so complaint has one fewer row. [`data/test.jsonl`](data/test.jsonl) has 24 emails, 8 of each class. The test emails are not in the training file.

## Fine-tune

[`scripts/finetune_qrc.py`](scripts/finetune_qrc.py) calls Laya's own training loop. `freeze_encoder` is false, so the 421M encoder and the decision head are both updated, on CPU.

Laya holds about 10 percent of `data/train.jsonl` out and uses those rows only to fit a confidence temperature. It scores `data/test.jsonl` before training and again after. The test file is never used to update weights.

Count the rows without loading the model:

```bash
python scripts/finetune_qrc.py --dry-run
```

Train:

```bash
python scripts/finetune_qrc.py
```

Weights are written to `checkpoints/qrc`. The script prints accuracy and loss on the test file before and after training. Epochs, batch size, and the loss are set in `config()` in that file.

## API

[`service/main.py`](service/main.py) loads `checkpoints/qrc` once, on CPU, and tags one email. The body is only the email. The response is the winning QRC tag and the probability of each tag.

`POST /qrc`

```json
{"email": "What is the interest rate on a 12-month fixed deposit?"}
```

```json
{"qrc": "query", "scores": {"query": 0.9721, "request": 0.0158, "complaint": 0.0122}}
```

Those scores are from one run of the fine-tuned checkpoint on that fixed-deposit question. `GET /health` returns `{"status": "ok"}` after the model has loaded.

Build the image only after `checkpoints/qrc` exists. The image is CPU-only. It copies that checkpoint and leaves out the duplicate `checkpoint_latest` folder.

```bash
docker build -t laya-qrc .
docker run --rm -p 8000:8000 laya-qrc
```

```bash
curl -s http://127.0.0.1:8000/qrc -H "Content-Type: application/json" -d "{\"email\": \"What is the interest rate on a 12-month fixed deposit?\"}"
```

To run the same app without Docker, install the API packages into the active environment and start it from the project folder:

```bash
python -m pip install -r service/requirements.txt
python -m uvicorn main:app --app-dir service --host 127.0.0.1 --port 8000
```

## Troubleshooting

**`WinError 1114` while loading `c10.dll`.** This Python is Anaconda. Anaconda's Visual C++ runtime is too old for current PyTorch. Use a Python from python.org, create a new virtual environment with it, and run `python scripts/setup.py` again.

**`torch.version.cuda` prints a version such as `13.0`.** The CUDA build was installed. From the same environment, run:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall
```

**Docker build cannot find `checkpoints/qrc`.** Run `python scripts/finetune_qrc.py` first. The weights are not in git.
