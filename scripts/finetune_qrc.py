"""Fine-tune the English Laya checkpoint on banking QRC emails.

Laya does not learn by generating text. Each training row is an email (the
state), the same QRC choice question used at inference, and the correct label
(query, request, or complaint). One forward pass scores the three labels, and
the loss pushes probability onto the correct one.

What this script does, in order:

1. Read data/train.jsonl. Laya holds about 10 percent of those rows out and
   uses them only to fit a confidence temperature. Those rows are not trained on.
2. Score data/test.jsonl with the untouched base model. That is the "before" accuracy.
3. Train for 4 epochs. freeze_encoder is false, so the 421M encoder and the
   decision head both change. The run is on CPU.
4. Score the same test file again. That is the "after" accuracy. The test emails
   are never in the training file, so this number is not memorization of the
   training rows.
5. Save the result to checkpoints/qrc. Load it later with
   laya.load("checkpoints/qrc") or scripts/first_decision.py --checkpoint.

    python scripts/finetune_qrc.py --dry-run
    python scripts/finetune_qrc.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The English checkpoint is about 800 MB. Keep it inside the repo cache
# instead of the default user cache.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

TRAIN = ROOT / "data" / "train.jsonl"
TEST = ROOT / "data" / "test.jsonl"
OUT = ROOT / "checkpoints" / "qrc"
# Published English model. Fine-tuning starts here and writes a new folder.
BASE = "convaiinnovations/laya"


def config():
    """Training settings passed straight into Laya's own loop.

    This file does not implement the optimizer. laya.train.finetune does.
    """
    from laya.train import TrainConfig

    return TrainConfig(
        # One epoch is one pass over the training rows. Eight passes were
        # enough for the head to move held-out accuracy; two passes were not.
        epochs=4,
        # How many emails go through the model before a gradient is computed.
        # 1 keeps CPU memory small. The full 421M encoder still runs forward.
        micro_batch=2,
        # Gradients are accumulated this many steps, then one weight update
        # is applied. Effective batch size is micro_batch * grad_accum.
        grad_accum=8,
        # Soft cross-entropy: train the head to match the labeled class.
        # Laya's other option, "rlcd", adds a reinforcement term. On their
        # own benchmark that extra term did not raise accuracy.
        loss="soft-ce", 
        # False updates the pretrained encoder as well as the decision head.
        freeze_encoder=False,
        # Held-out file. Laya scores it before training and again after.
        eval_data=str(TEST),
        # Fixes the shuffle of rows so a rerun sees the same order.
        seed=0,
    )


def print_eval(title: str, metrics: dict | None) -> None:
    """Print accuracy on the test emails.

    accuracy is the fraction of emails whose winning label matched the gold
    label. loss is how far the predicted probabilities were from that label.
    A lower loss with the same accuracy means the probabilities moved even
    when the winning label did not.
    """
    if not metrics:
        print(f"{title}: no evaluation")
        return
    print(
        f"{title}: accuracy {metrics.get('accuracy')} "
        f"on {metrics.get('items')} items, loss {metrics.get('loss')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Laya on banking QRC emails.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count rows and items without loading the model or training.",
    )
    args = parser.parse_args()

    # Imported here so --help works even if torch failed to load.
    import laya.train as train

    cfg = config()
    if args.dry_run:
        # Checks that every row became a training item, and prints how many
        # rows will be trained on versus held out for calibration.
        train.dry_run(data=str(TRAIN), model_dir=BASE, config=cfg)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    summary = train.finetune(
        data=str(TRAIN),
        model_dir=BASE,
        output_dir=str(OUT),
        config=cfg,
        device="cpu",
    )
    report = summary["train_report"]
    # "before" is the base checkpoint. "after" is the checkpoint just trained.
    # Both numbers are on data/test.jsonl.
    print_eval("Before", report.get("before"))
    print_eval("After", report.get("after"))
    print("Checkpoint:", summary["output_dir"])
    # train_items: rows the optimizer updated.
    # calibration_items: rows taken out of the training file to fit confidence.
    # eval_items: rows in the test file.
    # is_held_out: True when no test email also appears in training.
    print("Report:", json.dumps({
        "train_items": summary["train_items"],
        "calibration_items": summary["calibration_items"],
        "eval_items": summary["eval_items"],
        "is_held_out": report.get("is_held_out"),
    }))


if __name__ == "__main__":
    main()
