"""Run one Laya decision on a banking customer email.

Laya does not write a reply. You give it a state (here, the email text) and
typed questions. One forward pass returns a probability for every option.

Three question types are used so the first run shows each answer shape:

  choice  pick one label, here query / request / complaint
  score   place the email on an ordered scale
  noul    a yes/no question, returned as the probability that the answer is yes
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Keep the downloaded checkpoint inside this repo, on any machine.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import laya

from qrc import QRC_QUESTION

EMAIL = (
    "Hello, my savings account was debited twice for last month's maintenance fee. "
    "This is the second incorrect charge this quarter. Please reverse the duplicate "
    "debit and have someone from the branch call me."
)

# The criteria text is part of the question. Laya scores the email against
# these descriptions. Changing a description changes the decision.
QUESTIONS = {
    "qrc": QRC_QUESTION,
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this email?",
        "criteria": [
            "No deadline. A normal reply about the account is fine.",
            "The customer wants a reply soon.",
            "The customer needs a response today, or money is blocked or a payment has failed.",
        ],
    },
    "wants_a_person": {
        "type": "noul",
        "instructions": "Does the customer ask for a banker or branch staff to contact them?",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify one customer email with the English Laya checkpoint."
    )
    parser.add_argument(
        "email",
        nargs="?",
        default=EMAIL,
        help="Banking email text. If omitted, a sample complaint about a duplicate fee is used.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Folder saved by scripts/finetune_qrc.py. Omit it to use the base English model.",
    )
    args = parser.parse_args()
    email = args.email
    model_id = args.checkpoint or "convaiinnovations/laya"

    print(f"Loading {model_id} on CPU.")
    print("Checkpoint cache:")
    print(f"  {os.environ['HF_HOME']}")
    agent = laya.load(model_id, device="cpu")
    result = agent.predict(email, QUESTIONS)

    answers = result["answers"]
    print()
    print("Email:")
    print(email)
    print()
    print("QRC choice:", answers["qrc"]["choice"])
    print("  probabilities:", json.dumps(answers["qrc"].get("probabilities"), indent=None))
    print("Urgency score:", answers["urgency"]["score"])
    print("  probabilities:", json.dumps(answers["urgency"].get("probabilities"), indent=None))
    print("Wants a person:", answers["wants_a_person"]["noul"])
    print()
    print("Full answers:")
    print(json.dumps(answers, indent=2))


if __name__ == "__main__":
    main()
