"""The QRC choice shared by the demo and the fine-tuning data.

The criteria text is part of the question. Training and inference must use the same words.
"""

QRC_QUESTION = {
    "type": "choice",
    "instructions": "What kind of customer email is this?",
    "criteria": {
        "query": "The customer is asking for information about an account, product, rate, or policy, and is not asking the bank to change anything.",
        "request": "The customer is asking the bank to do something, such as reverse a charge, transfer funds, replace a card, or change a limit.",
        "complaint": "The customer is unhappy about a charge, a failed payment, or the service on their account.",
    },
}
