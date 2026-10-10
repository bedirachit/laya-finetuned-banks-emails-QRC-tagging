"""Score one banking email with the fine-tuned QRC checkpoint.

The caller sends only the email text. The QRC question is fixed here, the same
wording the model was trained on. The response is the winning tag and the
probability of each tag.

    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import laya
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Docker places qrc.py beside this file. A local run keeps it in scripts/.
_HERE = Path(__file__).resolve().parent
for _folder in (_HERE, _HERE.parent / "scripts"):
    if (_folder / "qrc.py").is_file():
        sys.path.insert(0, str(_folder))
        break
from qrc import QRC_QUESTION

CHECKPOINT = os.environ.get("CHECKPOINT", "checkpoints/qrc")
# Only the QRC choice is exposed. Urgency and follow-up stay out of this API.
QUESTIONS = {"qrc": QRC_QUESTION}


class EmailRequest(BaseModel):
    email: str = Field(min_length=1, description="The customer email to tag.")


class QrcResponse(BaseModel):
    qrc: str
    scores: dict[str, float]


agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    # Load once. The first prediction otherwise pays the model-build cost.
    agent = laya.load(CHECKPOINT, device="cpu")
    yield


app = FastAPI(title="Banking QRC", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/qrc", response_model=QrcResponse)
def tag_email(body: EmailRequest) -> QrcResponse:
    text = body.email.strip()
    if not text:
        raise HTTPException(status_code=422, detail="email is empty")
    answer = agent.predict(text, QUESTIONS)["answers"]["qrc"]
    scores = {label: round(float(prob), 4) for label, prob in answer["probabilities"].items()}
    return QrcResponse(qrc=answer["choice"], scores=scores)
