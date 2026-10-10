FROM python:3.11-slim-bookworm

WORKDIR /app

# CPU build of PyTorch. The default PyPI wheel is a CUDA build this service does not use.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir laya

COPY service/requirements.txt /app/service/requirements.txt
RUN pip install --no-cache-dir -r /app/service/requirements.txt

COPY scripts/qrc.py /app/qrc.py
COPY service/main.py /app/main.py
COPY checkpoints/qrc /app/checkpoints/qrc

ENV CHECKPOINT=/app/checkpoints/qrc
ENV HF_HUB_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
