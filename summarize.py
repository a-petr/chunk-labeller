"""
Inference: chunk a job description and return only title-relevant chunks joined as text.

Usage:
    echo "We are looking for a Senior Data Engineer..." | python summarize.py
    python summarize.py --text "..."
    python summarize.py --model-dir chunk_classifier --threshold 0.45
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from chunking import split_sentences
from train_classifier import JobPostingSummarizer


def load_model(model_dir: str = "chunk_classifier"):
    model = JobPostingSummarizer.from_pretrained(model_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def summarize(
    text: str,
    model: JobPostingSummarizer,
    tokenizer,
    threshold: float | None = None,
) -> str:
    config = model.config
    t = threshold if threshold is not None else config.threshold

    chunks = split_sentences(text)
    if not chunks:
        return ""

    active = chunks[: config.max_chunks]
    n = len(active)

    enc = tokenizer(
        active,
        max_length=config.max_chunk_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    pad = config.max_chunks - n
    input_ids      = F.pad(enc["input_ids"],      (0, 0, 0, pad)).unsqueeze(0)
    attention_mask = F.pad(enc["attention_mask"], (0, 0, 0, pad)).unsqueeze(0)
    chunk_mask     = F.pad(torch.ones(n, dtype=torch.bool), (0, pad)).unsqueeze(0)

    with torch.no_grad():
        out   = model(input_ids, attention_mask, chunk_mask)
        probs = torch.softmax(out.logits[0, :n], dim=-1)[:, 1].numpy()

    return " ".join(c for c, p in zip(active, probs) if p >= t)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",  default="chunk_classifier")
    parser.add_argument("--threshold",  type=float, default=None)
    parser.add_argument("--text",       default=None)
    args = parser.parse_args()

    text = args.text or sys.stdin.read()
    if not text.strip():
        print("No input text.", file=sys.stderr)
        sys.exit(1)

    model, tokenizer = load_model(args.model_dir)
    print(summarize(text, model, tokenizer, threshold=args.threshold))


if __name__ == "__main__":
    main()
