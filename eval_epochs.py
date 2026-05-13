from pathlib import Path

import torch
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_classifier import ChunkDataset, load_and_split_examples

device = "mps" if torch.backends.mps.is_available() else "cpu"

_, _, test_ex = load_and_split_examples(Path("chunks_labelled_qwen_parts"), seed=42)
print(f"Test examples: {len(test_ex):,}\n")

for epoch in range(1, 5):
    ckpt = f"jobbert_chunk_classifier/epoch_{epoch}"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)
    model.eval()

    loader = DataLoader(
        ChunkDataset(test_ex, tokenizer, 128),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            probs = torch.softmax(model(input_ids=ids, attention_mask=masks).logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(batch["labels"].tolist())

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    f1 = f1_score(all_labels, preds, average="binary")
    p  = precision_score(all_labels, preds, average="binary")
    r  = recall_score(all_labels, preds, average="binary")
    print(f"Epoch {epoch}  F1={f1:.4f}  Precision={p:.4f}  Recall={r:.4f}")
    print(classification_report(all_labels, preds, target_names=["irrelevant", "relevant"]))
    del model
