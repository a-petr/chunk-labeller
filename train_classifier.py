from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import torch
from sklearn.metrics import classification_report, f1_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


class ChunkDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_len: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex["label"], dtype=torch.long),
        }


def load_and_split_examples(
    parts_dir: Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    table = ds.dataset(parts_dir, format="parquet").to_table()
    rows = table.to_pylist()

    examples = []
    for row in rows:
        for c in row["chunks"]:
            if c["label"] >= 0:
                examples.append({"text": c["chunk_text"], "label": int(c["label"])})

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(examples))

    n = len(idx)
    t = int(n * train_frac)
    v = int(n * (train_frac + val_frac))

    return (
        [examples[i] for i in idx[:t]],
        [examples[i] for i in idx[t:v]],
        [examples[i] for i in idx[v:]],
    )


def compute_class_weights(examples: list[dict]) -> torch.Tensor:
    labels = [e["label"] for e in examples]
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, class_weights):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, steps = 0.0, 0

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    for batch in loader:
        ids = batch["input_ids"].to(device)
        masks = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(device_type=device, dtype=amp_dtype, enabled=(device in {"cuda", "cpu"})):
            out = model(input_ids=ids, attention_mask=masks)
            loss = loss_fn(out.logits, labels)

        total_loss += loss.item()
        steps += 1
        preds = out.logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    f1 = f1_score(all_labels, all_preds, average="binary")
    return f1, total_loss / max(steps, 1), all_preds, all_labels


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="TechWolf/JobBERT-v3")
    parser.add_argument("--max-len", type=int, default=128)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--parts-dir", default="chunks_labelled_qwen_parts")
    parser.add_argument("--output-dir", default="jobbert_chunk_classifier")
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)

    args = parser.parse_args()

    seed_everything(args.seed)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    use_autocast = device in {"cuda", "cpu"}
    use_scaler = device == "cuda"

    print(f"Device: {device}  AMP dtype: {amp_dtype}  seed={args.seed}")

    parts_dir = Path(args.parts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("Loading examples…")
    train_ex, val_ex, test_ex = load_and_split_examples(parts_dir, seed=args.seed)
    print(f"  train={len(train_ex):,}  val={len(val_ex):,}  test={len(test_ex):,}")

    train_pos = sum(e["label"] for e in train_ex)
    print(f"  train label distribution: 1={train_pos:,}  0={len(train_ex) - train_pos:,}")

    class_weights = compute_class_weights(train_ex)
    print(f"  class weights: {class_weights.tolist()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(output_dir)

    train_loader = DataLoader(
        ChunkDataset(train_ex, tokenizer, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        ChunkDataset(val_ex, tokenizer, args.max_len),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        ChunkDataset(test_ex, tokenizer, args.max_len),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("Loading model…")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    total_opt_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_opt_steps * args.warmup_ratio),
        num_training_steps=total_opt_steps,
    )
    scaler = GradScaler(enabled=use_scaler)

    best_val_f1 = 0.0
    best_epoch = 1
    epochs_no_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            ids = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type=device, dtype=amp_dtype, enabled=use_autocast):
                out = model(input_ids=ids, attention_mask=masks)
                loss = loss_fn(out.logits, labels) / args.accum_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * args.accum_steps

            if step % args.accum_steps == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 200 == 0:
                print(f"  epoch {epoch + 1}  step {step}/{len(train_loader)}  loss={running_loss / 200:.4f}")
                running_loss = 0.0

        if len(train_loader) % args.accum_steps != 0:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        val_f1, val_loss, _, _ = evaluate(model, val_loader, device, amp_dtype, class_weights)
        print(f"epoch {epoch + 1}  val F1={val_f1:.4f}  val loss={val_loss:.4f}")

        epoch_dir = output_dir / f"epoch_{epoch + 1}"
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        print(f"  → saved to {epoch_dir}/")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            epochs_no_improvement = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  → new best epoch {best_epoch}")
        else:
            epochs_no_improvement += 1
            print(f"  → no improvement ({epochs_no_improvement}/{args.patience})")
            if epochs_no_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch + 1}")
                break

    print(f"\nBest epoch: {best_epoch}  val F1={best_val_f1:.4f}")
    print("Evaluating on test set…")

    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    test_f1, test_loss, preds, labels = evaluate(model, test_loader, device, amp_dtype, class_weights)
    print(f"Test F1={test_f1:.4f}  loss={test_loss:.4f}")
    print(classification_report(labels, preds, target_names=["irrelevant", "relevant"]))
    print(f"\nModel saved to: {output_dir}/")

    if args.push_to_hub:
        from huggingface_hub import HfApi
        hub_id = args.hub_model_id or output_dir.name
        print(f"Pushing to HuggingFace Hub as '{hub_id}'…")
        HfApi().upload_folder(folder_path=str(output_dir), repo_id=hub_id, repo_type="model")
        print(f"  → https://huggingface.co/{hub_id}")


if __name__ == "__main__":
    main()
