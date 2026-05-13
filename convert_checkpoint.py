"""
Convert the old-format checkpoint (model.pt + hc_config.json) produced by
the first training run into the HuggingFace save_pretrained() format.

Usage:
    python convert_checkpoint.py
    python convert_checkpoint.py --input chunk_classifier --output chunk_classifier
"""
import argparse
import json
from pathlib import Path

import torch

from train_classifier import JobPostingSummarizer, JobPostingSummarizerConfig


def convert(input_dir: str, output_dir: str) -> None:
    src = Path(input_dir)
    dst = Path(output_dir)

    old_pt     = src / "model.pt"
    old_config = src / "hc_config.json"

    if not old_pt.exists():
        raise FileNotFoundError(f"{old_pt} not found — nothing to convert.")
    if not old_config.exists():
        raise FileNotFoundError(f"{old_config} not found.")

    raw = json.loads(old_config.read_text())
    config = JobPostingSummarizerConfig(
        encoder_name=raw["encoder_name"],
        num_cross_layers=raw["num_cross_layers"],
        dropout=raw["dropout"],
        max_chunks=raw["max_chunks"],
        max_chunk_len=raw["max_chunk_len"],
        threshold=raw["threshold"],
        # class_weights not stored in old format; restore balanced defaults
        class_weights=raw.get("class_weights", [0.62, 2.62]),
    )

    print(f"Loading weights from {old_pt}…")
    model = JobPostingSummarizer(config)
    state = torch.load(old_pt, map_location="cpu")
    model.load_state_dict(state, strict=False)

    print(f"Saving HF-format model to {dst}/")
    dst.mkdir(exist_ok=True)
    model.save_pretrained(dst)

    # Copy tokenizer files (already in the same dir)
    import shutil
    for fname in ("tokenizer.json", "tokenizer_config.json",
                  "vocab.txt", "special_tokens_map.json"):
        f = src / fname
        if f.exists() and dst != src:
            shutil.copy(f, dst / fname)

    # Remove old artefacts so from_pretrained() doesn't get confused
    if dst == src:
        old_pt.unlink(missing_ok=True)
        old_config.unlink(missing_ok=True)
        print("Removed model.pt and hc_config.json.")

    print("Done. Load with:")
    print(f"  JobPostingSummarizer.from_pretrained('{dst}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="chunk_classifier")
    parser.add_argument("--output", default="chunk_classifier")
    args = parser.parse_args()
    convert(args.input, args.output)
