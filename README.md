# JobBERT Chunk Classifier

Fine-tunes [TechWolf/JobBERT-v3](https://huggingface.co/TechWolf/JobBERT-v3) to classify individual job description chunks as **relevant** or **irrelevant** to the job title.

- **Model:** [AP678/jobbert-job-chunk-classifier](https://huggingface.co/AP678/jobbert-job-chunk-classifier)
- **Dataset:** [AP678/jobbert-chunk-relevance-data](https://huggingface.co/datasets/AP678/jobbert-chunk-relevance-data)

## Results

| Epoch | Val F1 | Test F1 | Precision | Recall |
|-------|--------|---------|-----------|--------|
| 1 | 0.584 | 0.588 | 0.497 | 0.720 |
| 2 | 0.636 | 0.638 | 0.558 | 0.745 |
| 3 | 0.652 | 0.654 | 0.569 | 0.769 |
| 4 | 0.677 | 0.681 | 0.627 | 0.745 |
| **5** | **0.685** | **0.688** | **0.650** | **0.720** |

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install torch transformers datasets huggingface_hub pyarrow scikit-learn
```

## Training

```bash
python train_classifier.py \
    --parts-dir chunks_labelled_qwen_parts \
    --output-dir jobbert_chunk_classifier \
    --epochs 5
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `TechWolf/JobBERT-v3` | Base model |
| `--max-len` | 128 | Token length per chunk |
| `--batch-size` | 32 | Chunks per batch |
| `--lr` | 2e-5 | Learning rate |
| `--epochs` | 5 | Max epochs |
| `--patience` | 3 | Early stopping patience |
| `--push-to-hub` | — | Push best model to HF Hub |
| `--hub-model-id` | — | HF Hub model ID |

## Evaluation

```bash
python eval_epochs.py
```

Evaluates all saved epoch checkpoints against the test split and prints F1, precision, and recall per epoch.

## Inference

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model     = AutoModelForSequenceClassification.from_pretrained("AP678/jobbert-job-chunk-classifier")
tokenizer = AutoTokenizer.from_pretrained("AP678/jobbert-job-chunk-classifier")
model.eval()

chunks = ["We are looking for a Senior Data Engineer...", "We are an equal opportunity employer..."]
enc    = tokenizer(chunks, max_length=128, padding="max_length", truncation=True, return_tensors="pt")

with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1]

for chunk, p in zip(chunks, probs):
    print(f"[{'RELEVANT' if p >= 0.5 else 'irrelevant'}] ({p:.2f})  {chunk}")
```

## Data

The training data is 529k labelled chunks from English job postings, labelled by Qwen. Available on HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("AP678/jobbert-chunk-relevance-data")
```
