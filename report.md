# Training Analysis Report

## Current Model

Fine-tuned `TechWolf/JobBERT-v3` (`XLMRobertaForSequenceClassification`) for binary chunk relevance classification. Each chunk is classified independently as `irrelevant (0)` or `relevant (1)`. No document-level context is used.

### Why JobBERT-v3

JobBERT-v3 is pre-trained on job posting corpora, giving it strong priors on job-domain vocabulary (titles, skills, responsibilities, qualifications). This domain alignment makes it a better backbone than a general-purpose sentence encoder for this task.

### Architecture

```
TechWolf/JobBERT-v3  (XLM-RoBERTa base, 12 layers, hidden=768)
    └── classification head: Linear(768 → 2)
```

The pretrained `pooler` from JobBERT is replaced by a new two-layer classification head (`dense → GELU → dropout → out_proj`) initialised randomly and trained on the labelled chunks.

### Training Setup

| Parameter | Value |
|-----------|-------|
| Base model | `TechWolf/JobBERT-v3` |
| Max token length | 128 |
| Batch size | 32 chunks |
| Learning rate | 2e-5 (AdamW, cosine schedule) |
| Warmup | 10% of steps |
| Weight decay | 0.01 |
| Loss | CrossEntropyLoss with class weights |
| Early stopping patience | 3 epochs |

### Data

370k train / 79k val / 79k test chunks drawn from `chunks_labelled_qwen_parts/` (70/15/15 split at the chunk level, stratified by random permutation).

Class distribution (train): ~20% relevant (1), ~80% irrelevant (0).  
Class weights applied: `w[0]=0.62`, `w[1]=2.52` to compensate for imbalance.

### Results

| Epoch | Val F1 |
|-------|--------|
| 1 | 0.5840 |

Training ongoing.

### Inference

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model     = AutoModelForSequenceClassification.from_pretrained('jobbert_chunk_classifier')
tokenizer = AutoTokenizer.from_pretrained('jobbert_chunk_classifier')

enc   = tokenizer(chunks, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1]  # P(relevant) per chunk
keep  = probs >= 0.5
```

