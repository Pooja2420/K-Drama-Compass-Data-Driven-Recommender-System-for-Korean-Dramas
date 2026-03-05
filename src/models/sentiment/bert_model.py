"""
BERT sentiment classifier.
Fine-tunes bert-base-uncased on K-Drama reviews for 3-class sentiment.
  0 = Negative  (overall_score < 4)
  1 = Neutral   (4 ≤ overall_score < 7)
  2 = Positive  (overall_score ≥ 7)

Achieved ~70% accuracy on a 10% sample in the original notebook.

Artifacts saved to: models/artifacts/bert_sentiment/
"""

from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

from src.utils.logger import get_logger

logger = get_logger("bert_model")

ARTIFACT_DIR = Path("models/artifacts/bert_sentiment")
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 3
MAX_LENGTH = 128


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ReviewDataset(Dataset):
    """PyTorch Dataset wrapping tokenised review texts and sentiment labels."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: BertTokenizer,
        max_length: int = MAX_LENGTH,
    ):
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train(
    texts: list[str],
    labels: list[int],
    sample_frac: float = 0.1,
    epochs: int = 1,
    batch_size: int = 16,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Trainer, BertTokenizer, dict]:
    """
    Fine-tune BERT on review texts.

    Args:
        texts        : list of raw review strings
        labels       : list of integer sentiment labels (0/1/2)
        sample_frac  : fraction of data to use (default 0.1 for speed)
        epochs       : training epochs
        batch_size   : per-device batch size
        test_size    : train/test split ratio
        random_state : reproducibility seed

    Returns:
        (trainer, tokenizer, metrics_dict)
    """
    # Sub-sample for manageable training time
    import pandas as pd

    df = pd.DataFrame({"text": texts, "label": labels}).sample(
        frac=sample_frac, random_state=random_state
    )
    texts_s = df["text"].tolist()
    labels_s = df["label"].tolist()

    logger.info(
        f"Training BERT on {len(texts_s)} samples "
        f"(sample_frac={sample_frac}, epochs={epochs})"
    )

    # Train / test split
    tr_texts, te_texts, tr_labels, te_labels = train_test_split(
        texts_s, labels_s, test_size=test_size, random_state=random_state
    )

    # Tokenizer & model
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    train_ds = ReviewDataset(tr_texts, tr_labels, tokenizer)
    test_ds = ReviewDataset(te_texts, te_labels, tokenizer)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(ARTIFACT_DIR / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=str(ARTIFACT_DIR / "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )

    trainer.train()

    # Evaluate
    preds_output = trainer.predict(test_ds)
    preds = torch.argmax(torch.tensor(preds_output.predictions), dim=1).numpy()
    acc = accuracy_score(te_labels, preds)
    report = classification_report(
        te_labels, preds, target_names=["Negative", "Neutral", "Positive"]
    )

    logger.info(f"BERT Accuracy: {acc:.4f}")
    logger.info(f"\n{report}")

    metrics = {"accuracy": acc, "classification_report": report}
    return trainer, tokenizer, metrics


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_model(trainer: Trainer, tokenizer: BertTokenizer) -> Path:
    out = ARTIFACT_DIR / "model"
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    logger.info(f"BERT model saved to {out}")
    return out


def load_model() -> tuple[BertForSequenceClassification, BertTokenizer]:
    model_path = str(ARTIFACT_DIR / "model")
    logger.info(f"Loading BERT model from {model_path}")
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict(
    texts: list[str],
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
) -> list[int]:
    """Run inference on a list of texts. Returns list of label ints."""
    ds = ReviewDataset(texts, [0] * len(texts), tokenizer)
    trainer = Trainer(model=model)
    output = trainer.predict(ds)
    return torch.argmax(torch.tensor(output.predictions), dim=1).tolist()
