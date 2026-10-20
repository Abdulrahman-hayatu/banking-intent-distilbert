"""Evaluate the fine-tuned DistilBERT on the Banking77 test set."""
from collections import Counter

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train import make_loader  # same tokenization as training

MODEL_DIR = "models/distilbert-banking77"
TOP_N = 10  # rows shown in each error table


def predict(model, loader, device):
    """Return (true labels, predicted labels) for every sample in the loader."""
    model.eval()
    truths, predictions = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            truths.extend(labels.tolist())
    return truths, predictions


def print_worst_classes(truths, predictions, label_names):
    """Show the intents with the lowest F1."""
    _, _, f1_scores, support = precision_recall_fscore_support(
        truths, predictions, labels=range(len(label_names)), zero_division=0)
    worst = sorted(range(len(label_names)), key=lambda i: f1_scores[i])[:TOP_N]
    print(f"\n{TOP_N} weakest intents by F1:")
    for i in worst:
        print(f"  {label_names[i]:45s} F1 {f1_scores[i]:.3f} (n={support[i]})")


def print_top_confusions(truths, predictions, label_names):
    """Show the most frequent (true -> predicted) mistakes."""
    errors = Counter((t, p) for t, p in zip(truths, predictions) if t != p)
    print(f"\n{TOP_N} most frequent confusions (true -> predicted):")
    for (true, pred), count in errors.most_common(TOP_N):
        print(f"  {count:3d}x {label_names[true]} -> {label_names[pred]}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    test_set = load_dataset("legacy-datasets/banking77")["test"]
    label_names = test_set.features["label"].names

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    loader = make_loader(test_set, tokenizer, shuffle=False)

    truths, predictions = predict(model, loader, device)
    print(f"Test size: {len(truths)}")
    print(f"Accuracy: {accuracy_score(truths, predictions):.4f}")
    print(f"Macro-F1: {f1_score(truths, predictions, average='macro'):.4f}")
    print_worst_classes(truths, predictions, label_names)
    print_top_confusions(truths, predictions, label_names)


if __name__ == "__main__":
    main()