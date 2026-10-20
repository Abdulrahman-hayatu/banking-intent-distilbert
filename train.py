"""Fine-tune DistilBERT on Banking77: layer-wise LR decay, warmup, early stopping."""
import argparse

import torch
from datasets import load_dataset
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_NAME = "distilbert-base-uncased"
OUT_DIR = "models/distilbert-banking77"
NUM_LAYERS = 6          # transformer layers in DistilBERT
MAX_LEN = 128           # messages are short; longer ones get truncated
BATCH_SIZE = 32
MAX_EPOCHS = 10
PATIENCE = 2            # stop after this many epochs without validation gain
BASE_LR = 5e-5          # LR of the classifier head
LR_DECAY = 0.9          # LR multiplier for each step further from the head
WARMUP_FRACTION = 0.1   # share of training steps spent warming up
SEED = 42


def make_loader(dataset, tokenizer, shuffle):
    """Tokenize a split and wrap it in a DataLoader."""
    # list() because datasets 5.x may return a lazy column object
    encoded = tokenizer(list(dataset["text"]), truncation=True, max_length=MAX_LEN,
                        padding="max_length", return_tensors="pt")
    tensors = TensorDataset(encoded["input_ids"], encoded["attention_mask"],
                            torch.tensor(list(dataset["label"])))
    return DataLoader(tensors, batch_size=BATCH_SIZE, shuffle=shuffle)


def layer_depth(param_name):
    """Distance from the head: head=0, top layer=1, ..., embeddings=NUM_LAYERS+1."""
    if param_name.startswith("distilbert.embeddings"):
        return NUM_LAYERS + 1
    if param_name.startswith("distilbert.transformer.layer."):
        layer_index = int(param_name.split(".")[3])  # e.g. distilbert.transformer.layer.3.attention...
        return NUM_LAYERS - layer_index
    return 0  # pre_classifier and classifier


def build_optimizer(model):
    """AdamW with one group per depth; LR shrinks toward the embeddings."""
    groups = {}
    for name, param in model.named_parameters():
        groups.setdefault(layer_depth(name), []).append(param)
    return torch.optim.AdamW(
        [{"params": params, "lr": BASE_LR * LR_DECAY ** depth} for depth, params in groups.items()]
    )


def train_one_epoch(model, loader, optimizer, scheduler, device):
    """One pass over the training data; returns mean loss."""
    model.train()
    total_loss = 0.0
    for input_ids, attention_mask, labels in loader:
        optimizer.zero_grad()
        output = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                       labels=labels.to(device))
        output.loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += output.loss.item()
    return total_loss / len(loader)


def evaluate_model(model, loader, device):
    """Return (accuracy, macro-F1) on a loader."""
    model.eval()
    predictions, truths = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            truths.extend(labels.tolist())
    accuracy = sum(p == t for p, t in zip(predictions, truths)) / len(truths)
    return accuracy, f1_score(truths, predictions, average="macro")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny 1-epoch run to check the script works")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data = load_dataset("legacy-datasets/banking77")
    label_names = data["train"].features["label"].names
    # Hold out 10% of train for early stopping; the test set stays untouched
    split = data["train"].train_test_split(test_size=0.1, seed=SEED, stratify_by_column="label")
    train_set, val_set = split["train"], split["test"]
    epochs = MAX_EPOCHS
    if args.smoke:
        train_set, val_set = train_set.select(range(200)), val_set.select(range(100))
        epochs = 1
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),  # saved with the model so app.py can read intent names
        label2id={name: i for i, name in enumerate(label_names)},
    ).to(device)

    train_loader = make_loader(train_set, tokenizer, shuffle=True)
    val_loader = make_loader(val_set, tokenizer, shuffle=False)

    optimizer = build_optimizer(model)
    print("Group LRs:", [f"{group['lr']:.2e}" for group in optimizer.param_groups])  # sanity check
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_FRACTION * total_steps), total_steps)

    best_f1, epochs_without_gain = 0.0, 0
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        accuracy, macro_f1 = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch}: loss {loss:.4f} | val acc {accuracy:.4f} | val macro-F1 {macro_f1:.4f}")

        if macro_f1 > best_f1:  # keep only the best checkpoint on disk
            best_f1, epochs_without_gain = macro_f1, 0
            model.save_pretrained(OUT_DIR)
            tokenizer.save_pretrained(OUT_DIR)
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= PATIENCE:
                print("Early stopping")
                break

    print(f"Best val macro-F1: {best_f1:.4f} | saved to {OUT_DIR}")


if __name__ == "__main__":
    main()