"""TF-IDF + Logistic Regression baseline on Banking77."""
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def load_banking77():
    """Return train and test splits as (texts, labels) tuples."""
    data = load_dataset("legacy-datasets/banking77")
    train = (data["train"]["text"], data["train"]["label"])
    test = (data["test"]["text"], data["test"]["label"])
    return train, test


def main():
    (train_texts, train_labels), (test_texts, test_labels) = load_banking77()
    print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")

    # Unigrams + bigrams; sublinear_tf dampens the effect of repeated words
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)  # reuse the vocabulary fitted on train

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, train_labels)

    predictions = model.predict(x_test)
    print(f"Accuracy: {accuracy_score(test_labels, predictions):.4f}")
    print(f"Macro-F1: {f1_score(test_labels, predictions, average='macro'):.4f}")


if __name__ == "__main__":
    main()