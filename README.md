# banking-intent-distilbert

Fine-tuning DistilBERT for intent classification on Banking77 (77 customer-service intents), compared against a TF-IDF + logistic regression baseline.

This repo is my submission for this week's deliverable for the Flexisaf internship. The brief: build two working models that demonstrate advanced deep learning and reinforcement learning, each in its own repo with a Gradio demo and a screen recording. This repo is the deep learning model. The reinforcement learning model is in a separate repo, `lunarlander-dqn-vs-ppo`.

## Models built

| # | Model | Technique |
|---|-------|-----------|
| 1 | TF-IDF + logistic regression | Baseline, not an advanced technique |
| 2 | DistilBERT, fine-tuned | Transfer learning with layer-wise learning-rate decay, a linear warmup and decay schedule, and early stopping |

## Data

Banking77: 10,003 training and 3,080 test messages, 77 intents. It is loaded with `datasets` from the `legacy-datasets/banking77` Parquet copy, because the original `PolyAI/banking77` uses a loading script that `datasets` 5.x no longer supports. For DistilBERT I held out 10% of the training set (1,001 messages, stratified, seed 42) for early stopping and checkpoint selection, which leaves 9,002 for training. The baseline has no tuning, so it trains on all 10,003. The test set was used only to report final numbers, never for tuning or selection.

## How to run

Tested with Python 3.12, transformers 5.20.1 and datasets 5.1.0.

```
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python baseline.py               # TF-IDF + logistic regression: test accuracy and macro-F1
python train.py                  # fine-tunes DistilBERT, saves the best checkpoint
python evaluate.py               # test metrics, weakest intents, most frequent confusions
python app.py                    # Gradio demo
```

I trained on a Colab GPU. On my CPU an epoch took over 10 minutes, so I stopped the local run. `train.py` runs on either.

The trained weights (about 268 MB) are too large for GitHub. Download `distilbert-banking77.zip` from [Google Drive](https://drive.google.com/drive/folders/1t_CAptXSCYD_JZTGZLHFHjYmQKSk9BYV?usp=sharing) and extract it before running `evaluate.py` or `app.py`:

```
python -m zipfile -e distilbert-banking77.zip models/distilbert-banking77
```

## Results

### Test set (3,080 messages)

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| TF-IDF + logistic regression | 0.8571 | 0.8558 |
| DistilBERT | 0.9279 | 0.9280 |

DistilBERT is about 7 points higher on both metrics. A 3,080-message test set has roughly ±1 point of sampling noise at this accuracy (binomial approximation), so the gap is well outside it. I did not run a paired test or repeat training with other seeds.

### Training

Layer-wise decay: the classifier head trains at 5e-5, and each layer below it gets 0.9 times the rate of the one above, down to 2.39e-5 for the embeddings. Warmup covers the first 10% of steps, then the rate decays linearly. Batch size 32, maximum 10 epochs, early-stopping patience 2.

| Epoch | Val accuracy | Val macro-F1 |
|-------|--------------|--------------|
| 1 | 0.6723 | 0.5961 |
| 2 | 0.8781 | 0.8635 |
| 3 | 0.8971 | 0.8951 |
| 4 | 0.9041 | 0.9017 |
| 5 | 0.9031 | 0.9031 |
| 6 | 0.9111 | 0.9085 |
| 7 | 0.9091 | 0.9067 |
| 8 | 0.9131 | 0.9118 |
| 9 | 0.9141 | 0.9127 |
| 10 | 0.9141 | 0.9114 |

- The saved weights are from epoch 9, the best validation macro-F1. Training loss kept falling after epoch 6 (0.062 to 0.019) while validation moved by under half a point. One message is 0.1 point on a 1,001-message validation set, so epochs 8 to 10 are effectively tied.
- Early stopping is implemented but never fired. No two epochs in a row failed to improve, so the run ended at the 10-epoch cap.
- The starting values (5e-5, decay 0.9, warmup 10%, patience 2) were not tuned, and I did not compare against training without decay or warmup. I cannot say how much each element helped.

### Where it fails

Lowest F1 on the test set:

| Intent | F1 |
|--------|----|
| declined_transfer | 0.812 |
| declined_card_payment | 0.822 |
| failed_transfer | 0.829 |
| balance_not_updated_after_bank_transfer | 0.835 |
| verify_my_identity | 0.840 |

The most frequent confusions are `why_verify_identity` → `verify_my_identity` (6), `declined_transfer` → `declined_card_payment` (6), `card_arrival` → `card_delivery_estimate` (4), `fiat_currency_support` → `exchange_via_app` (4) and `top_up_reverted` → `top_up_failed` (4). Most are between intents with near-identical meanings. My reading is that some of the remaining error comes from overlapping intent definitions, not the model, but I did not read the misclassified messages to check.

### Demo app

The app returns the top 5 intents with softmax probabilities. Trials I ran:

| Message | Top prediction | Probability |
|---------|----------------|-------------|
| Why was my transfer declined? | declined_transfer | 99% |
| I want to cancel my card | cancel_transfer | 28% |
| What is my account balance | verify_source_of_funds | 52% |
| What is the weather like tomorrow? | card_delivery_estimate | 86% |

The model always picks one of the 77 intents, and its confidence does not separate in-scope from out-of-scope input: the weather message gets a higher probability than the card message. A confidence threshold would not fix that, and I did not try anything else. I did not check whether Banking77 has intents matching the cancel-card or balance messages.

## A problem I hit

Loading `PolyAI/banking77` failed with "Dataset scripts are no longer supported". My first fix, a `refs/convert/parquet` revision, does not exist for that repo. The `legacy-datasets/banking77` copy loads with the expected 10,003 / 3,080 split sizes and the intent names intact.

## Limitations

- One training run and one seed, with no confidence intervals.
- Learning rate, decay factor and patience are initial values. There is no ablation of the three advanced elements and the baseline is untuned.
- Softmax confidence is not calibrated, and out-of-scope input still gets a confident answer.
- Banking77 is English, short, single-turn, banking-only text, so the results say nothing about other domains.
- The weights are hosted outside the repo.

## Repo layout

```
banking-intent-distilbert/
├── README.md
├── requirements.txt
├── baseline.py      # TF-IDF + logistic regression
├── train.py         # DistilBERT fine-tuning: layer-wise LR decay, warmup, early stopping
├── evaluate.py      # test-set metrics and error analysis
├── app.py           # Gradio demo
├── data/            # empty; datasets downloads and caches the data itself
└── models/          # weights are not in the repo (see How to run)
```

## Author

**Abdulrahman Hayatu Usman** BSc Computer Science — Ahmadu Bello University, Zaria

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/abdulrahman-hayatu) [![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?logo=github&logoColor=white)](https://github.com/Abdulrahman-Hayatu) ![Email](https://img.shields.io/badge/Email-Contact-EA4335?logo=gmail&logoColor=white)