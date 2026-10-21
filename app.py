"""Gradio demo: paste a banking message, get the predicted intent and confidence."""
import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "models/distilbert-banking77"
MAX_LEN = 128  # same as training

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
# int() because label keys loaded from JSON config may be strings
intent_names = {int(i): name for i, name in model.config.id2label.items()}
print("Sample intents:", intent_names[0], "|", intent_names[1])  # expect real names, not LABEL_0


def predict_intent(message):
    """Return {intent: probability} for all 77 intents."""
    inputs = tokenizer(message, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
    probabilities = torch.softmax(logits, dim=-1)[0]
    return {intent_names[i]: float(p) for i, p in enumerate(probabilities)}


demo = gr.Interface(
    fn=predict_intent,
    inputs=gr.Textbox(lines=3, label="Customer message"),
    outputs=gr.Label(num_top_classes=5, label="Predicted intent"),  # shows the top 5 with confidence bars
    title="Banking77 Intent Classifier (DistilBERT)",
    examples=[
        "My card has not arrived yet, when will it get here?",
        "Why was my transfer declined?",
        "I think someone stole my card",
    ],
)

if __name__ == "__main__":
    demo.launch()