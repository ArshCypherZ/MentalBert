"This file is for running the model."

import torch
from transformers import BertTokenizer, BertForSequenceClassification

"""
base model - mental/mental-roberta-base
fine tuned model 1 - arshjaved/MentalBert-Reddit
fine tuned model 2 - arshjaved/MentalBert-Custom
"""

token = "YOUR_HUGGINGFACE_ACCESS_TOKEN"

# you can change model name here
tokenizer = BertTokenizer.from_pretrained("arshjaved/MentalBert-Custom", token=token)
sentiment_model = BertForSequenceClassification.from_pretrained("arshjaved/MentalBert-Custom", token=token)

depression_labels = ["not depressing", "depressing"]

def analyze_text(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Get the logits from the model (output layer)
        outputs = sentiment_model(**inputs)
        logits = outputs.logits

        # Get probabilities and the predicted class (0 or 1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        max_prob, max_index = torch.max(probs, dim=1)

        # Get the corresponding depression class (0: not depressing, 1: depressing)
        depression_class = depression_labels[max_index.item()]

        print("Detected Class:", depression_class)
        print("Confidence Score:", f"{max_prob.item():.4f}")
    except Exception as e:
        print("Error:", str(e))


text_input = "I feel like I'm trapped in a never-ending cycle of sadness, where every day feels heavier than the last."
analyze_text(text_input)
