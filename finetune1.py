# Fine Tuned Model 1

import pandas as pd
import re
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.nn.utils import clip_grad_norm_
from accelerate import Accelerator
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv('/datasets/dataset1.csv')
token = "YOUR_HUGGINGFACE_ACCESS_TOKEN"

# Check the structure of dataset
print(data.head())

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#','', text)  # Remove mentions and hashtags
    text = text.lower()  # Lowercase
    return text

data['cleaned_text'] = data['clean_text'].apply(preprocess_text)

le = LabelEncoder()
data['label'] = le.fit_transform(data['is_depression'])

X_train, X_test, y_train, y_test = train_test_split(data['cleaned_text'], data['label'], test_size=0.1, random_state=42) # testing 10%

tokenizer = BertTokenizer.from_pretrained("mental/mental-bert-base-uncased", token=token)

class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        return {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze(), 'label': torch.tensor(label)}

train_dataset = CustomDataset(X_train.tolist(), y_train.tolist())
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Device management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BertForSequenceClassification.from_pretrained("mental/mental-bert-base-uncased", num_labels=len(le.classes_), token=token)
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)

accelerator = Accelerator()  # Enable mixed precision
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

# Training Loop
EPOCHS = 3
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        if torch.isnan(input_ids).any() or torch.isinf(input_ids).any() or torch.isnan(attention_mask).any() or torch.isinf(attention_mask).any():
            print("NaN or inf detected in input_ids or attention_mask, skipping batch")
            continue

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        accelerator.backward(loss)
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}')

# Create a test dataset and dataloader for evaluation
test_dataset = CustomDataset(X_test.tolist(), y_test.tolist())
test_loader = DataLoader(test_dataset, batch_size=8)

# Evaluate
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device)
        }
        outputs = model(**inputs)
        logits = outputs.logits
        predictions.extend(logits.argmax(dim=-1).cpu().numpy())

# Compare predictions with true labels
print(classification_report(y_test, predictions))


# Upload trained model to huggingface
"""
model.save_pretrained("/content/MentalBert-Custom")
tokenizer.save_pretrained("/content/MentalBert-Custom")
from huggingface_hub import HfApi

api = HfApi(token=token)
api.upload_folder(
    folder_path="/content/MentalBert-Custom",
    repo_id="username/reponame",
    commit_message="Initial model upload",
    repo_type="model",
)
"""