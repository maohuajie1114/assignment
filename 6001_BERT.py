#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import kagglehub
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tqdm import tqdm

path = kagglehub.dataset_download("suchintikasarkar/sentiment-analysis-for-mental-health")

print("Path to dataset files:", path)


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import AdamW
from tqdm.notebook import tqdm

file_name = "Combined Data.csv"
file_path = os.path.join(path, file_name)

print(f"Attempting to load data from: {file_path}")

# --- 1. Load Data ---
try:
    df = pd.read_csv(file_path)
    df = df.drop(columns=['Unnamed: 0'])
    df.dropna(subset=['statement'], inplace=True)
except Exception:
    print("Could not load data from path, please ensure file path is correct.")

# --- 2. Label Encoding ---
le = LabelEncoder()
df['status_encoded'] = le.fit_transform(df['status'])

# Get the number of unique classes
NUM_LABELS = len(le.classes_)

# Map for decoding results later
label_map = {i: label for i, label in enumerate(le.classes_)}
print(f"Number of classes (NUM_LABELS): {NUM_LABELS}")

# --- 3. Data Splitting ---
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['status_encoded']
)

print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")


# In[ ]:


# --- 4. Custom Dataset Class for BERT ---

# Initialize BERT Tokenizer
PRETRAINED_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
MAX_LEN = 128  # Set maximum sequence length, adjust based on data analysis


class MentalHealthDataset(Dataset):
    """A custom PyTorch Dataset for loading and tokenizing the text data."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# In[ ]:


# --- 5. Model Initialization ---

# Create Dataset and DataLoader
train_dataset = MentalHealthDataset(
    train_df['statement'],
    train_df['status_encoded'],
    tokenizer,
    MAX_LEN
)

val_dataset = MentalHealthDataset(
    val_df['statement'],
    val_df['status_encoded'],
    tokenizer,
    MAX_LEN
)

BATCH_SIZE = 16  # Adjust based on GPU memory
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT with a classification layer configured for 7 labels
# Note: Ensure NUM_LABELS is correctly defined from the previous steps
try:
    if 'NUM_LABELS' not in locals():
        # Fallback in case previous cells were not executed in the same kernel session
        NUM_LABELS = df['status_encoded'].nunique()
        PRETRAINED_MODEL_NAME = 'bert-base-uncased'
except NameError:
    print("Error: 'df' or 'NUM_LABELS' is not defined. Please run Part 1 and Part 2 cells first.")
    # Exit or define placeholders
    exit()

model = BertForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL_NAME,
    num_labels=NUM_LABELS
)
model = model.to(device)

# --- 6. Optimizer and Parameters ---
EPOCHS = 3
LEARNING_RATE = 2e-5

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
print("\nOptimizer AdamW initialized successfully.")

# --- 7. Training Loop Function ---
def train_epoch(model, data_loader, optimizer, device, n_examples):
    """Performs one epoch of training."""
    model = model.train()
    losses = []
    correct_predictions = 0

    # Use tqdm for progress bar
    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        # Calculate accuracy
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

# --- 8. Evaluation Function ---
def eval_model(model, data_loader, device):
    """Evaluates the model on the validation set."""
    model = model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculations
        # Use tqdm for progress bar
        for d in tqdm(data_loader, desc="Validation"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            all_labels.extend(labels.cpu().tolist())

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(all_labels), np.mean(losses), all_labels, all_preds

# --- 9. Main Training Loop Execution and Tracking ---

print("\n--- Starting BERT Fine-tuning ---")
best_accuracy = 0
history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    # Training
    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        optimizer,
        device,
        len(train_df)
    )
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc.item())

    print(f"Train loss {train_loss:.4f} | Accuracy {train_acc:.4f}")

    # Validation
    val_acc, val_loss, Y_true, Y_pred = eval_model(
        model,
        val_data_loader,
        device
    )
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc.item())

    print(f"Val loss   {val_loss:.4f} | Accuracy {val_acc:.4f}")

    # Save the best model
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_bert_model.bin')
        best_accuracy = val_acc
        print(f"Model saved! New best accuracy: {best_accuracy:.4f}")

# --- 10. Visualization of Training History ---

def plot_history(history):
    """Plots training and validation loss and accuracy over epochs."""
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


print("\n--- Plotting Training History ---")
plot_history(history)

# --- 11. Final Evaluation ---

# Load the best model weights for final evaluation
model.load_state_dict(torch.load('best_bert_model.bin'))
print("\n--- Final Evaluation on Validation Set (Using Best Model) ---")

# Run evaluation with the best model to get the final predictions
_, _, Y_true, Y_pred = eval_model(
    model,
    val_data_loader,
    device
)

# Print detailed classification report
# Note: Ensure the 'label_map' and 'target_names_list' are correctly defined from Part 1
target_names_list = [label_map[i] for i in range(NUM_LABELS)]
print("\n--- Classification Report (Best Model) ---")
print(classification_report(Y_true, Y_pred, target_names=target_names_list))

