#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import GPT2Tokenizer, GPT2Model
import os

path = kagglehub.dataset_download("suchintikasarkar/sentiment-analysis-for-mental-health")

print("Path to dataset files:", path)


# In[ ]:


file_name = "Combined Data.csv"
file_path = os.path.join(path, file_name)

print(f"Attempting to load data from: {file_path}")

try:
    # Load the dataset into a pandas DataFrame
    df = pd.read_csv(file_path)

    print("\n--- 1. DataFrame Head (First 5 Rows) ---")
    print(df.head())

    print("\n--- 2. DataFrame Information (Columns, Non-null counts, Dtypes) ---")
    print(df.info())

    print("\n--- 3. Dataset Shape (Rows, Columns) ---")
    print(df.shape)

    print("\n--- 4. Target Variable Distribution (Sentiment/Label) ---")
    try:
        target_column = 'label'
        print(df[target_column].value_counts())
        print(f"\nTarget Variable Unique Values: {df[target_column].nunique()}")
    except KeyError:
        print("Could not find a 'label' column. Please check the actual column names from .head()")
        # Fallback: check all unique values in all object columns
        print("\nObject (Text) Column Unique Values Check:")
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() < 20:
                print(f"Column '{col}' value counts:\n{df[col].value_counts()}\n")

    print("\n--- 5. Check for Missing Values ---")
    print(df.isnull().sum())
except FileNotFoundError:
    print(f"\nError: File not found at {file_path}. Please check the downloaded path and file name.")
except Exception as e:
    print(f"\nAn error occurred during data loading: {e}")


# In[ ]:


# --- 1. Load Data ---

file_path = os.path.join(path, "Combined Data.csv")
df = pd.read_csv(file_path)

# Drop the irrelevant index column
df = df.drop(columns=['Unnamed: 0'])

# --- 2. Data Cleaning and Preprocessing ---

# Handle missing values by dropping rows where 'statement' is NaN
# Since only 362 out of 53043 are missing, dropping them is generally safe.
df.dropna(subset=['statement'], inplace=True)
print(f"Data shape after handling missing values: {df.shape}")

# Convert all text to string (ensures robustness)
df['statement'] = df['statement'].astype(str)

# --- 3. Label Encoding for the Target Variable (status) ---

# Initialize LabelEncoder
le = LabelEncoder()
# Fit and transform the 'status' column
df['status_encoded'] = le.fit_transform(df['status'])

# Display the mapping
print("\n--- Label Encoding Mapping ---")
for i, label in enumerate(le.classes_):
    print(f"{label}: {i}")

# --- 4. GPT-2 Feature Extraction Function ---

def get_gpt2_embeddings(texts, model, tokenizer, batch_size=32):
    """Generates sentence embeddings using GPT-2 by mean pooling."""

    # Set pad token for GPT-2 (it doesn't have one by default)
    # Using the EOS token as PAD token is a common workaround for feature extraction
    tokenizer.pad_token = tokenizer.eos_token

    # Place model in evaluation mode and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_embeddings = []

    # Process texts in batches to manage memory
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize the batch
        # Padding='longest' for dynamic padding, Truncation=True to prevent sequence overflow
        # return_tensors='pt' returns PyTorch tensors
        encoded_input = tokenizer(
            batch_texts.tolist(),
            padding='longest',
            truncation=True,
            max_length=128,  # Choose a max length based on data, 128 is common
            return_tensors='pt'
        )

        # Move tensors to the appropriate device
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)

        with torch.no_grad():
            # Get the model output
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            # The last hidden state contains the token embeddings
            embeddings = output.last_hidden_state  # Shape: (Batch Size, Sequence Length, Hidden Dimension)

        # --- Mean Pooling to get Sentence Vector ---
        # Multiply embeddings by the attention mask to zero out padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask_expanded

        # Sum all non-padding tokens
        summed_embeddings = torch.sum(masked_embeddings, 1)

        # Calculate the actual number of tokens (non-padding)
        summed_mask = torch.clamp(attention_mask.sum(1), min=1e-9)  # Avoid division by zero

        # Divide sum by count to get the mean (mean pooling)
        mean_pooled_embeddings = (summed_embeddings / summed_mask.unsqueeze(-1)).cpu().numpy()

        all_embeddings.append(mean_pooled_embeddings)

    return np.concatenate(all_embeddings, axis=0)

# Initialize GPT-2 components
print("\n--- Loading GPT-2 Model and Tokenizer (This may take a moment) ---")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# Generate the feature vectors (X)
X_features = get_gpt2_embeddings(df['statement'], model, tokenizer, batch_size=64)
Y_labels = df['status_encoded'].values

print(f"\nGenerated Feature Matrix X shape: {X_features.shape}")
print(f"Generated Label Vector Y shape: {Y_labels.shape}")

# --- 5. Data Splitting and Logistic Regression ---

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X_features, Y_labels, test_size=0.2, random_state=42, stratify=Y_labels
)

print(f"\nTraining on {X_train.shape[0]} samples, Testing on {X_test.shape[0]} samples.")

# Initialize and train Logistic Regression model
print("\n--- Training Logistic Regression Model ---")
# Use a high max_iter and appropriate solver for large datasets/features
lr_model = LogisticRegression(max_iter=500, solver='sag', multi_class='multinomial', random_state=42)
lr_model.fit(X_train, Y_train)

# --- 6. Model Evaluation ---

# Predict on the test set
Y_pred = lr_model.predict(X_test)

print("\n--- Evaluation Results (Logistic Regression with GPT-2 Features) ---")
print(classification_report(Y_test, Y_pred, target_names=le.classes_))

