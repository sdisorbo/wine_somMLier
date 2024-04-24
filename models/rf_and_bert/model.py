import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.rf_and_bert.preprocess_data import preprocess_and_divide 
from BRModel import BertRegressionModel


def train_model(model, train_loader, val_loader, extension, optimizer, num_epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = nn.MSELoss()(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        val_preds = []
        val_true = []
        for i, batch in enumerate(val_loader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                val_losses.append(nn.MSELoss()(outputs, labels.unsqueeze(1)).item())
                val_preds.extend(outputs.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        val_rmse = mean_squared_error(val_true, val_preds, squared=False)
        val_mae = mean_absolute_error(val_true, val_preds)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation RMSE: {val_rmse}, Validation MAE: {val_mae}')

    # torch.save(model.state_dict(), f'bert_regression_model_{extension}.pth')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

_, data_under_split, data_over_split = preprocess_and_divide()

train_under, val_under = train_test_split(data_under_split, test_size=0.2, random_state=42)
train_over, val_over = train_test_split(data_over_split, test_size=0.2, random_state=42)

train_encodings_under = tokenizer(list(train_under['description']), truncation=True, padding=True)
val_encodings_under = tokenizer(list(val_under['description']), truncation=True, padding=True)

train_encodings_over = tokenizer(list(train_over['description']), truncation=True, padding=True)
val_encodings_over = tokenizer(list(val_over['description']), truncation=True, padding=True)

train_labels_under = torch.tensor(train_under['price'].values, dtype=torch.float32)
val_labels_under = torch.tensor(val_under['price'].values, dtype=torch.float32)

train_labels_over = torch.tensor(train_over['price'].values, dtype=torch.float32)
val_labels_over = torch.tensor(val_over['price'].values, dtype=torch.float32)

train_dataset_under = TensorDataset(torch.tensor(train_encodings_under['input_ids']),
                                    torch.tensor(train_encodings_under['attention_mask']),
                                    train_labels_under)

val_dataset_under = TensorDataset(torch.tensor(val_encodings_under['input_ids']),
                                torch.tensor(val_encodings_under['attention_mask']),
                                val_labels_under)

train_dataset_over = TensorDataset(torch.tensor(train_encodings_over['input_ids']),
                                torch.tensor(train_encodings_over['attention_mask']),
                                train_labels_over)

val_dataset_over = TensorDataset(torch.tensor(val_encodings_over['input_ids']),
                                torch.tensor(val_encodings_over['attention_mask']),
                                val_labels_over)

train_loader_under = DataLoader(train_dataset_under, batch_size=16, shuffle=True)
val_loader_under = DataLoader(val_dataset_under, batch_size=16, shuffle=False)

train_loader_over = DataLoader(train_dataset_over, batch_size=16, shuffle=True)
val_loader_over = DataLoader(val_dataset_over, batch_size=16, shuffle=False)


model = BertRegressionModel()
optimizer = AdamW(model.parameters(), lr=1e-5)

train_model(model, train_loader_under, val_loader_under, 'under', optimizer)

model = BertRegressionModel()
optimizer = AdamW(model.parameters(), lr=1e-5)

train_model(model, train_loader_over, val_loader_over, 'over', optimizer)


