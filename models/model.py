import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from preprocess import preprocess_and_divide 
import numpy as np


class BertRegressionModel(nn.Module):
    def __init__(self):
        super(BertRegressionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1) 
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits


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
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation RMSE: {val_rmse}')

    torch.save(model.state_dict(), f'bert_regression_model_{extension}.pth')

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
train_model(model, train_loader_over, val_loader_over, 'over', optimizer)



# def preprocess_data(data):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     encodings = tokenizer(data['description'].tolist(), truncation=True, padding=True, return_tensors='pt')
#     labels = data['price'].values 
#     return TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels, dtype=torch.float))

# def train_bert_regression_model(train_loader, val_loader, config, save_path):
#     model = BertModel.from_pretrained('bert-base-uncased', config=config)
#     model.resize_token_embeddings(len(tokenizer))

#     optimizer = AdamW(model.parameters(), lr=2e-4)
#     loss_fn = torch.nn.MSELoss()

#     model.to(device)

#     num_epochs = 3
#     # save mse values for plotting later
#     mse_values = []
#     for epoch in range(num_epochs):
#         # train model
#         model.train()
#         running_loss = 0.0
#         for i, batch in enumerate(train_loader):
#             input_ids, attention_mask, targets = batch
#             input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)

#             optimizer.zero_grad()
#             outputs = model(input_ids, attention_mask=attention_mask)

#             predictions = outputs.last_hidden_state.mean(dim=1)
#             predictions = torch.mean(predictions, dim=1)
#             predictions = predictions.squeeze()

#             loss = loss_fn(predictions, targets)
#             loss.backward()
#             optimizer.step()

#             print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item()}')


#             # running_loss += loss.item()
#             # if (i + 1) % 100 == 0:
#             #     mse_values.append(running_loss / 100)
#             #     print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100}')
#             #     running_loss = 0.0

#         # backwards validate
#         model.eval()
#         val_loss = 0.0
#         for i, batch in enumerate(val_loader):
#             input_ids, attention_mask, targets = batch
#             input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)
            
#             with torch.no_grad():
#                 outputs = model(input_ids, attention_mask=attention_mask)

#                 predictions = outputs.last_hidden_state.mean(dim=1)
#                 predictions = torch.mean(predictions, dim=1)
#                 predictions = predictions.squeeze()

#                 loss = loss_fn(predictions, targets)
#                 val_loss += loss.item()

#         print(f'Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}')

#     torch.save(model.state_dict(), save_path)

# def evaluate_model(model, dataloader, device):
#     model.eval()
#     total_loss = 0.0
#     running_loss = 0.0
#     true_prices = []
#     predicted_prices = []
#     mse_values = []

#     loss_fn = torch.nn.MSELoss()
#     model.to(device)

#     with torch.no_grad():
#         for batch in dataloader:

#             input_ids, attention_mask, targets = batch
#             input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)

#             outputs = model(input_ids, attention_mask=attention_mask)

#             predictions = outputs.last_hidden_state.mean(dim=1)
#             predictions = torch.mean(predictions, dim=1)
#             predictions = predictions.squeeze()

#             loss = loss_fn(predictions, targets)

#             total_loss += loss.item()

#             true_prices.extend(targets.cpu().numpy())
#             predicted_prices.extend(predictions.cpu().numpy())

#             running_loss += loss.item()
#             if (i + 1) % 100 == 0:
#                 mse_values.append(running_loss / 100)
#                 print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100}')
#                 running_loss = 0.0

#     mae = mean_absolute_error(true_prices, predicted_prices)
#     rmse = mean_squared_error(true_prices, predicted_prices, squared=False)
#     return total_loss / len(dataloader), mae, rmse, true_prices, predicted_prices, mse_values

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# config = BertConfig.from_pretrained('bert-base-uncased')
# config.num_labels = 1 

# _, data_under_split, data_over_split = preprocess_and_divide()

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# train_data_under, val_data_under = train_test_split(data_under_split, test_size=0.2, random_state=42)
# train_data_over, val_data_over = train_test_split(data_over_split, test_size=0.2, random_state=42)

# train_loader_under = DataLoader(preprocess_data(train_data_under), batch_size=16, shuffle=True)
# val_loader_under = DataLoader(preprocess_data(val_data_under), batch_size=16, shuffle=False)

# train_loader_over = DataLoader(preprocess_data(train_data_over), batch_size=16, shuffle=True)
# val_loader_over = DataLoader(preprocess_data(val_data_over), batch_size=16, shuffle=False)


# model_under_path = 'bert_regression_model_under.pth'
# model_over_path = "bert_regression_model_over.pth"

# if os.path.exists(model_under_path) and os.path.exists(model_over_path):
#     # Load the pre-trained models
#     model_under = BertModel.from_pretrained('bert-base-uncased')
#     model_under.load_state_dict(torch.load(model_under_path))
#     model_over = BertModel.from_pretrained('bert-base-uncased')
#     model_over.load_state_dict(torch.load(model_over_path))
#     print("Pre-trained models loaded.")
# else:
#     train_bert_regression_model(train_loader_under, val_loader_under, tokenizer, save_path=model_under_path)
#     train_bert_regression_model(train_loader_over, val_loader_over, tokenizer, save_path=model_over_path)

#     model_under = BertModel.from_pretrained('bert-base-uncased')
#     model_under.load_state_dict(torch.load(model_under_path))
#     model_over = BertModel.from_pretrained('bert-base-uncased')
#     model_over.load_state_dict(torch.load(model_over_path))
#     print("Models trained and saved.")

# # train_bert_regression_model(train_loader_under, val_loader_under, config, save_path=model_under_path)
# # train_bert_regression_model(train_loader_over, val_loader_over, config, save_path=model_over_path)

# # # Load the trained models for evaluation
# # model_under = BertModel.from_pretrained('bert-base-uncased', config=config)
# # model_under.resize_token_embeddings(len(tokenizer))
# # model_under.load_state_dict(torch.load(model_under_path))
# # model_under.to(device)

# # model_over = BertModel.from_pretrained('bert-base-uncased', config=config)
# # model_over.resize_token_embeddings(len(tokenizer))
# # model_over.load_state_dict(torch.load(model_over_path))
# # model_over.to(device)

# eval_loss_under, mae_under, rmse_under, true_prices_under, predicted_prices_under, mse_under_values = evaluate_model(model_under, val_loader_under, device)
# eval_loss_over, mae_over, rmse_over, true_prices_over, predicted_prices_over, mse_over_values = evaluate_model(model_over, val_loader_over, device)

# print(f"Validation Loss for under-sampled data: {eval_loss_under}")
# print(f"MAE for under-sampled data: {mae_under}")
# print(f"RMSE for under-sampled data: {rmse_under}")

# print(f"Validation Loss for over-sampled data: {eval_loss_over}")
# print(f"MAE for over-sampled data: {mae_over}")
# print(f"RMSE for over-sampled data: {rmse_over}")

# # Print true and predicted prices
# print("True and Predicted Prices for under-sampled data:")
# for true_price, pred_price in zip(true_prices_under, predicted_prices_under):
#     print(f"True Price: {true_price}, Predicted Price: {pred_price}")

# print("True and Predicted Prices for over-sampled data:")
# for true_price, pred_price in zip(true_prices_over, predicted_prices_over):
#     print(f"True Price: {true_price}, Predicted Price: {pred_price}")



# plt.plot(range(1, len(mse_under_values) + 1), mse_under_values)
# plt.xlabel('Batches (x100)')
# plt.ylabel('Mean Squared Error')
# plt.title('Mean Squared Error vs. Batches')
# plt.savefig('mse_plot.png')



