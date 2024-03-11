import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

# Load and preprocess data
print('reading csv...')
data = pd.read_csv("cleaned_wine_validation_data.csv")
data = data.dropna(subset=['price'])
print('cutting labels...')
data['price_category'] = pd.cut(data['price'], bins=[0, 20, 50, 100, float('inf')],
                                 labels=['cheap', 'moderate', 'expensive', 'very expensive'])

X_train, X_val, y_train, y_val = train_test_split(data['description'], data['price_category'],
                                                  test_size=0.2, random_state=42)
print('tokenizing...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
val_encodings = tokenizer(list(X_val), truncation=True, padding=True)

# Convert labels to numerical values
label_map = {'cheap': 0, 'moderate': 1, 'expensive': 2, 'very expensive': 3}
train_labels = [label_map[label] for label in y_train]
val_labels = [label_map[label] for label in y_val]

train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']),
                              torch.tensor(train_encodings['attention_mask']),
                              torch.tensor(train_labels))

val_dataset = TensorDataset(torch.tensor(val_encodings['input_ids']),
                            torch.tensor(val_encodings['attention_mask']),
                            torch.tensor(val_labels))

print('creating dataloader...')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
optimizer = AdamW(model.parameters(), lr=1e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print('training...')
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(train_loader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f'Batch {i}: {loss}')

    # Validation
    model.eval()
    val_losses = []
    val_preds = []
    val_true = []
    for i, batch in enumerate(val_loader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)  # Include labels during inference
            loss = outputs.loss
            logits = outputs.logits
            val_losses.append(loss.item())
            val_preds.extend(torch.argmax(logits, axis=1).cpu().numpy())
            val_true.extend(labels.cpu().numpy())
            print(f'Batch {i}: {loss}')


    val_accuracy = (sum([1 if p == t else 0 for p, t in zip(val_preds, val_true)]) / len(val_true)) * 100
    print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {sum(val_losses) / len(val_losses)}, '
          f'Validation Accuracy: {val_accuracy}%')

# Epoch 2/3, Validation Loss: 0.8391911613059716, Validation Accuracy: 62.99559471365639%
# Epoch 3/3, Validation Loss: 0.8728375508331917, Validation Accuracy: 61.43171806167401%

# import pandas as pd
# import torch
# import os
# from sklearn.model_selection import train_test_split
# from transformers import BertTokenizer, BertModel, BertConfig
# from torch.utils.data import DataLoader, TensorDataset
# import torch.nn as nn
# import torch.optim as optim



# data = pd.read_csv("cleaned_wine_validation_data.csv")
# data = data.dropna(subset=['price'])

# print('splitting data...')
# X_train, X_val, y_train, y_val = train_test_split(data['description'], data['price'],
#                                                   test_size=0.2, random_state=42)

# saved_model = "bert_regression_model.pth"

# if os.path.exists(saved_model):
#     model = torch.load(saved_model)
# else:
#     # bert tokenize
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
#     val_encodings = tokenizer(list(X_val), truncation=True, padding=True)

#     # create PyTorch dataloaders
#     train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']),
#                                 torch.tensor(train_encodings['attention_mask']),
#                                 torch.tensor(y_train.values, dtype=torch.float))

#     val_dataset = TensorDataset(torch.tensor(val_encodings['input_ids']),
#                                 torch.tensor(val_encodings['attention_mask']),
#                                 torch.tensor(y_val.values, dtype=torch.float))

#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#     print('configuring bert...')
#     # pretrained bert model
#     config = BertConfig.from_pretrained('bert-base-uncased')
#     config.num_labels = 1
#     model = BertModel.from_pretrained('bert-base-uncased', config=config)

#     # head to replace classification with regression
#     class RegressionHead(nn.Module):
#         def __init__(self, hidden_size):
#             super().__init__()
#             self.dense = nn.Linear(hidden_size, 1)

#         def forward(self, features):
#             x = self.dense(features)
#             return x

#     # replace bert's default classification head with custom regression head
#     model.resize_token_embeddings(len(tokenizer))
#     model.pooler = nn.Sequential()
#     model.cls = RegressionHead(config.hidden_size)
#     torch.save(model, saved_model)

# optimizer = optim.AdamW(model.parameters(), lr=2e-4)
# loss_fn = nn.MSELoss()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# print(len(train_loader))

# num_epochs = 3
# for epoch in range(num_epochs):
#     # forwards training
#     model.train()
#     running_loss = 0.0
#     for i, batch in enumerate(train_loader):
#         input_ids, attention_mask, targets = tuple(t.to(device) for t in batch)
#         optimizer.zero_grad()
#         outputs = model(input_ids, attention_mask=attention_mask)
#         # predictions = outputs.last_hidden_state.mean(dim=1)
#         # predictions = torch.mean(predictions, dim=1)
#         predictions = outputs.pooler_output.squeeze()
#         # predictions = predictions.squeeze()
#         loss = loss_fn(predictions, targets)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * input_ids.size(0)
#         print(f'Batch {i}: {running_loss}')

#     epoch_loss = running_loss / len(train_loader.dataset)
#     print(epoch_loss)

#     # backwards validation
#     model.eval()
#     val_loss = 0.0
#     for i, batch in enumerate(val_loader):
#         input_ids, attention_mask, targets = tuple(t.to(device) for t in batch)
#         with torch.no_grad():
#             outputs = model(input_ids, attention_mask=attention_mask)
#             predictions = outputs.last_hidden_state.mean(dim=1)
#             predictions = torch.mean(predictions, dim=1)
#             predictions = predictions.squeeze()
#             loss = loss_fn(predictions, targets)
#             val_loss += loss.item() * input_ids.size(0)
#             print(f'Batch {i}: {val_loss}')

#     epoch_val_loss = val_loss / len(val_loader.dataset)

#     print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
