import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaModel
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

# Load and preprocess data
print('reading csv...')
train_data = pd.read_csv("C:/Users/Owner/OneDrive/Documents/Eecs 448/Project/cleaned_wine_training_data.csv")
train_data = train_data.dropna(subset=['price'])
print(f"Training Data shape: {train_data.shape}")
test_data = pd.read_csv("C:/Users/Owner/OneDrive/Documents/Eecs 448/Project/cleaned_wine_training_data.csv")
test_data = test_data.dropna(subset=['price'])
print(f"Testing Data shape: {test_data.shape}")

# Tokenize
tokenizer = RobertaTokenizer.from_pretrained('FacebookAI/roberta-base')
encoded_corpus = tokenizer(text = train_data.description.tolist(),
                           add_special_tokens=True,
                           padding='max_length',
                           truncation='longest_first',
                           max_length=300,
                           return_attention_mask=True)
train_inputs = encoded_corpus['input_ids']
train_masks = encoded_corpus['attention_mask']
train_labels = train_data.price.to_numpy(dtype=float)

test_corpus = tokenizer(text = test_data.description.tolist(),
                           add_special_tokens=True,
                           padding='max_length',
                           truncation='longest_first',
                           max_length=300,
                           return_attention_mask=True)
test_inputs = test_corpus['input_ids']
test_masks = test_corpus['attention_mask']
test_labels = test_data.price.to_numpy(dtype=float)

# Data Scaling
price_scaler = StandardScaler()
price_scaler.fit(train_labels.reshape(-1, 1))
train_labels = price_scaler.transform(train_labels.reshape(-1, 1))
test_labels = price_scaler.transform(test_labels.reshape(-1, 1))

# Data Loaders
def create_dataloaders(inputs, masks, labels):
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    label_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor, label_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True) # or batch=16
    return dataloader

train_loader = create_dataloaders(train_inputs, train_masks, train_labels)
test_loader = create_dataloaders(test_inputs, test_masks, test_labels)

# Class
class RobertaRegressor(nn.Module):
    def __init__(self, drop_rate=0.2):
        super(RobertaRegressor, self).__init__()
        D_in, D_out = 768, 1
        self.roberta = RobertaModel.from_pretrained('FacebookAI/roberta-base')
        self.regressor = nn.Sequential(nn.Dropout(drop_rate),
                                       nn.Linear(D_in, D_out))
        self.double() # or self.double()

    def forward(self, input_ids, attention_masks):
        outputs = self.roberta(input_ids, attention_masks)
        class_label_output = outputs[1]
        outputs = self.regressor(class_label_output)
        return outputs

# Model
model = RobertaRegressor()
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
loss_fn = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

# Train
def train(model, optimizer, loss_fn, epochs, train_loader, device):
    for epoch in range(epochs):
        print(f"Training Epoch {epoch+1}...")
        model.train()
        # for step, batch in enumerate(train_loader):
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # print(step)
            batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
            model.zero_grad()
            outputs = model(batch_inputs, batch_masks)
            loss = loss_fn(outputs.squeeze(), batch_labels.squeeze())
            loss.backward()
            optimizer.step()
    return model

epochs = 1
model = train(model, optimizer, loss_fn, epochs, train_loader, device)

# Evaluation
def evaluate(model, loss_fn, test_loader, device):
    model.eval()
    test_loss, test_r2 = [], []
    for batch in test_loader:
        batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
        with torch.no_grad():
            outputs = model(batch_inputs, batch_masks)
        loss = loss_fn(outputs, batch_labels)
        test_loss.append(loss.item())
        r2 = r2_score(outputs, batch_labels)
        test_r2.append(r2.item())
    return test_loss, test_r2

def r2_score(outputs, labels):
    labels_mean = torch.mean(labels)
    ss_tot = torch.sum((labels - labels_mean) ** 2)
    ss_res = torch.sum((labels - outputs) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

test_loss, test_r2 = evaluate(model, loss_fn, test_loader, device)
print(f"Test Loss: {test_loss[0]:.4f}")
print(f"Test R^2: {test_r2[0]:.6f}")

# Predict
def predict(model, data_loader, device):
    model.eval()
    output = []
    for batch in data_loader:
        batch_inputs, batch_masks, _ = tuple(b.to(device) for b in batch)
        with torch.no_grad():
            output += model(batch_inputs, batch_masks).view(1, -1).tolist()[0]
    return output

# Validation
val_data = pd.read_csv("C:/Users/Owner/OneDrive/Documents/Eecs 448/Project/cleaned_wine_validation_data.csv")
val_data = val_data.dropna(subset=['price'])

val_corpus = tokenizer(text = val_data.description.tolist(),
                           add_special_tokens=True,
                           padding='max_length',
                           truncation='longest_first',
                           max_length=300,
                           return_attention_mask=True)
val_inputs = val_corpus['input_ids']
val_masks = val_corpus['attention_mask']
val_labels = val_data.price.to_numpy(dtype=float)
val_labels = price_scaler.transform(val_labels.reshape(-1, 1))
val_loader = create_dataloaders(val_inputs, val_masks, val_labels)

val_pred_scaled = predict(model, val_loader, device)
val_pred = price_scaler.inverse_transform(val_pred_scaled)

# Summary
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

mae = mean_absolute_error(val_labels, val_pred)
mdae = median_absolute_error(val_labels, val_pred)
mse = mean_squared_error(val_labels, val_pred)
mape = mean_absolute_percentage_error(val_labels, val_pred)
mdape = ((pd.Series(val_labels) - pd.Series(val_pred))\
         / pd.Series(val_labels)).abs().median()
r_squared = r2_score(val_labels, val_pred)

print(f"Mean Absolute Error: {mae:.4f}")
print(f"Median Absolute Error: {mdae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Percentage Error: {mape:.4f}")
print(f"Median Absolute Percentage Error: {mdape:.4f}")
print(f"R^2 Score: {r_squared:.6f}")