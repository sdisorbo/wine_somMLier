import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from preprocess import preprocess_and_divide
from sklearn.metrics import mean_squared_error, accuracy_score

# Define a function to preprocess data
def load_model(model_path):
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.num_labels = 1
    model = BertModel.from_pretrained('bert-base-uncased', config=config)
    model.load_state_dict(torch.load(model_path))
    return model

def tokenize(data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(data['description'].tolist(), truncation=True, padding=True, return_tensors='pt')
    labels = data['price'].values 
    return TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels, dtype=torch.float))

# Define a function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    real_prices = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, targets = batch
            input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions.extend(outputs.last_hidden_state.mean(dim=1).mean(dim=1).squeeze().cpu().numpy())
            real_prices.extend(targets.cpu().numpy())
    return predictions, real_prices


model_path = "bert_regression_model_under.pth"
model = load_model(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

_, _, test_data = preprocess_and_divide()
test_data = test_data[test_data['price'] < 36.0]
test_data.sample(n=5000, random_state=42)
test_loader = DataLoader(tokenize(test_data), batch_size=16, shuffle=False)

predictions, real_prices = evaluate_model(model, test_loader)

print("Real Price\tPredicted Price")
for real_price, predicted_price in zip(real_prices, predictions):
    print(f"{real_price}\t\t{predicted_price}")


accuracy = accuracy_score(real_prices, predictions)
print("Accuracy:", accuracy)

mse = mean_squared_error(real_prices, predictions)
print(f"Mean Squared Error: {mse}")
