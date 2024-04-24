from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from models.rf_and_bert.preprocess_data import preprocess_text
from BRModel import BertRegressionModel
import joblib


print('loading BERTs...')

bert_under_model = torch.load('bert_regression_model_under_2.pth')
bert_over_model = torch.load('bert_regression_model_over_2.pth')

print('done loading BERTs!')

model_under = BertRegressionModel()
model_over = BertRegressionModel()

model_under.load_state_dict(bert_under_model)
model_over.load_state_dict(bert_over_model)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_bert_predictions(text):
    print('entered get bert_predictions')

    text = preprocess_text(text)  

    tokenized_text = tokenizer(text, truncation=True, padding=True, return_tensors='pt')

    input_data = {
        'input_ids': tokenized_text['input_ids'],
        'attention_mask': tokenized_text['attention_mask']
    }

    input_ids = torch.tensor(input_data['input_ids'])
    attention_mask = torch.tensor(input_data['attention_mask'])

    print('running models...')

    with torch.no_grad():
        price_under = model_under(input_ids, attention_mask=attention_mask)[0].item()
        price_over = model_over(input_ids, attention_mask=attention_mask)[0].item()
    
    
    return price_under, price_over


rf_regressor_under_split = joblib.load('rf_regressor_under_split.pkl')
rf_regressor_over_split = joblib.load('rf_regressor_over_split.pkl')


# 2011 = average year 
def average_predictions(description, 
                      year=2011, 
                      province='Unknown', 
                      region_1='Unknown', 
                      region_2='Unknown'):
    
    # get predictions from BERT
    price_bert_under, price_bert_over = get_bert_predictions(description)
    print(price_bert_under, price_bert_over)
    
    # extract features for random forest
    rf_wine_data = pd.DataFrame({
        'year': [year],
        'province': [province],
        'region_1': [region_1],
        'region_2': [region_2]
    })

    label_encoder = LabelEncoder()
    rf_wine_data['province'] = label_encoder.fit_transform(rf_wine_data['province'])
    rf_wine_data['region_1'] = label_encoder.fit_transform(rf_wine_data['region_1'])
    rf_wine_data['region_2'] = label_encoder.fit_transform(rf_wine_data['region_2'])

    
    price_rf_under = rf_regressor_under_split.predict(rf_wine_data.values.reshape(1, -1))
    price_rf_over = rf_regressor_over_split.predict(rf_wine_data.values.reshape(1, -1))

    # weigh the two models based on MAE 

    bert_rmse = 5.319
    rf_rmse = 5.024

    inv_bert = 1 / bert_rmse
    inv_rf = 1 / rf_rmse

    scale = inv_bert + inv_rf

    norm_bert = inv_bert / scale
    norm_rf = inv_rf / scale


    final_price_under = price_bert_under * norm_bert + price_rf_under * norm_rf
    final_price_over = price_bert_over * norm_bert + price_rf_over * norm_rf
    
    return final_price_under[0], final_price_over[0]

# example 
description = "a cool vintage syrah, very dry and tough while having a flavorful tannin astringent \
    smells of ripe blackberry with dark chocolate"
year = 2007
province = "California"
region_1 = "Oak Knoll District"
region_2 = "Napa"

print('Predicting...')

final_prediction_under, final_prediction_over = average_predictions(description)

print("Final Prediction Under:", final_prediction_under)
print("Final Prediction Over:", final_prediction_over)

