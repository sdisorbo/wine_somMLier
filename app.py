from flask import Flask, request, render_template
import gensim
import tensorflow as tf  # Or import torch for PyTorch
import torch
import random
import pandas as pd
import numpy as np
import xgboost as xgb

#split the data into training and testing data
from sklearn.model_selection import train_test_split


app = Flask(__name__)

# Load models (Assuming they're already trained and saved)
w2v = gensim.models.Word2Vec.load("/Users/samdisorbo/Documents/code/python_projects/eecs448/wine/wine_word2vec_model")


model = xgb.XGBRegressor()
model.load_model("/Users/samdisorbo/Documents/code/python_projects/eecs448/wine/xgboost_wine.json")

#load the training data cleaned_wine_training_data
training_data = pd.read_csv("/Users/samdisorbo/Documents/code/python_projects/eecs448/wine/cleaned_wine_training_data.csv")

columns = [
    'country_Argentina', 'country_Australia', 'country_Austria', 'country_Brazil', 'country_Bulgaria',
    'country_Canada', 'country_Chile', 'country_Croatia', 'country_Cyprus', 'country_Czech Republic',
    'country_Egypt', 'country_England', 'country_France', 'country_Georgia', 'country_Germany',
    'country_Greece', 'country_Hungary', 'country_Israel', 'country_Italy', 'country_Lebanon',
    'country_Luxembourg', 'country_Mexico', 'country_Moldova', 'country_Morocco', 'country_New Zealand',
    'country_Peru', 'country_Portugal', 'country_Romania', 'country_Slovenia', 'country_South Africa',
    'country_Spain', 'country_Turkey', 'country_US', 'country_Ukraine', 'country_Uruguay'
] + [f'description_{i}' for i in range(100)]  # Adding description columns from 0 to 99

# Creating an empty DataFrame with these columns
pred= pd.DataFrame(columns=columns)

#add a single row of false values to the dataFrame for the country columns and NA values for the description columns
pred.loc[0] = False
for i in range(100):
    pred[f'description_{i}'] = np.nan



#model.eval()

@app.route('/', methods=['GET', 'POST'])
def home():
    vocab = list(w2v.wv.key_to_index.keys())
    if request.method == 'POST':
        description = request.form['description']
        location = request.form['location']
        actual_price = int(request.form['actual_price'])
        #for the words in description, if they are not in the vocab then remove them
        description = description.split()
        description = [word for word in description if word in vocab]
        #convert the words to vectors
        description = [w2v.wv[word] for word in description]
        #average the vectors for each word in the description
        description = np.mean(description, axis=0)
        #go through the dataframe pred columns and if the location is in the column name then set the value to true, otherwise set the value to false
        for column in pred.columns:
            if location in column:
                pred[column] = True
            else:
                pred[column] = False
        #set the description columns to the description values
        for i in range(100):
            pred[f'description_{i}'] = description[i]
       
        price_prediction = round(model.predict(pred)[0])
        


        return render_template('result.html', price=price_prediction, actual_price=actual_price)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
