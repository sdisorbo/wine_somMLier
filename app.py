from flask import Flask, request, render_template
import gensim
import tensorflow as tf  # Or import torch for PyTorch
import random

app = Flask(__name__)

# Load models (Assuming they're already trained and saved)
word2vec_model = gensim.models.Word2Vec.load("/Users/samdisorbo/Documents/code/python_projects/eecs448/wine/wine_word2vec_model")
#nn_model = tf.keras.models.load_model("wine_price_prediction_model.h5")  # Adjust for PyTorch

@app.route('/', methods=['GET', 'POST'])
def home():
    vocab = list(word2vec_model.wv.key_to_index.keys())
    if request.method == 'POST':
        description = request.form['description']
        year = request.form['year']
        location = request.form['location']
        actual_price = int(request.form['actual_price'])
       
        vector = [word2vec_model.wv[word] for word in description.split() if word in vocab]

        price_prediction = 74


        return render_template('result.html', price=price_prediction, actual_price=actual_price)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
