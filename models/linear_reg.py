import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score

# helper function for measuring accuracy within threshold
def accuracy(y_true, y_pred, threshold):
    correct = 0
    total = len(y_true)
    for true, pred in zip(y_true, y_pred):
        if abs(true - pred) <= threshold:
            correct += 1
    return correct / total

# load validation data
data = pd.read_csv("cleaned_wine_validation_data.csv")

# if data is has price column as empty, exclude from data
data = data[data['price'].notnull()]

print("\n---------Linear Regression with Just Description---------")

# tfidf
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(data['description'])
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# suggests model prediction have avg error of $31.32 and $35.53
train_rmse = root_mean_squared_error(y_train, train_preds)  # 31.32
test_rmse = root_mean_squared_error(y_test, test_preds)  # 35.53

# proportion of variance in price (y) that is explanable by description (X)
# 1 = perfect fit
train_r2 = r2_score(y_train, train_preds)  # 0.2603
test_r2 = r2_score(y_test, test_preds)  # 0.1681

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Train R2 Score:", train_r2)
print("Test R2 Score:", test_r2)

# calculate accuracy within $10
acc = accuracy(y_test, test_preds, threshold=10)
print("Accuracy within $10:", acc)

# calculate accuracy within 10% 
tp = 0.20
acc = accuracy(y_test, test_preds, threshold=tp * y_test.mean())
print("Accuracy within 20%:", acc)



new_descriptions = ["rich and fruity with hints of oak", 
                    "crisp and refreshing with citrus hints of fruit",
                    "rich and refreshing with citrus hints of fruit",
                    '''wine has amazing mix of flavors that dance in the mouth
                    it smells like a mix of blackcurrant cherry and a bit of cedar when you taste it 
                    it feels smooth and the aftertaste lasts a while''',
                    '''symphony of flavors that dance gracefully on the palate
                    nose reveals an elegant blend of blackcurrant cherry and subtle hints of cedar 
                    leading to a palate that boasts velvety tannins and a lingering finish''']
new_X = tfidf_vectorizer.transform(new_descriptions)
predicted_prices = model.predict(new_X)
print("Predicted Prices:", predicted_prices)





# reuse for linear and logistic regression
data['year'] = data['title'].str.extract(r'(\b\d{4}\b)')
years = data['year'][~data['year'].isna()]
avg_year = round(years.astype(float).mean())
data['year'] = data['year'].fillna(f'{avg_year}')

data['all'] = data['description'] + ' ' + \
                data['province'].fillna('') + ' ' + \
                data['year']


print("\n---------Linear Regression with Region and Vintage---------")


print(data['all'][0])

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(data['all'])
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_rmse = root_mean_squared_error(y_train, train_preds)
test_rmse = root_mean_squared_error(y_test, test_preds)

train_r2 = r2_score(y_train, train_preds)  # 0.2998
test_r2 = r2_score(y_test, test_preds)  # 0.2039

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Train R2 Score:", train_r2)
print("Test R2 Score:", test_r2)

acc = accuracy(y_test, test_preds, threshold=10)
print("Accuracy within $10:", acc)

tp = 0.20
acc = accuracy(y_test, test_preds, threshold=tp * y_test.mean())
print("Accuracy within 20%:", acc)



# LOGISTIC REGRESSION WITH REGION AND VINTAGE
# Much better accuracy but loses some precision
print("\n----------Logistic Regression with Region and Vintage----------")

bins = [0, 5, 20, 50, 100, 300, float('inf')]
labels = ['free', 'very cheap', 'moderate', 'expensive', 'very expensive', 'very fancy']

data['price_category'] = pd.cut(data['price'], bins=bins, labels=labels, right=False)

data['description'] = data['description'] + ' ' + \
                      data['province'].fillna('') + ' ' + \
                      data['region_1'].fillna('') + ' ' + \
                      data['region_2'].fillna('') + ' ' + \
                      data['year']

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(data['description'])
y = data['price_category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print(train_preds)

train_accuracy = accuracy_score(y_train, train_preds)
test_accuracy = accuracy_score(y_test, test_preds)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

