from sklearn.preprocessing import LabelEncoder
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(description):

    description = description.lower()
    description = description.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(description)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    result = ' '.join(words)
    result = re.sub(r'[0-9]', '', result)
    return result


def preprocess_and_divide():
    data1 = pd.read_csv('cleaned_wine_testing_data.csv')
    data1 = data1.dropna(subset=['price'])

    data2 = pd.read_csv('cleaned_wine_validation_data.csv')
    data2 = data2.dropna(subset=['price'])

    data = pd.concat([data1, data2], ignore_index=True)

    # third standard deviation
    std3_bottom = data['price'].quantile(0.003)
    std3_top = data['price'].quantile(0.997)


    outliers = data[(data['price'] < std3_bottom) | (data['price'] > std3_top)]
    print(len(outliers))

    # remove outliers
    data = data[(data['price'] >= std3_bottom) & (data['price'] <= std3_top)]


    data['province'] = data['province'].fillna('Unknown')
    data['region_1'] = data['region_1'].fillna('Unknown')
    data['region_2'] = data['region_2'].fillna('Unknown')

    label_encoder = LabelEncoder()
    data['province'] = label_encoder.fit_transform(data['province'])
    data['region_1'] = label_encoder.fit_transform(data['region_1'])
    data['region_2'] = label_encoder.fit_transform(data['region_2'])


    # treat missing values as a separate category, suitable for categorical variables 
    # where the absence of a value could carry some meaning
    # encode missing values as a new category before applying label encoding

    data['year'] = data['title'].str.extract(r'(\b\d{4}\b)')
    years = data['year'][~data['year'].isna()]
    avg_year = round(years.astype(float).mean())
    print('Average year across testing+validation test set: ', avg_year)
    data['year'] = data['year'].fillna(f'{avg_year}')
    
    # first std
    split = data['price'].quantile(0.683)

    data_under_split = data[data['price'] < split]

    data_over_split = data[data['price'] >= split]

    return split, data_under_split, data_over_split

