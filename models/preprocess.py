from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
# from sklearn.externals import joblib
import joblib



def preprocess_and_divide():
    data1 = pd.read_csv('cleaned_wine_testing_data.csv')
    data1 = data1.dropna(subset=['price'])

    data2 = pd.read_csv('cleaned_wine_validation_data.csv')
    data2 = data2.dropna(subset=['price'])

    data = pd.concat([data1, data2], ignore_index=True)

    # second standard deviation
    std2_bottom = data['price'].quantile(0.003)
    std2_top = data['price'].quantile(0.997)

    # boxplot
    # plt.figure(figsize=(8, 6))
    # plt.boxplot(data['price'], vert=False)
    # plt.xlabel('Price')
    # plt.title('Box Plot of Wine Prices')
    # plt.show()

    outliers = data[(data['price'] < std2_bottom) | (data['price'] > std2_top)]
    print(len(outliers))

    # remove outliers
    data = data[(data['price'] >= std2_bottom) & (data['price'] <= std2_top)]


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
    data['year'] = data['year'].fillna(f'{avg_year}')
    
    # first std
    split = data['price'].quantile(0.683)

    data_under_split = data[data['price'] < split]

    data_over_split = data[data['price'] >= split]

    return split, data_under_split, data_over_split

