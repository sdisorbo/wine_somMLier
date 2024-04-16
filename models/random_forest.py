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

from preprocess import preprocess_and_divide


split, data_under_split, data_over_split = preprocess_and_divide()

# ------------------------PRICE UNDER 100---------------------------

X_lt_100 = data_under_split[['province', 'region_1', 'region_2', 'year']]
y_lt_100 = data_under_split['price']

X_train, X_test, y_train, y_test = train_test_split(X_lt_100, y_lt_100, test_size=0.2, random_state=42)
rf_regressor_under_split = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor_under_split.fit(X_train, y_train)

y_pred = rf_regressor_under_split.predict(X_test).round()
y_test = y_test.values

absolute_errors = np.abs(y_test - y_pred)
medae = np.median(absolute_errors)

print(f'Median Absolute Error: {medae}')

# starts getting bad at 3 std above mean = 146
# 2593/93964 > 100

price_range_edges = (np.linspace(0, split, num=11).round().astype(int))

price_ranges = []
for i in range(len(price_range_edges) - 1):
    price_ranges.append((price_range_edges[i], 
                         price_range_edges[i+1]))


median_errors = []
for price_range in price_ranges:
    range_indices = np.where((y_test >= price_range[0]) & (y_test < price_range[1]))[0]
    if len(range_indices) != 0:
        median_error = np.median(absolute_errors[range_indices])
        median_errors.append(median_error)
    else:
        median_errors.append(0)

norm = Normalize(vmin=min(median_errors), vmax=max(median_errors))
cmap = plt.get_cmap('coolwarm')

fig, ax = plt.subplots(figsize=(14, 6))


bars = ax.bar(range(len(price_ranges)), 
              median_errors, 
              tick_label=[f"{r[0]}-{r[1]}" for r in price_ranges])

for bar, error in zip(bars, median_errors):
    color = cmap(norm(error))
    bar.set_color(color)

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Median Absolute Error')

plt.xlabel('Price Range ($)')
plt.ylabel('Median Absolute Error')
plt.title(f'Median Absolute Error by Price Range < ${split}')
plt.savefig('price_lt_100.png', dpi=300, bbox_inches='tight')

threshold = 10
acc_preds = np.where(absolute_errors <= threshold)[0]
accuracy = len(acc_preds) / len(y_test)

print(f"Accuracy within threshold price < $100 {threshold}: {accuracy}")

# ---------------------------PRICE OVER 100------------------------------

X_over_100 = data_over_split[['province', 'region_1', 'region_2', 'year']]
y_over_100 = data_over_split['price']

X_train, X_test, y_train, y_test = train_test_split(X_over_100, y_over_100, test_size=0.2, random_state=42)
rf_regressor_over_split = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor_over_split.fit(X_train, y_train)

y_pred = rf_regressor_over_split.predict(X_test).round()
y_test = y_test.values

absolute_errors = np.abs(y_test - y_pred)
medae = np.median(absolute_errors)

print(f'Median Absolute Error: {medae}')

# starts getting bad at 3 std above mean = 146
# 2593/93964 > 100

price_range_edges = np.linspace(split, np.max(data_over_split['price']), num=11).round().astype(int)

price_ranges = []
for i in range(len(price_range_edges) - 1):
    price_ranges.append((price_range_edges[i], 
                         price_range_edges[i+1]))

median_errors = []
for price_range in price_ranges:
    range_indices = np.where((y_test >= price_range[0]) & (y_test < price_range[1]))[0]
    if len(range_indices) != 0:
        median_error = np.median(absolute_errors[range_indices])
        median_errors.append(median_error)
    else:
        median_errors.append(0)


norm = Normalize(vmin=min(median_errors), vmax=max(median_errors))
cmap = plt.get_cmap('coolwarm')

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(len(price_ranges)), 
              median_errors, 
              tick_label=[f"{r[0]}-{r[1]}" for r in price_ranges])

for bar, error in zip(bars, median_errors):
    color = cmap(norm(error))
    bar.set_color(color)

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Median Absolute Error')

plt.xlabel('Price Range ($)')
plt.ylabel('Median Absolute Error')
plt.title(f'Median Absolute Error by Price Range > ${split}')
plt.savefig('price_over_100.png', dpi=300, bbox_inches='tight')


threshold = 10
acc_preds = np.where(absolute_errors <= threshold)[0]
accuracy = len(acc_preds) / len(y_test)

print(f"Accuracy within threshold price > $100 {threshold}: {accuracy}")

joblib.dump(rf_regressor_under_split, 'rf_regressor_lt_100.pkl')
joblib.dump(rf_regressor_over_split, 'rf_regressor_over_split.pkl')
