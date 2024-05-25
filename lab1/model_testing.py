import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib


preprocessed_test_data = pd.read_csv('test/preprocessed_test_data.csv')


model = joblib.load('model.joblib')


y_pred = model.predict(preprocessed_test_data[['x']])


mse = mean_squared_error(preprocessed_test_data['y'], y_pred)
print(f'Mean squared error: {mse:.4f}')
