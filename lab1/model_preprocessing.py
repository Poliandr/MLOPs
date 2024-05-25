import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load data
train_data = pd.read_csv('train/train_data.csv')
test_data = pd.read_csv('test/test_data.csv')

# Preprocess data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data[['x']])
y_train = train_data['y'].to_numpy()
X_test = scaler.transform(test_data[['x']])
y_test = test_data['y'].to_numpy()

# Save preprocessed data
preprocessed_train_data = pd.DataFrame(np.hstack((X_train, y_train.reshape(-1, 1))), columns=['x', 'y'])
preprocessed_test_data = pd.DataFrame(np.hstack((X_test, y_test.reshape(-1, 1))), columns=['x', 'y'])
preprocessed_train_data.to_csv('train/preprocessed_train_data.csv', index=False)
preprocessed_test_data.to_csv('test/preprocessed_test_data.csv', index=False)
