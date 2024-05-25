import pandas as pd
import os
import numpy as np

if not os.path.exists('train'):
    os.makedirs('train')
if not os.path.exists('test'):
    os.makedirs('test')

def generate_data(n, noise=0.0, anomaly=False):
    x = np.linspace(0, 10, n)
    y = np.sin(x) + noise * np.random.randn(n)
    if anomaly:
        y[-1] += 5  
    return pd.DataFrame({'x': x, 'y': y})

train_data1 = generate_data(100, noise=0.1)
train_data2 = generate_data(100, noise=0.2, anomaly=True)
train_data = pd.concat([train_data1, train_data2])
train_data.to_csv('train/train_data.csv', index=False)

test_data1 = generate_data(50, noise=0.1)
test_data2 = generate_data(50, noise=0.3, anomaly=True)
test_data = pd.concat([test_data1, test_data2])
test_data.to_csv('test/test_data.csv', index=False)