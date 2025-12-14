import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Load data
data = pd.read_csv("data/wave_data.csv", sep='\s+', comment='#', header=None,
                   names=['YY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE'])

# Replace placeholder values with NaN and interpolate
data.replace({99.00: np.nan, 999: np.nan, 999.0: np.nan}, inplace=True)
wave = data[['WVHT']].interpolate(method='linear').ffill().bfill()

# 2. Normalize data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(wave)

# 3. Create sequences
def create_seq(data, steps=10):
    X, y = [], []
    for i in range(len(data) - steps):
        X.append(data[i:i+steps])
        y.append(data[i+steps])
    return np.array(X), np.array(y)

X, y = create_seq(scaled, 10)

# 4. Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = X_train.reshape(X_train.shape[0], 10, 1)
X_test = X_test.reshape(X_test.shape[0], 10, 1)

# 5. Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(10,1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 6. Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# 7. Save model with native Keras format
model.save("model/wave_height_lstm_model.keras")
print("Model trained and saved")
