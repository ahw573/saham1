!pip install yfinance numpy pandas scikit-learn tensorflow streamlit matplotlib

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Unduh data saham
symbol = 'AAPL'
data = yf.download(symbol, start='2020-01-01', end='2023-01-01')

# Pra-pemrosesan data
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()
data['Daily Return'] = data['Close'].pct_change()
data.dropna(inplace=True)

# Normalisasi data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Membuat dataset untuk LSTM
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Membangun model LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, batch_size=1, epochs=1)

# Prediksi menggunakan model
train_predict = model.predict(X)
train_predict = scaler.inverse_transform(train_predict)

# Hitung MSE
mse = mean_squared_error(data['Close'][time_step + 1:], train_predict)
print(f"Mean Squared Error: {mse}")

# Plot hasil prediksi
plt.figure(figsize=(10, 6))
plt.plot(data['Close'][time_step + 1:], label='Harga Aktual')
plt.plot(data.index[time_step + 1:], train_predict, label='Prediksi LSTM')
plt.title(f'Prediksi Harga Saham {symbol}')
plt.xlabel('Tanggal')
plt.ylabel('Harga Penutupan')
plt.legend()
plt.show()

import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

st.title('Aplikasi Analisis Saham')

# Input simbol saham
symbol = st.text_input('Masukkan simbol saham (contoh: AAPL):', 'AAPL')

# Unduh data
data = yf.download(symbol, start='2020-01-01', end='2023-01-01')

# Tampilkan grafik harga
st.write(f"### Grafik Harga Saham {symbol}")
plt.figure(figsize=(10, 6))
plt.plot(data['Close'])
plt.title(f'Harga Saham {symbol}')
plt.xlabel('Tanggal')
plt.ylabel('Harga Penutupan')
st.pyplot(plt)
