import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.linear_model import LinearRegression

def create_sequences(data, seq_length=60):
    X, y = [],[]
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_lstm(X_train, y_train, epochs=20, optimizer='adam', loss='mean_squared_error', batch_size=32):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    lstm_model.compile(optimizer=optimizer, loss=loss)
    lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return lstm_model
    
def build_linear_regression(X_train, y_train):
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    return lin_model

def make_hybrid_predictions(lstm_predictions, lin_predictions):
    predictions = (0.7 * lstm_predictions) + (0.3 * lin_predictions)
    return predictions