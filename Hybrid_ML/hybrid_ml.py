# importing the necessary Python libraries and the dataset
import pandas as pd
import numpy as np
data = pd.read_csv('apple_stock_data.csv')
print(data.head())

# As the dataset is based on stock market data, 
# I’ll convert the date column to a datetime type, 
# set it as the index, and focus on the Close price
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data[['Close']]
print(data.head())

"""
We will be using LSTM (Long Short-Term Memory) and Linear Regression models for this task. 
I chose LSTM because it effectively captures sequential dependencies and patterns in time-series data, 
which makes it suitable for modelling stock price movements influenced by historical trends.


Linear Regression, on the other hand, is a straightforward model that captures simple linear relationships 
and long-term trends in data. By combining these two models into a hybrid approach, 
we leverage the LSTM’s ability to model complex time-dependent patterns alongside the Linear Regression’s 
ability to identify and follow broader trends. This combination aims to create a more balanced and accurate prediction system.
"""

# let’s scale the Close price data between 0 and 1 using MinMaxscalar to ensure compatibility with the LSTM model
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler(feature_range=(0, 1))
data['Close'] = scalar.fit_transform(data[['Close']])

# Prepare the data for LSTM by creating sequences of a defined length (e.g., 60 days) to predict the next day’s price
from useful_functions import create_sequences
seq_length = 60
X, y = create_sequences(data['Close'].values, seq_length)

# split the sequences into training and test sets (e.g., 80% training, 20% testing)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# sequential LSTM model with layers to capture the temporal dependencies in the data
from useful_functions import build_lstm
lstm_model = build_lstm(X_train, y_train)

# let’s train the second model. 
# I’ll start by generating lagged features for Linear Regression 
# (e.g., using the past 3 days as predictors)
data['Lag_1'] = data['Close'].shift(1)
data['Lag_2'] = data['Close'].shift(2)
data['Lag_3'] = data['Close'].shift(3)
data = data.dropna()

# split the data accordingly for training and testing
X_lin = data[['Lag_1', 'Lag_2', 'Lag_3']]
y_lin = data['Close']
X_train_lin, X_test_lin = X_lin[:train_size], X_lin[train_size:]
y_train_lin, y_test_lin = y_lin[:train_size], y_lin[train_size:]

# train the linear regression model
from useful_functions import build_linear_regression
lin_model = build_linear_regression(X_train_lin, y_train_lin)

# make predictions using LSTM on the test set 
# and inverse transform the scaled predictions
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
lstm_predictions = lstm_model.predict(X_test_lstm)
lstm_predictions = scalar.inverse_transform(lstm_predictions)

# generate predictions using Linear Regression and inverse-transform them
lin_predictions = lin_model.predict(X_test_lin)
lin_predictions = scalar.inverse_transform(lin_predictions.reshape(-1, 1))

# use a weighted average to create hybrid predictions
from useful_functions import make_hybrid_predictions
hybrid_predictions = make_hybrid_predictions(lstm_predictions, lin_predictions[:lstm_predictions.shape[0]])

# make predictions for the next 10 days using our hybrid model
# predict the Next 10 Days using LSTM
lstm_future_predictions = []
last_sequence = X[-1].reshape(1, seq_length, 1)
for _ in range(10):
    lstm_pred = lstm_model.predict(last_sequence)[0, 0]
    lstm_future_predictions.append(lstm_pred)
    lstm_pred_reshaped = np.array([[lstm_pred]]).reshape(1, 1, 1)
    last_sequence = np.append(last_sequence[:, 1:, :], lstm_pred_reshaped, axis=1)
lstm_future_predictions = scalar.inverse_transform(np.array(lstm_future_predictions).reshape(-1, 1))
# predict the Next 10 Days using Linear Regression
recent_data = data['Close'].values[-3:]
lin_future_predictions = []
for _ in range(10):
    lin_pred = lin_model.predict(recent_data.reshape(1, -1))[0]
    lin_future_predictions.append(lin_pred)
    recent_data = np.append(recent_data[1:], lin_pred)
lin_future_predictions = scalar.inverse_transform(np.array(lin_future_predictions).reshape(-1, 1))

# combine the predictive power of both models to make predictions for the next 10 days
hybrid_future_predictions = make_hybrid_predictions(lstm_future_predictions, lin_future_predictions)

# create the final DataFrame to look at the predictions
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=10)
predictions_df = pd.DataFrame({
    'Date':future_dates, 
    'LSTM Predictions': lstm_future_predictions.flatten(), 
    'Linear Regression Predictions': lin_future_predictions.flatten(),
    'Hybrid Model Predictions': hybrid_future_predictions.flatten()
})
print(predictions_df)
