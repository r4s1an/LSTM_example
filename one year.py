import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, InputLayer, LSTMV1
from tensorflow.python.keras.callbacks import ModelCheckpoint
from dataset_modifications import get_last_month_data, df_to_X_y, get_dataset, mldatasets,df_to_X_y_horizon, get_last_two_months_data,last_one_year_data
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random

# Setting random seed for reproducibility
tf.random.set_seed(31)

# Loading and preprocessing the dataset
dataset_path = "C:/Users/Yoked/Desktop/Internship/Forecasting/linkedin/dataset/de_new.csv"
dataset = get_dataset(x=dataset_path)
tm_df = last_one_year_data(dataset)

plt.plot(tm_df)
plt.show()

# Defining the window size
window_size = 36
forecast_horizon=12

# Converting the DataFrame to feature and target arrays
X, y = df_to_X_y_horizon(tm_df, window_size, forecast_horizon)

# Scaling the data using StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
y = scaler_y.fit_transform(y.reshape(-1, 1)).reshape(-1, forecast_horizon)

# Preparing the training, validation, and test datasets
X_train, y_train, X_val, y_val, X_test, y_test = mldatasets(X, y)

# Defining the LSTM model with the correct input shape
model = Sequential()
model.add(InputLayer(input_shape=(window_size, 1)))
model.add(LSTMV1(128,activation= 'relu', return_sequences= True))
model.add(LSTMV1(64,activation= 'relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(forecast_horizon, activation='linear'))

# Displaying the model summary
print(model.summary())

# Setting up model checkpointing
cp = ModelCheckpoint('model1/', save_best_only=True)

# Compiling the model
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# Training the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=[cp])

# Making predictions on the test set
y_pred = model.predict(X_test)

# Reshape y_test and y_pred to 2D for inverse transformation
y_test_flat = y_test.reshape(-1, forecast_horizon)
y_pred_flat = y_pred.reshape(-1, forecast_horizon)

# Inverse transform the predictions and actual values
y_pred_inv = scaler_y.inverse_transform(y_pred_flat)
y_test_inv = scaler_y.inverse_transform(y_test_flat)

# Reshape back to original 3D shape
y_pred_inv = y_pred_inv.reshape(y_pred.shape)
y_test_inv = y_test_inv.reshape(y_test.shape)

# Error Metrics
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_values = []

for i in range(y_test_inv.shape[0]):
    mape_values.append(mape(y_test_inv[i], y_pred_inv[i]))

mean_mape = np.mean(mape_values)
mae = mean_absolute_error(y_test_inv.flatten(), y_pred_inv.flatten())
mse = mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten())
rmse = np.sqrt(mse)
print(f'MAE : {mae}\nMSE : {mse}\nRMSE : {rmse}\nMAPE : {mean_mape}')

# Plot predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv.flatten(), label='Actual')
plt.plot(y_pred_inv.flatten(), label='Predicted')
plt.legend()
plt.show()

# Plot predictions vs actual values for some test samples
plt.figure(figsize=(12, 6))
for i in range(5):
    random_value = random.randint(1, len(y_test_inv))  # Picks a random integer between 1 and 100 (inclusive)
    plt.plot(y_test_inv[random_value], color='blue', alpha=0.1, label='Actual')
    plt.plot(y_pred_inv[random_value], color='red', alpha=0.1, label='Predicted')
    plt.legend()
    plt.show()