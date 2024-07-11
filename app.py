import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model
model = load_model('Stock Prediction Model.h5')

def load_data_and_predict(df1, model, days_to_predict=50):
    # Scaling data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1_scaled = scaler.fit_transform(np.array(df1).reshape(-1, 1))

    # Split data into training and testing sets
    training_size = int(len(df1_scaled) * 0.65)
    test_size = len(df1_scaled) - training_size
    train_data, test_data = df1_scaled[0:training_size, :], df1_scaled[training_size:len(df1_scaled), :1]

    # Prepare input data
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform predictions
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Generate future predictions
    x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
    x_input = x_input[0].tolist()
    temp_input = list(x_input)


    future_predictions = []
    i=0

    for i in range(days_to_predict):
        if len(temp_input) > time_step:
            x_input = np.array(temp_input[-time_step:]).reshape((1, time_step, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            future_predictions.extend(yhat.tolist())
            i += 1
        else:
            x_input = np.array(temp_input).reshape((1, time_step, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            future_predictions.extend(yhat.tolist())
            i += 1


        future_predictions.append(yhat[0])

    future_predictions = scaler.inverse_transform(future_predictions)

    return train_predict, test_predict, future_predictions
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

st.title('Stock Price Prediction')

st.sidebar.header('Upload or Input Data')
uploaded_file = st.sidebar.file_uploader("Upload CSV file or input data (comma-separated)", type=['csv'])
ticker = st.sidebar.text_input("Enter Ticker Symbol ", "AAPL")
import requests

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    start_date = '2021-01-01'
    end_date = '2023-12-31'
    api_key='b116a3fb23bd7683f16c442f0f219df69d51dd99'
    url = f'https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&endDate={end_date}'
    headers={
    'Content-type':'application/json',
    'Authorization':f'Token {api_key}'
    }
    # Fetch data from Tiingo API or any other source
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
    else:
        st.error(f"Failed to fetch data: {response.status_code}, {response.text}")

st.subheader('Raw Data')
st.write(df)
import matplotlib.pyplot as plt

if st.button('Predict'):
    df1 = df.reset_index()['close'].values  
    train_predict, test_predict, future_predict = load_data_and_predict(df1, model, days_to_predict=50)
    st.subheader('Predicted Stock Prices')
    plt.figure(figsize=(12, 6))
    plt.plot(df1, label='Original Data')
    plt.plot(np.arange(len(df1) - len(test_predict), len(df1)), test_predict, label='Test Predict')
    plt.plot(np.arange(len(df1), len(df1) + len(future_predict)), future_predict, label='Future Predict')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    st.pyplot(plt)







