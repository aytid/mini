import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import datetime
import yfinance as yf
from datetime import date,timedelta
from sklearn.svm import SVR

today = date.today()
d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=1825)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

st.title('Stock Trend Prediction')

user_input=st.text_input("Enter Stock Ticker","TSLA")
df=yf.download(user_input,start=start_date,end=end_date,progress=False)

st.subheader('Data from 2020-2023')
st.write(df.describe())

st.subheader('Closing Price Vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart with 100MA and 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

  
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)

#Load my model
model=load_model('keras_model.h5')

past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
	x_test.append(input_data[i-100:i])
	y_test.append(input_data[i,0])


x_test, y_test=np.array(x_test), np.array(y_test)
y_predicted=model.predict(x_test)
scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test=y_test*scale_factor

st.subheader('Prediction Vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
st.subheader('Prediction Graph')
df['Year'] = pd.DatetimeIndex(df.index).year
# Create an array of the independent variables (features)
X = np.array(df['Year']).reshape(-1, 1)

# Create an array of the dependent variable (target)
y = np.array(df['Close'])

# Create a Support Vector Regression model with a radial basis function kernel
model = SVR(kernel='rbf', C=1e3, gamma=0.1)

# Train the model on the data
model.fit(X, y)

# Make predictions for the previous 3 years and the next 3 years
previous_years = np.array(range(df['Year'].min() - 3, df['Year'].min())).reshape(-1, 1)
future_years = np.array(range(df['Year'].max() + 1, df['Year'].max() + 4)).reshape(-1, 1)
all_years = np.concatenate((previous_years, X, future_years), axis=0)
predictions = model.predict(all_years)


fig=plt.figure(figsize=(12,6))
plt.plot(all_years, predictions, label='Predictions')
plt.xlabel("Year")
plt.ylabel("Stock Price")
plt.legend()
st.pyplot(fig)