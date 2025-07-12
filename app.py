import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Title
st.title("📈 Stock Market Prediction using Linear Regression")

# Generate dummy stock market data
@st.cache_data
def load_data():
    np.random.seed(42)
    days = np.arange(1, 101)
    prices = 100 + days * 0.5 + np.random.normal(0, 2, size=100)
    data = pd.DataFrame({'Day': days, 'Stock Price': prices})
    return data

df = load_data()

st.subheader("📊 Dummy Stock Data")
st.dataframe(df.head(10))

# Feature and label
X = df[['Day']]
y = df['Stock Price']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📉 Model Evaluation")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Plotting
fig, ax = plt.subplots()
ax.scatter(X_test, y_test, color='blue', label='Actual Price')
ax.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Price')
ax.set_xlabel("Day")
ax.set_ylabel("Stock Price")
ax.set_title("Stock Price Prediction")
ax.legend()
st.pyplot(fig)

# Predict a future stock price
st.subheader("🔮 Predict Future Price")
future_day = st.slider("Select future day (beyond 100)", 101, 150)
future_price = model.predict([[future_day]])
st.write(f"📌 Predicted Stock Price on Day {future_day}: ₹{future_price[0]:.2f}")
