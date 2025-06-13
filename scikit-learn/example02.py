# We will predict stock price in this example using scikit-learn linear regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime

data = pd.read_csv('scikit-learn/data/apple.stocks.csv')

data = data[['Close']]
data = data.dropna()

# Create target variable by shifting the Close price
data['Target'] = data['Close'].shift(-1)
data = data[:-1]  # Remove the last row since it will have NaN target

print("Training data head:")
print(data.head())

X = data[['Close']]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print the predictions
print("\nTest predictions (first 5):")
print(predictions[:5])

# Print the mean squared error
mse = mean_squared_error(y_test, predictions)

print("Model Performance:")
print("Mean Squared Error: ", mse)

# Get the most recent closing price
latest_price = data['Close'].iloc[-1]
print(f"\nMost recent closing price: ${latest_price:.2f}")

# Predict today's price
today_prediction = model.predict([[latest_price]])[0]
print(f"Predicted price for today: ${today_prediction:.2f}")
print(f"Predicted change: ${(today_prediction - latest_price):.2f} ({(today_prediction - latest_price)/latest_price*100:.2f}%)")