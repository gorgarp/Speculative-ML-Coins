import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import requests

# list of exchange we want to include
exchanges = ["Coinbase","Kucoin","Tradeogre"]

# download all historical data from CoinMarketCap API
endpoint = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
params = {
    "start": "1",
    "convert": "USD"
}
headers = {
    "X-CMC_PRO_API_KEY": "YOUR_API_KEY"
}
response = requests.get(endpoint, headers=headers, params=params)
data = response.json()

# Extract the data
df = pd.DataFrame(data["data"])
df = df[["name", "quote.USD.price", "quote.USD.volume_24h","exchanges"]]
df = df[df["exchanges"].apply(lambda x: any(i['name'] in exchanges for i in x))]
df.columns = ["name", "price", "volume","exchanges"]

# save the data to file
df.to_csv("historical_data.csv", index=False)

# Load previous predictions from file
try:
    predictions_df = pd.read_csv("predictions.csv")
except FileNotFoundError:
    predictions_df = pd.DataFrame(columns=["coin", "predicted_return", "actual_return"])

# Loop through each coin in the DataFrame
for coin in df["name"].unique():
    # Create a new DataFrame for the current coin
    coin_df = df[df["name"] == coin]

    # Create the X and y
    X = coin_df[["volume"]]
    y = coin_df["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest regression model on the data
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    coin_predictions_df = pd.DataFrame({"coin": coin,
                                       "predicted_return": y_pred,
                                       "actual_return": y_test})
    predictions_df = predictions_df.append(coin_predictions_df, ignore_index=True)

    # Check if any predictions have a return greater than 1000%
    predictions_df["predicted_return"] = predictions_df["predicted_return"] * 100
    predictions_df["actual_return"] = predictions_df["actual_return"] * 100
    predictions_df = predictions_df[predictions_df["predicted_return"] > 1000]

    # Save predictions to file
    predictions_df.to_csv("predictions.csv", index=False)

# Print the number of predictions
print(f"{len(predictions_df)} predictions made.")
