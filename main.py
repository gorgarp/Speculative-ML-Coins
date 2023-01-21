import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

exchanges = ["Coinbase","Kucoin","Tradeogre"]
headers = {
    "X-CMC_PRO_API_KEY": "YOUR_API_KEY"
}
params = {
    "convert":"USD"
}

# download all historical data from CoinMarketCap API
response = requests.get("https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest", headers=headers, params=params)
data = response.json()["data"]

# Extract the data
df = pd.DataFrame(data)
df = df[["name", "symbol", "quotes.USD.price", "quotes.USD.volume_24h"]]
df.columns = ["name", "symbol", "price", "volume"]

# filter the data to include only the coins listed on the specified exchanges
response = requests.get("https://pro-api.coinmarketcap.com/v1/exchange/listings/latest", headers=headers)
data = response.json()["data"]
exchanges_df = pd.DataFrame(data)
exchanges_df = exchanges_df[["name","slug"]]
exchanges_df = exchanges_df[exchanges_df["name"].isin(exchanges)]
exchanges_df = exchanges_df.rename(columns={'slug':'symbol'})
df = pd.merge(df, exchanges_df, on='symbol')

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
    predictions_df["predicted_return"] = predictions_df["predicted_return"].apply(lambda x: x / y_test.mean() - 1)
    predictions_df["actual_return"] = predictions_df["actual_return"].apply(lambda x: x / y_test.mean() - 1)
    predictions_df = predictions_df[predictions_df["predicted_return"] > 0.1]

    # Save predictions to file
    predictions_df.to_csv("predictions.csv", index=False)
