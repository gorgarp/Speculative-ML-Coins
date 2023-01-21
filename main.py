import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# list of exchange we want to include
exchanges = ["Coinbase","Kucoin","Tradeogre"]

# Load historical data from file
try:
    df = pd.read_csv("historical_data.csv")
except FileNotFoundError:
    # If file doesn't exist, download data from CoinMarketCap API
    endpoint = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
    params = {
        "interval": "1d",
        "start": "2022-01-01",
        "end": "2022-01-31",
        "convert": "USD"
    }
    headers = {
        "X-CMC_PRO_API_KEY": "YOUR_API_KEY"
    }
    response = requests.get(endpoint, headers=headers, params=params)
    data = json.loads(response.text)
    df = pd.DataFrame(data["data"])
    df = df[["name", "quote.USD.close", "quote.USD.volume_24h","market_data.exchanges"]]
    df = df[df["market_data.exchanges"].apply(lambda x: any(i['name'] in exchanges for i in x))]
    df.columns = ["name", "close", "volume","exchanges"]
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

    # Create the X and y arrays for the model
    X = coin_df[["volume"]].values
    y = coin_df["close"].values

    # Create the model
    model = RandomForestRegressor().fit(X, y)

    # Make predictions for next 7 days
    X_test = coin_df.tail(7)["volume"].values
    predictions = model.predict(X_test.reshape(-1, 1))
    predicted_return = (predictions[-1] / predictions[0]) - 1

    # Check if coin is already in predictions DataFrame
    coin_predictions_df = predictions_df[predictions_df["coin"] == coin]
    if coin_predictions_df.empty:
        # Coin is not in predictions DataFrame, so add it
        actual_return = None
        predictions_df = predictions_df.append({"coin": coin, "predicted_return": predicted_
