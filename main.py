import requests
import json
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

# Function to download historical price data for all coins
def download_data():
    # Make API call
    url = "https://api.cryptowat.ch/assets"
    response = requests.get(url)
    
    # Parse JSON response
    data = json.loads(response.text)
    
    # Extract list of coins
    coins = data["result"]
    
    # Loop through each coin to download historical price data
    for coin in coins:
        symbol = coin["symbol"]
        # check if the coin have trading pair in the exchange
        if 'exchange' in coin:
            exchange = coin["coinbase"]
            # get the trading pair
            trading_pair = symbol + 'usd'
            url = f"https://api.cryptowat.ch/markets/{exchange}/{trading_pair}/ohlc"
            response = requests.get(url)
            data = json.loads(response.text)
            # Extract historical price data
            df = pd.read_json(json.dumps(data['result']['86400']))
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volume_quote']
             # Save to file
            df.to_csv(f"{symbol}_prices.csv")
        else:
            print(f"Coin {symbol} doesn't have trading pair in the exchange.")
    return coins
# Function to persist data between runs
def persist_data(coins):
    data = {}
    for coin in coins:
        symbol = coin["symbol"]
        # check if the coin have trading pair in the exchange
        if 'exchange' in coin:
            if os.path.exists(f"{symbol}_prices.csv"):
                df = pd.read_csv(f"{symbol}_prices.csv")
                data[symbol] = df
    return data

# Function to download latest data each run
def update_data(data):
    for symbol in data.keys():
        df = data[symbol]
        last_timestamp = df['timestamp'].iloc[-1]
        url = f"https://api.cryptowat.ch/markets/{exchange}/{trading_pair}/ohlc?after={last_timestamp}"
        response = requests.get(url)
        new_data = json.loads(response.text)
        new_df = pd.read_json(json.dumps(new_data['result']['86400']))
        new_df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volume_quote']
        df = df.append(new_df)
        df.to_csv(f"{symbol}_prices.csv", index=False)
    return data

# Function to use machine learning to predict next day's closing price
def predict_price(df):
    # Prepare data for training
    X = df[['open', 'high', 'low', 'volume', 'volume_quote']]
    y = df['close']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)

    # Return predictions
    return y_pred
# Function to look back at predictions versus actual performance
def evaluate_performance(data, predictions):
    for symbol in data.keys():
        df = data[symbol]
        y_test = df['close'].tail(len(predictions))
        actual_performance = (y_test.iloc[-1] - y_test.iloc[0]) / y_test.iloc[0] * 100
        predicted_performance = (predictions[symbol][-1] - predictions[symbol][0]) / predictions[symbol][0] * 100
        if actual_performance > 1000:
            print(f"{symbol} actual performance: {actual_performance:.2f}%")
            print(f"{symbol} predicted performance: {predicted_performance:.2f}%")
            if actual_performance > predicted_performance:
                print(f"{symbol} performance was better than predicted.")
            else:
                print(f"{symbol} performance was worse than predicted.")

# Main function to run the script
def main():
    # Download data
    coins = download_data()
    # Persist data
    data = persist_data(coins)
    # Update data
    data = update_data(data)
    # Make predictions
    predictions = {}
    for symbol in data.keys():
        df = data[symbol]
        y_pred = predict_price(df)
        predictions[symbol] = y_pred
    # Evaluate performance
    evaluate_performance(data, predictions)

# Call main function to run script
if __name__ == "__main__":
    main()
       
