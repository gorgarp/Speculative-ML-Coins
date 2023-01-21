import requests
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

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
            exchange = coin["exchange"]
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


# Function to persist data between runs
def persist_data():
    # Load data from files and store in a variable
    data = {}
    for coin in coins:
        symbol = coin["symbol"]
        df = pd.read_csv(f"{symbol}_prices.csv")
        data[symbol] = df
    return data

# Function to download latest data each run
def update_data(data):
    # Loop through each coin to download latest data
    for coin in coins:
        symbol = coin["symbol"]
        url = f"https://api.cryptowat.ch/markets/{symbol}/ohlc"
        response = requests.get(url)
        new_data = json.loads(response.text)
        
        # Extract latest price data
        prices = new_data["result"]["86400"]
        
        # Convert to pandas DataFrame
        new_df = pd.DataFrame(prices, columns=["timestamp", "open", "high", "low", "close", "volume", "volume_quote"])
        
        # Append new data to existing data
        data[symbol] = data[symbol].append(new_df)
    return data

# Function to use machine learning to predict next day's closing price
def predict_price(df):
    # Prepare data for training
    X = df[['open', 'high', 'low', 'volume', 'volume_quote']]
    y = df['close']

# Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train linear regression model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Make predictions on test set
    y_pred = regressor.predict(X_test)

    # Evaluate model performance
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")

    # Make predictions on latest data
    latest_data = X.tail(1)
    prediction = regressor.predict(latest_data)
    return prediction


# Function to look back at predictions versus actual performance
def evaluate_performance(data, predictions):
    # Compare predictions to actual performance
    for symbol in data.keys():
        df = data[symbol]
        prediction = predictions[symbol]
        actual_performance = df["close"].tail(1).values[0]
        error = abs(prediction - actual_performance)
        print(f"Coin: {symbol}, Prediction: {prediction}, Actual: {actual_performance}, Error: {error}")

# Main function to run the script
def main():
    download_data()
    data = persist_data()
    data = update_data(data)
    predictions = {}
    for symbol in data.keys():
        df = data[symbol]
        predictions[symbol] = predict_price(df)
    evaluate_performance(data, predictions)

if __name__ == "__main__":
    main()
