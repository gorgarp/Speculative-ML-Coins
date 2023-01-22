import yahoo_fin.stock_info as si
import pandas as pd
from sklearn.neural_network import MLPRegressor

# Get a list of all ticker symbols that match the keyword "crypto" or "coin"
crypto_tickers = si.ticker_search("crypto") + si.ticker_search("coin")

# Filter the list to only include ticker symbols that have historical data available
crypto_tickers = [ticker for ticker in crypto_tickers if si.get_data(ticker, start_date="01/01/2018") is not None]

# Create an empty DataFrame to store the data
data = pd.DataFrame()

for ticker in crypto_tickers:
    try:
        # Get historical data for the cryptocurrency
        crypto_data = si.get_data(ticker, start_date = "01/01/2018")
        
        # Rename the columns
        crypto_data.rename(columns={'close': 'y', 'date': 'ds'}, inplace=True)

        # Add the ticker symbol as a column in the data
        crypto_data["ticker"] = ticker

        # Add the new data to the existing data
        data = pd.concat([data, crypto_data])

        # Split the data into training and testing sets
        X_train = crypto_data[['ds']]
        y_train = crypto_data[['y']]
        X_test = crypto_data[['ds']]
        y_test = crypto_data[['y']]

        # Initialize and fit the neural network model
        model = MLPRegressor(hidden_layer_sizes=(100,100,100), max_iter=5000)
        model.fit(X_train, y_train)

        # Make predictions
        predictions_7_days = model.predict(X_test + 7)
        predictions_30_days = model.predict(X_test + 30)
        predictions_90_days = model.predict(X_test + 90)

        # Print the predictions
        print(f"Predicted price for {ticker} in 7 days: {predictions_7_days[-1]}")
        print(f"Predicted price for {ticker} in 30 days: {predictions_30_days[-1]}")
        print(f"Predicted price for {ticker} in 90 days: {predictions_90_days[-1]}")
        
        # Compare the predictions to the actual values
        error = predictions - y_test
        # Update the model with the new data and the error
        model.fit(X_train, y_train, error)

    except:
        #if the ticker symbol is not found or the data is not available
        print(f"{ticker} not found or data not available")

# Store the data in a csv file
data.to_csv("crypto_data.csv", index = False)
