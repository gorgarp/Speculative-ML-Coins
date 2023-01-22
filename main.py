import yahoo_fin.stock_info as si
import pandas as pd
from fbprophet import Prophet

# Get list of all ticker symbols for cryptocurrencies
crypto_tickers = si.ticker_search("cryptocurrency")

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
    # Initialize and fit the model
    model = Prophet()
    model.fit(crypto_data)
    # Create DataFrames to hold the predictions for different time periods
    future_7_days = model.make_future_dataframe(periods=7)
    future_30_days = model.make_future_dataframe(periods=30)
    future_90_days = model.make_future_dataframe(periods=90)

    # Make predictions
    forecast_7_days = model.predict(future_7_days)
    forecast_30_days = model.predict(future_30_days)
    forecast_90_days = model.predict(future_90_days)

    # Print the predictions
    print(f"Predicted price for {ticker} in 7 days: {forecast_7_days['yhat'][-1]}")
    print(f"Predicted price for {ticker} in 30 days: {forecast_30_days['yhat'][-1]}")
    print(f"Predicted price for {ticker} in 90 days: {forecast_90_days['yhat'][-1]}")
    
crypto_data = pd.concat([crypto_data, forecast_7_days[['ds', 'yhat']]], ignore_index=True)
crypto_data = pd.concat([crypto_data, forecast_30_days[['ds', 'yhat']]], ignore_index=True)
crypto_data = pd.concat([crypto_data, forecast_90_days[['ds', 'yhat']]], ignore_index=True)
    # Compare the predictions to the actual values
    crypto_data['error'] = crypto_data['y'] - crypto_data['yhat']
    
    # Update the model with the new data and the error
    model.fit(crypto_data)
except:
    #if the ticker symbol is not found or the data is not available
    print(f"{ticker} not found or data not available")
    
   data.to_csv("crypto_data.csv", index=False)
