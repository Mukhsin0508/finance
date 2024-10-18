# data_fetcher.py

import requests
from datetime import datetime
from .models import StockData
from django.conf import settings

def fetch_stock_data(symbol='AAPL', data_type='stock', market='USD'):
    """
    Fetch stock or cryptocurrency data from Alpha Vantage API and store it in the database.
    :param symbol: Stock or cryptocurrency symbol
    :param data_type: 'stock' or 'crypto'
    :param market: Market for cryptocurrency (e.g., 'USD')
    """
    ALPHA_VANTAGE_API_KEY = settings.ALPHA_VANTAGE_API_KEY

    if data_type == 'stock':
        # url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=full'
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        data = response.json()
        print(f"Fetched data from {symbol}: {data}")

        if 'Time Series (Daily)' not in data:
            raise Exception(f"Error fetching data for {symbol}: {data.get('Error Message', 'Unknown error')}")

        time_series = data['Time Series (Daily)']

        for date_str, daily_data in time_series.items():
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            StockData.objects.update_or_create(
                symbol=symbol,
                date=date,
                defaults={
                    'open_price': daily_data['1. open'],
                    'high_price': daily_data['2. high'],
                    'low_price': daily_data['3. low'],
                    'close_price': daily_data['4. close'],
                    'volume': daily_data['6. volume'],
                }
            )
        return f"Stock data for {symbol} fetched successfully."

    elif data_type == 'crypto':
        # url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market={market}&apikey={ALPHA_VANTAGE_API_KEY}'
        url = (f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&market={market}&apikey={ALPHA_VANTAGE_API_KEY}')
        response = requests.get(url)
        data = response.json()

        if 'Time Series (Digital Currency Daily)' not in data:
            raise Exception(f"Error fetching data for {symbol}: {data.get('Error Message', 'Unknown error')}")

        time_series = data['Time Series (Digital Currency Daily)']

        for date_str, daily_data in time_series.items():
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            StockData.objects.update_or_create(
                symbol=symbol,
                date=date,
                defaults={
                    'open_price': daily_data.get(f'1a. open ({market})', 0),
                    'high_price': daily_data.get(f'2a. high ({market})', 0),
                    'low_price': daily_data.get(f'3a. low ({market})', 0),
                    'close_price': daily_data.get(f'4a. close ({market})', 0),
                    'volume': daily_data.get('5. volume', 0),
                }
            )
        return f"Cryptocurrency data for {symbol} fetched successfully."

    else:
        raise ValueError("data_type must be 'stock' or 'crypto'")
