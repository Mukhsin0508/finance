from datetime import datetime
import requests
from django.conf import settings
from celery import shared_task
from .models import StockData

ALPHA_VANTAGE_API_KEY = settings.ALPHA_VANTAGE_API_KEY

@shared_task
def fetch_stock_data(symbol='BTC', market='USD'):
    """
    Celery task to fetch cryptocurrency data from Alpha Vantage API.
    :param symbol:
    :param market:
    :return:
    """
    url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market={market}&apikey={ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    data = response.json()

    time_series = data.get('Time Series (Digital Currency Daily)', {})

    for date, daily_data in time_series.items():
        StockData.objects.update_or_create(
            symbol=symbol,
            date=datetime.strptime(date, '%Y-%m-%d').date(),
            defaults={
                'open_price': daily_data.get('1a. open (USD)', 0),  # Provide default value 0 if the key is missing
                'close_price': daily_data.get('4a. close (USD)', 0),
                'high_price': daily_data.get('2a. high (USD)', 0),
                'low_price': daily_data.get('3a. low (USD)', 0),
                'volume': daily_data.get('5. volume', 0),
            }
        )

    return f"Cryptocurrency data for {symbol} has been fetched successfully."
