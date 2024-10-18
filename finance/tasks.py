import requests
from celery import shared_task

from config import settings
from finance.models import StockData
from datetime import datetime

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
                'open_price':daily_data['1a. open (USD)'],
                'close_price':daily_data['4a. close (USD)'],
                'high_price':daily_data['2a. high (USD)'],
                'low_price':daily_data['3a. low (USD)'],
                'volume':daily_data['5. volume'], }
        )

    return f"Cryptocurrency data for {symbol} has been fetched successfully."
