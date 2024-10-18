import os

from django.conf import settings
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
import requests
import pandas as pd
from decimal import Decimal
import pickle
from datetime import timedelta

from .tasks import fetch_stock_data
from config.settings import ALPHA_VANTAGE_API_KEY
from .models import StockData, PredictedStockPrice
from .serializers import StockPriceSerializer, PredictedStockPriceSerializer, DateRangeValidator


def load_model(filename='linear_model.pkl'):
    model_path = os.path.join(settings.BASE_DIR, filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Ensure the file pointer is at the start and load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

        print(model)

    return model


class FetchStockDataView(APIView):
    permission_classes = [AllowAny]
    serializer_class = StockPriceSerializer
    fetch_stock_data.delay(symbol='BTC', market='USD')



class BacktestView(APIView):
    def post(self, request):
        symbol = request.data.get('symbol')
        initial_investment = Decimal(request.data.get('initial_investment', 10000))
        short_window = int(request.data.get('short_window', 50))
        long_window = int(request.data.get('long_window', 200))

        # Fetch stock data from the database
        stock_data = StockData.objects.filter(symbol=symbol).order_by('date')
        if not stock_data.exists():
            return Response({"error": "No data available"}, status=404)

        # Create a DataFrame for processing
        data = pd.DataFrame(list(stock_data.values('date', 'close_price')))
        data['50_day_ma'] = data['close_price'].rolling(window=short_window).mean()
        data['200_day_ma'] = data['close_price'].rolling(window=long_window).mean()

        # Simulate backtesting
        cash = initial_investment
        position = 0
        trades_executed = 0  # To track if any trades are made
        for i in range(len(data)):

            if data['50_day_ma'][i] < data['200_day_ma'][i] and position == 0:
                position = cash / Decimal(data['close_price'][i])  # Buy
                cash = Decimal(0)
                print(f"Buy at {data['date'][i]} for {data['close_price'][i]}")

            elif data['50_day_ma'][i] > data['200_day_ma'][i] and position > 0:
                cash = position * Decimal(data['close_price'][i])  # Sell
                position = 0
                print(f"Sell at {data['date'][i]} for {data['close_price'][i]}")

        total_return = cash - initial_investment

        return Response({"total_return": float(total_return)}, status=200)



class PredictStockPricesView(APIView):
    def get(self, request, symbol):

        # Получаем исторические данные для символа
        stock_data = StockData.objects.filter(symbol=symbol).order_by('date')
        if not stock_data.exists():
            return Response({"error": "No data available for prediction"}, status=404)

        # Подготовка данных для модели
        data = pd.DataFrame(list(stock_data.values('date', 'close_price')))
        data['days'] = (pd.to_datetime(data['date']) - pd.to_datetime(data['date'].min())).dt.days

        # Загружаем предобученную модель
        model = load_model()

        # Предсказание цен на следующие 30 дней
        last_day = data['days'].max()
        future_days = pd.DataFrame({'days': range(last_day + 1, last_day + 31)})
        predictions = model.predict(future_days[['days']])

        # Создаем предсказанные цены как объекты PredictedStockPrice
        predicted_prices = []
        for i, pred in enumerate(predictions):
            predicted_price = PredictedStockPrice(
                symbol=symbol,
                date=data['date'].max() + timedelta(days=i + 1),
                predicted_close_price=Decimal(pred)
            )
            predicted_prices.append(predicted_price)

        # Сохраняем предсказанные данные в базу данных
        PredictedStockPrice.objects.bulk_create(predicted_prices)

        # Сериализуем предсказанные цены и возвращаем как ответ
        serializer = PredictedStockPriceSerializer(predicted_prices, many=True)
        return Response(serializer.data, status=200)


class CompareStockPricesView(APIView):
    def get(self, request, symbol):

        # Получаем реальные данные для указанного символа
        real_data = StockData.objects.filter(symbol=symbol).order_by('date')
        if not real_data.exists():
            return Response({"error": f"No real data available for {symbol}"}, status=404)

        # Получаем предсказанные данные, но временно делаем предсказания на существующие даты
        real_dates = real_data.values_list('date', flat=True)
        predicted_data = PredictedStockPrice.objects.filter(symbol=symbol, date__in=real_dates).order_by('date')

        # Если предсказанных данных для этих дат нет, просто сгенерируем временные предсказания
        if not predicted_data.exists():
            predicted_data = [PredictedStockPrice(symbol=symbol, date=real.date,
                                                  predicted_close_price=real.close_price * Decimal(1.05)) for real in
                              real_data]

        # Формируем список для сравнения
        comparison_results = []
        for real in real_data:
            predicted = next((p for p in predicted_data if p.date == real.date), None)
            if predicted:
                comparison_results.append({
                    "date": real.date,
                    "predicted_close_price": float(predicted.predicted_close_price),
                    "real_close_price": float(real.close_price),
                    "difference": float(predicted.predicted_close_price - real.close_price)
                })
            else:
                comparison_results.append({
                    "date": real.date,
                    "predicted_close_price": None,
                    "real_close_price": float(real.close_price),
                    "difference": None
                })

        # Возвращаем ответ с результатами сравнения
        return Response({
            "symbol": symbol,
            "comparison": comparison_results
        }, status=200)