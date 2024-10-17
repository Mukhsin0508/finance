import io
import os
from rest_framework.response import Response
from rest_framework.views import APIView
import requests
import pandas as pd
from decimal import Decimal
import pickle
from datetime import timedelta
import matplotlib.pyplot as plt

from io import BytesIO
import base64
from django.http import HttpResponse


from config import settings
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
    def get(self, request, symbol):
        # Валидируем диапазон дат
        validator = DateRangeValidator(data=request.query_params)
        if not validator.is_valid():
            return Response(validator.errors, status=400)

        # Извлекаем start_date и end_date из валидированных данных
        validated_data = validator.validated_data
        start_date = validated_data['start_date']
        end_date = validated_data['end_date']

        # Преобразуем даты в формат строки для API Alpha Vantage
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        api_key = settings.ALPHA_VANTAGE_API_KEY
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json().get('Time Series (Daily)', {})
            stock_objects = []
            for date, prices in data.items():
                date_obj = pd.to_datetime(date).date()

                # Фильтруем данные по диапазону дат
                if start_date <= date_obj <= end_date:
                    stock_price, created = StockPrice.objects.update_or_create(
                        symbol=symbol,
                        date=date_obj,
                        defaults={
                            'open_price': prices['1. open'],
                            'close_price': prices['4. close'],
                            'high_price': prices['2. high'],
                            'low_price': prices['3. low'],
                            'volume': prices['5. volume']
                        }
                    )
                    stock_objects.append(stock_price)

            serializer = StockPriceSerializer(stock_objects, many=True)
            return Response(serializer.data, status=200)
        else:
            return Response({"error": "Failed to fetch data from API"}, status=400)


class BacktestView(APIView):
    def post(self, request):
        symbol = request.data.get('symbol')
        initial_investment = Decimal(request.data.get('initial_investment', 10000))
        short_window = int(request.data.get('short_window', 50))
        long_window = int(request.data.get('long_window', 200))

        # Fetch stock data from the database
        stock_data = StockPrice.objects.filter(symbol=symbol).order_by('date')
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
        stock_data = StockPrice.objects.filter(symbol=symbol).order_by('date')
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
        real_data = StockPrice.objects.filter(symbol=symbol).order_by('date')
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


class ReportGenerationView(APIView):
    def get(self, request, symbol):
        print(f"Request received for symbol: {symbol}")

        # Получение реальных данных
        real_data = StockPrice.objects.filter(symbol=symbol).order_by('date')
        if not real_data.exists():
            return Response({"error": "No real data available for symbol: {}".format(symbol)}, status=404)

        # Получение предсказанных данных
        predicted_data = PredictedStockPrice.objects.filter(symbol=symbol).order_by('date')
        if not predicted_data.exists():
            return Response({"error": "No predicted data available for symbol: {}".format(symbol)}, status=404)

        # Преобразование данных в pandas DataFrame
        real_df = pd.DataFrame(list(real_data.values('date', 'close_price')))
        predicted_df = pd.DataFrame(list(predicted_data.values('date', 'predicted_close_price')))

        # Генерация графика с использованием Matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(real_df['date'], real_df['close_price'], label='Actual Prices', color='blue')
        ax.plot(predicted_df['date'], predicted_df['predicted_close_price'], label='Predicted Prices', color='green')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'Stock Price Prediction vs Actual for {symbol}')
        ax.legend()

        # Сохранение графика в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        # Генерация PDF отчета с графиком
        if request.query_params.get('format') == 'pdf':
            # Создание PDF отчета
            pdf_buffer = io.BytesIO()
            pdf_canvas = canvas.Canvas(pdf_buffer, pagesize=letter)

            # Заголовок PDF
            pdf_canvas.drawString(100, 800, f"Stock Prediction Report for {symbol}")

            # Отображение ключевых метрик
            key_metrics = {
                "symbol": symbol,
                "start_date": real_df['date'].min(),
                "end_date": real_df['date'].max(),
                "mean_real_price": real_df['close_price'].mean(),
                "mean_predicted_price": predicted_df['predicted_close_price'].mean(),
                "price_difference": abs(real_df['close_price'].mean() - predicted_df['predicted_close_price'].mean())
            }
            pdf_canvas.drawString(100, 750, f"Symbol: {key_metrics['symbol']}")
            pdf_canvas.drawString(100, 730, f"Start Date: {key_metrics['start_date']}")
            pdf_canvas.drawString(100, 710, f"End Date: {key_metrics['end_date']}")
            pdf_canvas.drawString(100, 690, f"Mean Real Price: {key_metrics['mean_real_price']:.2f}")
            pdf_canvas.drawString(100, 670, f"Mean Predicted Price: {key_metrics['mean_predicted_price']:.2f}")
            pdf_canvas.drawString(100, 650, f"Price Difference: {key_metrics['price_difference']:.2f}")

            # Добавление графика из base64
            img_data = base64.b64decode(request.data.get('graph_base64'))
            img_buffer = BytesIO(img_data)
            pdf_canvas.drawImage(img_buffer, 50, 400, width=500, height=300)

            # Закрытие PDF файла
            pdf_canvas.showPage()
            pdf_canvas.save()

            # Вернуть PDF отчет
            pdf_buffer.seek(0)
            response = HttpResponse(pdf_buffer, content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="{symbol}_report.pdf"'
            return response

        # Альтернативно можно вернуть JSON с ключевыми метриками и графиком
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return Response({
            # "key_metrics": key_metrics,
            "graph_base64": img_base64
        })