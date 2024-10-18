# views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from .models import StockData, PredictedStockPrice
from .serializers import StockDataSerializer, PredictedStockPriceSerializer
from .data_fetcher import fetch_stock_data
from django.conf import settings
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
import os
import pickle
import matplotlib.pyplot as plt
import io
from django.http import FileResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from sklearn.linear_model import LinearRegression



class FetchStockDataView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, symbol):
        data_type = request.query_params.get('data_type', 'stock')
        market = request.query_params.get('market', 'USD')  # For cryptocurrencies

        try:
            message = fetch_stock_data(symbol=symbol, data_type=data_type, market=market)
            return Response({"message": message}, status=200)
        except Exception as e:
            return Response({"error": str(e)}, status=400)


class BacktestView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        symbol = request.data.get('symbol')
        initial_investment = Decimal(request.data.get('initial_investment', '10000'))
        short_window = int(request.data.get('short_window', 50))
        long_window = int(request.data.get('long_window', 200))

        # Ensure data is available
        stock_data = StockData.objects.filter(symbol=symbol).order_by('date')
        if not stock_data.exists():
            fetch_stock_data(symbol=symbol)
            stock_data = StockData.objects.filter(symbol=symbol).order_by('date')
            if not stock_data.exists():
                return Response({"error": "No data available"}, status=404)

        # Prepare data
        data = pd.DataFrame(list(stock_data.values('date', 'close_price')))
        data['close_price'] = data['close_price'].astype(float)
        data['50_day_ma'] = data['close_price'].rolling(window=short_window).mean()
        data['200_day_ma'] = data['close_price'].rolling(window=long_window).mean()

        # Initialize variables for backtesting
        cash = initial_investment
        position = 0
        trades_executed = 0
        peak_value = initial_investment
        max_drawdown = Decimal(0)

        # Simulate the buy/sell strategy
        for i in range(len(data)):
            current_price = Decimal(data['close_price'].iloc[i])

            if data['50_day_ma'].iloc[i] < data['200_day_ma'].iloc[i] and position == 0:
                # Buy
                position = cash / current_price
                cash = Decimal(0)
                trades_executed += 1
            elif data['50_day_ma'].iloc[i] > data['200_day_ma'].iloc[i] and position > 0:
                # Sell
                cash = position * current_price
                position = 0
                trades_executed += 1

            # Calculate total portfolio value
            total_value = cash + position * current_price
            if total_value > peak_value:
                peak_value = total_value
            drawdown = (peak_value - total_value) / peak_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Final value
        final_value = cash + position * Decimal(data['close_price'].iloc[-1])
        total_return = final_value - initial_investment

        performance_summary = {
            "total_return": float(total_return),
            "final_value": float(final_value),
            "max_drawdown": float(max_drawdown),
            "trades_executed": trades_executed
        }

        return Response(performance_summary, status=200)

class PredictStockPricesView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, symbol):
        # Ensure data is available
        stock_data = StockData.objects.filter(symbol=symbol).order_by('date')
        if not stock_data.exists():
            fetch_stock_data(symbol=symbol)
            stock_data = StockData.objects.filter(symbol=symbol).order_by('date')
            if not stock_data.exists():
                return Response({"error": "No data available"}, status=404)

        # Prepare data
        data = pd.DataFrame(list(stock_data.values('date', 'close_price')))
        data['date'] = pd.to_datetime(data['date'])
        data['days'] = (data['date'] - data['date'].min()).dt.days
        data['close_price'] = data['close_price'].astype(float)

        # Prepare features and target
        X = data[['days']]
        y = data['close_price']

        # Train the model or load existing model
        model_filename = f'linear_model_{symbol}.pkl'
        model_path = os.path.join(settings.BASE_DIR, model_filename)

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            model = LinearRegression()
            model.fit(X, y)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        # Predict next 30 days
        last_day = data['days'].max()
        future_days = pd.DataFrame({'days': range(last_day + 1, last_day + 31)})
        predictions = model.predict(future_days)

        # Save predictions to database
        last_date = data['date'].max()
        predicted_prices = []
        for i, pred in enumerate(predictions):
            pred_date = last_date + timedelta(days=i+1)
            predicted_price = PredictedStockPrice(
                symbol=symbol,
                date=pred_date.date(),
                predicted_close_price=Decimal(pred)
            )
            predicted_prices.append(predicted_price)

        # Save to database if not already exists
        existing_dates = PredictedStockPrice.objects.filter(
            symbol=symbol,
            date__in=[p.date for p in predicted_prices]
        ).values_list('date', flat=True)

        new_predictions = [p for p in predicted_prices if p.date not in existing_dates]
        if new_predictions:
            PredictedStockPrice.objects.bulk_create(new_predictions)

        # Serialize and return predictions
        serializer = PredictedStockPriceSerializer(predicted_prices, many=True)
        return Response(serializer.data, status=200)

class ReportView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, symbol):
        response_format = request.query_params.get('format', 'pdf')
        # Ensure data is available
        stock_data = StockData.objects.filter(symbol=symbol).order_by('date')
        predicted_data = PredictedStockPrice.objects.filter(symbol=symbol).order_by('date')

        if not stock_data.exists():
            return Response({"error": "No stock data available"}, status=404)
        if not predicted_data.exists():
            return Response({"error": "No predicted data available"}, status=404)

        # Prepare data
        actual_data = pd.DataFrame(list(stock_data.values('date', 'close_price')))
        actual_data['date'] = pd.to_datetime(actual_data['date'])
        actual_data['close_price'] = actual_data['close_price'].astype(float)

        predicted_data = pd.DataFrame(list(predicted_data.values('date', 'predicted_close_price')))
        predicted_data['date'] = pd.to_datetime(predicted_data['date'])
        predicted_data['predicted_close_price'] = predicted_data['predicted_close_price'].astype(float)

        if response_format == 'json':
            # Return data as JSON
            combined_data = pd.merge(actual_data, predicted_data, on='date', how='outer')
            combined_data.sort_values('date', inplace=True)
            combined_data.fillna('', inplace=True)
            data = combined_data.to_dict(orient='records')
            return Response(data, status=200)
        else:
            # Generate PDF report
            # Plot the data
            plt.figure(figsize=(10, 6))
            plt.plot(actual_data['date'], actual_data['close_price'], label='Actual Price')
            plt.plot(predicted_data['date'], predicted_data['predicted_close_price'], label='Predicted Price')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'Stock Price Prediction for {symbol}')
            plt.legend()

            # Save plot to a BytesIO buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

            # Generate PDF report
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            width, height = letter

            # Add plot image to PDF
            c.drawImage(buf, 50, height / 2, width=500, height=300)

            # Add text
            c.setFont("Helvetica", 14)
            c.drawString(50, height - 50, f"Stock Price Prediction Report for {symbol}")

            # Add more details as needed

            c.showPage()
            c.save()
            pdf_buffer.seek(0)

            return FileResponse(pdf_buffer, as_attachment=True, filename=f'report_{symbol}.pdf')
