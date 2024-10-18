# urls.py

from django.urls import path
from .views import FetchStockDataView, BacktestView, PredictStockPricesView, ReportView

urlpatterns = [
    path('fetch/<str:symbol>/', FetchStockDataView.as_view(), name='fetch_stock_data'),
    path('backtest/', BacktestView.as_view(), name='backtest'),
    path('predict/<str:symbol>/', PredictStockPricesView.as_view(), name='predict_stock_prices'),
    path('report/<str:symbol>/', ReportView.as_view(), name='report'),
]
