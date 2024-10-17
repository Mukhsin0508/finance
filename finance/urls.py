from django.urls import path
from .views import FetchStockDataView, BacktestView, PredictStockPricesView, CompareStockPricesView, \
    ReportGenerationView

urlpatterns = [
    path('fetch/<str:symbol>/', FetchStockDataView.as_view(), name='fetch_stock_data'),
    path('backtest/', BacktestView.as_view(), name='backtest'),
    path('predict/<str:symbol>/', PredictStockPricesView.as_view(), name='predict_stock_prices'),
    path('compare/<str:symbol>/', CompareStockPricesView.as_view(), name='compare_stock_prices'),
    path('report/<str:symbol>/', ReportGenerationView.as_view(), name='report-generation'),
]
