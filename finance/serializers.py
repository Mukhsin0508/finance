# serializers.py

from rest_framework import serializers
from .models import StockData, PredictedStockPrice

class StockDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = StockData
        fields = '__all__'

class PredictedStockPriceSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictedStockPrice
        fields = '__all__'
