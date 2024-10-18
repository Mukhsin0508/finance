from datetime import timedelta

from django.utils import timezone
from rest_framework import serializers
from .models import StockData, PredictedStockPrice

class StockPriceSerializer(serializers.ModelSerializer):
    class Meta:
        model = StockData
        fields = '__all__'


class DateRangeValidator(serializers.Serializer):
    start_date = serializers.DateField(required=False)
    end_date = serializers.DateField(required=False)

    def validate(self, data):
        # Получаем текущую дату
        today = timezone.now().date()

        # Дефолтные значения — последние 2 года
        default_start = today - timedelta(days=730)  # 2 года назад
        default_end = today

        # Если пользователь не указал start_date и end_date, используем дефолтные
        start_date = data.get('start_date', default_start)
        end_date = data.get('end_date', default_end)

        # Проверка на корректность диапазона
        if start_date > end_date:
            raise serializers.ValidationError("Start date must be earlier than end date.")
        if end_date > today:
            raise serializers.ValidationError("End date cannot be in the future.")

        data['start_date'] = start_date
        data['end_date'] = end_date

        return data


class PredictedStockPriceSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictedStockPrice
        fields = '__all__'