from django.db import models

class StockData(models.Model):
    symbol = models.CharField(max_length=10)
    date = models.DateField()
    open_price = models.DecimalField(max_digits=10, decimal_places=2)
    close_price = models.DecimalField(max_digits=10, decimal_places=2)
    high_price = models.DecimalField(max_digits=10, decimal_places=2)
    low_price = models.DecimalField(max_digits=10, decimal_places=2)
    volume = models.BigIntegerField()

    def __str__(self):
        return f"{self.symbol} - {self.date}"


class PredictedStockPrice(models.Model):
    symbol = models.CharField(max_length=10)
    date = models.DateField()
    predicted_close_price = models.DecimalField(max_digits=10, decimal_places=2)
    predicted_open_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    predicted_high_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    predicted_low_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    predicted_volume = models.BigIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)