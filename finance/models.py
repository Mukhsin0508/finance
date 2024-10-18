from django.db import models

class StockData(models.Model):
    symbol = models.CharField(max_length=10)
    date = models.DateField()
    open_price = models.DecimalField(max_digits=20, decimal_places=8)
    close_price = models.DecimalField(max_digits=20, decimal_places=8)
    high_price = models.DecimalField(max_digits=20, decimal_places=8)
    low_price = models.DecimalField(max_digits=20, decimal_places=8)
    volume = models.DecimalField(max_digits=20, decimal_places=8)

    def __str__(self):
        return f"{self.symbol} - {self.date}"

class PredictedStockPrice(models.Model):
    symbol = models.CharField(max_length=10)
    date = models.DateField()
    predicted_close_price = models.DecimalField(max_digits=20, decimal_places=8)
    predicted_open_price = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    predicted_high_price = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    predicted_low_price = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    predicted_volume = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.symbol} - Predicted on {self.date}"