import os
from celery import Celery
from celery.utils.log import get_task_logger


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

logger = get_task_logger(__name__)

app = Celery('finance')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
