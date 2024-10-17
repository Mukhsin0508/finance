#!/usr/bin/env bash
set -e

python manage.py migrate --noinput
python manage.py collectstatic --noinput

if [ "$DJANGO_DEBUG" = "True" ]; then
  # Development configuration with hot reloading
  exec watchmedo auto-restart --directory=./ --pattern="*.py" --recursive -- python manage.py runserver 0.0.0.0:8000
else
  # Production configuration
  exec gunicorn --reload -b 0.0.0.0:8000 config.wsgi --workers 5 --timeout 300 --graceful-timeout 10 --log-level warning
fi