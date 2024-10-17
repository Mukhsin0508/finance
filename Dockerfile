FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --upgrade pip --root-user-action=ignore && \
    pip install setuptools && \
    pip install -r requirements.txt && \
    rm -rf /root/.cache/pip /root/.cache && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

# Default command to run the Django app (or any custom command)
# CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
