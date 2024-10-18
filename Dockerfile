FROM python:3.11-slim-buster

ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN mkdir /app
WORKDIR /app
ENV PATH="/home/app/.local/bin:${PATH}"
COPY ./requirements.txt .
RUN pip install -r requirements.txt
RUN pip install watchdog
COPY . .
RUN chmod +x /app/docker-entrypoint.sh
