#https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker
FROM python:3.9

COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY . .

RUN uvicorn app:app --host 0.0.0.0 --port 5000
