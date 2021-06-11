#https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY . /app
RUN pip3 install -r requirements.txt --no-cache-dir