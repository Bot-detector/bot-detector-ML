#https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker
FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code/app

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "5000"