#https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker
FROM python:3.9

COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--root-path", "/ml"]
