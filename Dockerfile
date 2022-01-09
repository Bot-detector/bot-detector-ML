#https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker
FROM python:3.9

WORKDIR /root

COPY ./requirements.txt /root/requirements.txt
RUN pip install --no-cache-dir -r /root/requirements.txt

COPY . /root/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "6532", "--root-path", "/ml"]
