FROM python:3.9-slim

WORKDIR /project

COPY ./requirements.txt /project/requirements.txt
RUN pip install -r /project/requirements.txt

COPY . /project/
RUN mkdir -p api/MachineLearning/models

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--root-path", "/ml"]

