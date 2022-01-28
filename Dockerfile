FROM python:3.9-slim

ARG api_port
ENV UVICORN_PORT ${api_port}

ARG root_path
ENV UVICORN_ROOT_PATH ${root_path}

WORKDIR /project

COPY ./requirements.txt /project/requirements.txt
RUN pip install -r /project/requirements.txt

COPY . /project/
RUN mkdir -p api/MachineLearning/models

CMD ["sh", "-c" , "uvicorn api.app:app --proxy-headers --host 0.0.0.0 --port ${UVICORN_PORT} --root-path ${UVICORN_ROOT_PATH}"] 