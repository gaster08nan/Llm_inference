FROM python:3.10-slim
LABEL author="HDTHang"

WORKDIR /src

RUN apt-get update && apt-get install -y gcc

COPY requirements.txt /src/requirements.txt
COPY model_utils.py /src/model_utils.py
COPY gui.py /src/gui.py

RUN pip install -r requirements.txt

ENV CC=gcc
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=8080
EXPOSE 8080


CMD ["python","gui.py"]