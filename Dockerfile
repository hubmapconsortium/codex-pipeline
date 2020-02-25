FROM python:3

RUN pip install --no-cache-dir pyyaml

COPY bin /opt
