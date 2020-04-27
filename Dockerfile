FROM python:3

COPY requirements.txt /opt
RUN python3 -m pip install -r /opt/requirements.txt \
 && rm -rf /root/.cache/pip

COPY bin /opt

COPY $SPRM_CHECKOUT /opt/sprm
