FROM python:3

COPY requirements.txt /opt
RUN python3 -m pip install -r /opt/requirements.txt \
 && rm -rf /root/.cache/pip

WORKDIR /opt

COPY sprm /opt/sprm
RUN cd sprm \
 && python3 -m pip install . \
 && cd .. \
 && rm -rf sprm

COPY bin /opt

CMD ["/bin/bash"]
