FROM python:3

RUN pip install --no-cache-dir pyyaml \
 && pip install --no-cache-dir aicsimageio \
 && pip install --no-cache-dir tifffile \
 && pip install --no-cache-dir pandas \
 && pip install --no-cache-dir sklearn \
 && pip install --no-cache-dir matplotlib

COPY bin /opt

COPY $SPRM_CHECKOUT /opt/sprm
