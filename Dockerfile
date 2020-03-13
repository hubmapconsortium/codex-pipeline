FROM python:3

RUN pip install --no-cache-dir pyyaml \
 && pip install --no-cache-dir aicsimageio \
 && pip install --no-cache-dir tifffile \
 && pip install --no-cache-dir shapely

COPY bin /opt
