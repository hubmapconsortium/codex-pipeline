FROM python:3

RUN pip install --no-cache-dir pyyaml
RUN pip install --no-cache-dir aicsimageio
RUN pip install --no-cache-dir tifffile
