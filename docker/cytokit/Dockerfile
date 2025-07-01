FROM eczech/cytokit:latest

WORKDIR /lab/data/.cytokit
RUN curl https://s3.amazonaws.com/hubmap-pipeline-assets/cytokit-cache.tar | tar -xf -

WORKDIR /opt

# Update tensorflow-gpu to version 1.14
RUN pip install --upgrade tensorflow-gpu==1.14.0

COPY setup_data_directory.py /opt
COPY cytokit_wrapper.py /opt
