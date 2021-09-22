FROM eczech/cytokit:latest

WORKDIR /lab/data/.cytokit
RUN curl https://s3.amazonaws.com/hubmap-pipeline-assets/cytokit-cache.tar | tar -xf -

WORKDIR /opt

COPY setup_data_directory.py /opt
COPY cytokit_wrapper.py /opt
