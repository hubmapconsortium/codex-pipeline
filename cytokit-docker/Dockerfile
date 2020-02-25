FROM eczech/cytokit:latest

COPY setup_data_directory.py /opt
COPY cytokit_wrapper.py /opt

WORKDIR /lab/data/.cytokit
RUN curl https://giygas.compbio.cs.cmu.edu/cytokit-cache.tar | tar -xf -

WORKDIR /opt
