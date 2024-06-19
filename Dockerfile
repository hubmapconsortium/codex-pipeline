FROM ubuntu:focal

RUN apt-get -qq update \
    && apt-get -qq install --no-install-recommends --yes \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    unzip \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_24.4.0-0-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# update base environment from yaml file
RUN conda install -n base conda-libmamba-solver \
    && conda config --set solver libmamba
COPY environment.yml /tmp/
RUN conda env update -f /tmp/environment.yml \
    && echo "source activate base" > ~/.bashrc \
    && conda clean --index-cache --tarballs --yes \
    && rm /tmp/environment.yml

ENV PATH /opt/conda/envs/hubmap/bin:$PATH

#Copy fiji from container
COPY --from=hubmap/fiji_bigstitcher:latest /opt/Fiji.app /opt/Fiji.app
ENV PATH /opt/Fiji.app:$PATH

RUN mkdir /output && chmod -R a+rwx /output

WORKDIR /opt
COPY bin /opt

CMD ["/bin/bash"]
