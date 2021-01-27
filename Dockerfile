FROM ubuntu:bionic


RUN apt-get -qq update \
    && apt-get -qq install --no-install-recommends --yes \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# update base environment from yaml file
COPY environment.yml /tmp/
RUN conda env update -f /tmp/environment.yml \
    && echo "source activate base" > ~/.bashrc \
    && conda clean --index-cache --tarballs --yes \
    && rm /tmp/environment.yml

ENV PATH /opt/conda/envs/hubmap/bin:$PATH

WORKDIR /opt

COPY sprm /opt/sprm
RUN cd sprm \
 && pip install . \
 && cd .. \
 && rm -rf sprm

COPY bin /opt
COPY codex_stitching /opt/codex_stitching
RUN mkdir /output && chmod -R a+rwx /output


#Copy fiji from container
COPY --from=vaskivskyi/fiji_bigstitcher:latest /opt/Fiji.app /opt/Fiji.app

ENV PATH /opt/Fiji.app:$PATH


CMD ["/bin/bash"]
