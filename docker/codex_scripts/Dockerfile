FROM continuumio/miniconda3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get -y install \
    autoconf \
    automake \
    bison \
    fastqc \
    flex \
    git \
    libfreetype6-dev \
    libxml2-dev \
    libtool \
    m4 \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    zlib1g-dev \
    curl \
 && rm -rf /var/cache/apt/*

COPY docker/codex_scripts/environment.yml /opt/environment.yml
RUN conda env create -f /opt/environment.yml

ENV CONDA_ENV_NAME=codex-scripts
ENV CONDA_PREFIX=/opt/conda/envs/${CONDA_ENV_NAME}
ENV PATH=${CONDA_PREFIX}/bin:$PATH
ENV LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH

#Copy fiji from container
COPY --from=hubmap/fiji_bigstitcher:latest /opt/Fiji.app /opt/Fiji.app
ENV PATH=/opt/Fiji.app:$PATH

RUN mkdir /output && chmod -R a+rwx /output

COPY bin/ /opt

WORKDIR /opt

CMD ["/bin/bash"]