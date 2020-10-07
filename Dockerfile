FROM python:3


# update and install wget
RUN apt-get -qq update \
    && apt-get -qq install --no-install-recommends --yes \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt
RUN python3 -m pip install numpy
RUN python3 -m pip install -r /opt/requirements.txt \
 && rm -rf /root/.cache/pip

WORKDIR /opt

COPY sprm /opt/sprm
RUN cd sprm \
 && python3 -m pip install . \
 && cd .. \
 && rm -rf sprm

COPY bin /opt
COPY codex_stitching /opt/codex_stitching


#Get imagej
RUN wget --quiet https://downloads.imagej.net/fiji/latest/fiji-linux64.zip -P /tmp/ \
    && unzip /tmp/fiji-linux64.zip -d /opt/ \
    && rm /tmp/fiji-linux64.zip

ENV PATH /opt/Fiji.app:$PATH

# Update imagej
RUN ImageJ-linux64 --headless --update add-update-site BigStitcher https://sites.imagej.net/BigStitcher/ \
&& ImageJ-linux64 --headless --update update


CMD ["/bin/bash"]
