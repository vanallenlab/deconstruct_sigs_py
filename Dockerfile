FROM ubuntu:17.10

WORKDIR /

RUN apt-get update && \
	apt-get upgrade -y && \
	apt-get install python-tk -y && \
	apt-get install -y wget vim bzip2 less


RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH

RUN conda update conda -y


WORKDIR /

COPY use_deconstructsigs.py /
COPY deconstructSigs/ /deconstructSigs/

RUN mkdir outputs

RUN apt-get update && apt-get install -y \
  bzip2 \
  g++ \
  libbz2-dev \
  liblzma-dev \
  make \
  ncurses-dev \
  wget \
  zlib1g-dev


ENV SAMTOOLS_INSTALL_DIR=/opt/samtools

WORKDIR /tmp
RUN wget https://github.com/samtools/samtools/releases/download/1.7/samtools-1.7.tar.bz2 && \
  tar --bzip2 -xf samtools-1.7.tar.bz2


WORKDIR /tmp/samtools-1.7
RUN ./configure --enable-plugins --prefix=$SAMTOOLS_INSTALL_DIR && \
  make all all-htslib && \
  make install install-htslib

WORKDIR /
RUN ln -s $SAMTOOLS_INSTALL_DIR/bin/samtools /usr/bin/samtools && \
  rm -rf /tmp/samtools-1.7

WORKDIR /

RUN pip install -r /deconstructSigs/requirements.txt

RUN	echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections && \
	apt-get -o Acquire::https::mirror.ufs.ac.za::Verify-Peer=false install msttcorefonts -y && \
    cp usr/share/fonts/truetype/msttcorefonts/Courier_New.ttf /opt/conda/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/