FROM nvidia/cuda:11.0.3-devel-ubuntu18.04

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

#install HIF-KAT
WORKDIR /methods
RUN wget https://zenodo.org/record/7020029/files/em2.tar.gz
RUN tar -xf em2.tar.gz
RUN rm em2.tar.gz
RUN conda create -n em --clone /methods/em2
RUN wget https://zenodo.org/record/6466387/files/wiki.en.bin
RUN mkdir /root/.vector_cache
RUN mv wiki.en.bin /root/.vector_cache/wiki.en.bin.bin
WORKDIR /methods
RUN git clone https://github.com/gpapadis/HIF-KAT.git
SHELL ["conda", "run", "-n", "em", "/bin/bash", "-c"]
WORKDIR /methods/HIF-KAT/dataset/magellan_dataset/
RUN python 1.bigtable-attrdrop-ind.py
RUN python 2.mag-table.py
RUN python 3.stat.py
RUN python 4.mag.py
RUN python 5.traditional_feature.py
