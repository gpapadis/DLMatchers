FROM nvidia/cuda:11.6.0-devel-ubuntu18.04 

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
ENV LD_LIBRARY_PATH /usr/local/cuda-11.6/lib64:/usr/local/cuda-11.6/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# download datasets
WORKDIR /data
RUN wget https://zenodo.org/record/7233051/files/magellanExistingDatasets.tar.gz
RUN tar -xf magellanExistingDatasets.tar.gz
RUN rm magellanExistingDatasets.tar.gz
RUN wget https://zenodo.org/record/7233051/files/magellanNewDatasets.tar.gz
RUN tar -xf magellanNewDatasets.tar.gz
RUN rm magellanNewDatasets.tar.gz

# install baselines
WORKDIR /methods
RUN git clone https://github.com/gpapadis/thresholdBasedBaselines.git
RUN pip install pandas nltk
RUN python -m nltk.downloader stopwords

# fasttext baseline
RUN pip install gensim scipy numpy
WORKDIR /vectors
RUN wget https://zenodo.org/record/6466387/files/wiki.en.bin

# st-5 baseline
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
RUN pip install pytorch_transformers scikit-learn tensorboardX pandas transformers sentence_transformers
