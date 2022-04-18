FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# install JointBERT + EMTransformer + DeepMatcher + Magellan
WORKDIR /methods
RUN git clone https://github.com/wbsg-uni-mannheim/jointbert.git
WORKDIR /methods/jointbert
RUN conda env create -f jointbert.yml
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "jointbert", "/bin/bash", "-c"]

# install HierMatcher
WORKDIR /methods
RUN git clone https://github.com/casnlu/EntityMatcher.git
WORKDIR /methods/EntityMatcher
RUN pip install -r requirements.txt
WORKDIR /methods/EntityMatcher/embedding/
RUN wget https://zenodo.org/record/6466387/files/wiki.en.bin

# get GNEM
WORKDIR /methods
RUN git clone https://github.com/ChenRunjin/GNEM.git
RUN pip install pytorch_transformers

# install AutoML4EM
WORKDIR /methods
RUN git clone https://github.com/softlab-unimore/automl-for-em.git
RUN curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install
RUN pip install transformers==3 auto-sklearn h2o
