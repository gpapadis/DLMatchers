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

# install GNEM
RUN conda create -n p39 python=3.9 -y
SHELL ["conda", "run", "-n", "p39", "/bin/bash", "-c"]
RUN pip install pandas torch pytorch_transformers tensorboard sklearn

WORKDIR /methods
RUN git clone https://github.com/nishadi/GNEM.git

# install ZeroER
WORKDIR /methods
RUN git clone https://github.com/nishadi/zeroer.git
WORKDIR /methods/zeroer
RUN conda env create -f environment.yml

# install DITTO
RUN conda create -n p377 python=3.7.7 -y
SHELL ["conda", "run", "-n", "p377", "/bin/bash", "-c"]
RUN conda install pytorch=1.8.0 cudatoolkit=11.1 nvidia-apex -c pytorch -c conda-forge -y

WORKDIR /methods
RUN git clone https://github.com/nishadi/ditto.git
WORKDIR /methods/ditto
RUN sed -i '8d' requirements.txt
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_lg
RUN pip install tensorboardX nltk
RUN python -m nltk.downloader stopwords

# install JointBERT + EMTransformer + DeepMatcher + Magellan
WORKDIR /methods
RUN git clone https://github.com/wbsg-uni-mannheim/jointbert.git
WORKDIR /methods/jointbert
RUN mkdir /methods/jointbert/data/
RUN conda env create -f jointbert.yml
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "jointbert", "/bin/bash", "-c"]

# install HierMatcher
WORKDIR /methods
RUN git clone https://github.com/nishadi/EntityMatcher.git
WORKDIR /methods/EntityMatcher
RUN pip install -r requirements.txt
WORKDIR /methods/EntityMatcher/embedding/
RUN wget https://zenodo.org/record/6466387/files/wiki.en.bin

SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# install EMTransformer
WORKDIR /methods
RUN git clone https://github.com/gpapadis/EMTransformer.git
RUN pip install pytorch_transformers scikit-learn tensorboardX pandas
