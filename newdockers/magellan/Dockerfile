FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
RUN pip install numpy scipy
RUN apt-get update && apt-get install -y git build-essential
RUN pip install py_entitymatching numpy==1.20.3
