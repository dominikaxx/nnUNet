# Parent Image
FROM nvcr.io/nvidia/pytorch:20.11-py3

# Environment Variables:
#ENV nnUNet_raw_data_base "/mnt/grafika/nnUNet_raw_data_base"
#ENV nnUNet_preprocessed "/mnt/grafika/nnUNet_preprocessed"
#ENV RESULTS_FOLDER "/mnt/grafika/nnUNet_trained_models"

# Installing nnU-Net
RUN git clone -b DP --single-branch https://github.com/dominikaxx/nnUNet.git
WORKDIR /workspace/nnUNet
RUN pip install -e .
COPY nnUNet_trained_models nnUNet_trained_models/

ENV RESULTS_FOLDER=/workspace/nnUNet/nnUNet_trained_models/

# Installing additional libraries
WORKDIR /workspace/
RUN pip3 install --upgrade git+https://github.com/nanohanno/hiddenlayer.git@bugfix/get_trace_graph#egg=hiddenlayer
RUN pip3 install progress
RUN pip3 install graphviz

# Setting up User on Image
# Match UID to be same as the one on host machine, run command 'id'
RUN useradd -u 1000 grafika
RUN chown -R grafika:grafika nnUNet/
USER grafika

# Git Credentials
#RUN git config --global user.name "dominikaxx"
#RUN git config --global user.email "domca.moly@gmail.com"

ENTRYPOINT ["python3", "/workspace/nnUNet/predikuj.py"]








#FROM nvidia/cuda:11.6.2-base-ubuntu20.04
#RUN apt-get update && apt install -y software-properties-common
#RUN add-apt-repository ppa:deadsnakes/ppa
#RUN apt-get update
#RUN apt install -y python3.8
#RUN apt install -y python3-pip
#RUN apt install -y git
#COPY requirements.txt /usr/local/bin
#RUN pip3 install -r /usr/local/bin/requirements.txt
#RUN pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html
#COPY nnunet /usr/local/bin/nnUNet/
#COPY predikuj.py /usr/local/bin/nnUNet/
#COPY setup.py /usr/local/bin/nnUNet/
#
#COPY nnUNet_trained_models /usr/local/bin/trained_models/
#RUN pip3 install -U setuptools
#RUN pip3 install -e /usr/local/bin/nnUNet
#ENV RESULTS_FOLDER=/usr/local/bin/trained_models/
#COPY predikuj.py /usr/local/bin
#
#ENTRYPOINT ["python3", "/usr/local/bin/predikuj.py"]

#FROM python:slim-buster
#
#RUN apt-get update && apt-get install -y --no-install-recommends apt-utils ca-certificates wget unzip git
#RUN update-ca-certificates
#
#WORKDIR /nnunet
#RUN pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html
#
#RUN pip3 install nnunet
#COPY setup.sh /nnunet
#COPY SegModel.zip /nnunet
#
#RUN mkdir -p /nnunet/data/input/nnUNet_raw_data
#RUN mkdir -p /nnunet/data/input/nnUNet_preprocessed
#RUN mkdir -p /nnunet/data/output
#
#ENV nnUNet_raw_data_base=/nnunet/data/input/
#ENV nnUNet_preprocessed=/nnunet/data/input/nnUNet_preprocessed
#ENV RESULTS_FOLDER=/nnunet/data/output/
#
#RUN nnUNet_install_pretrained_model_from_zip SegModel.zip
#
#COPY predikuj.py /nnunet/

