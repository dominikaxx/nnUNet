# Parent Image
FROM nvcr.io/nvidia/pytorch:20.11-py3

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


