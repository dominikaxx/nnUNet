# Parent Image
FROM nvcr.io/nvidia/pytorch:20.11-py3

RUN pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html

# Installing nnU-Net
ADD https://api.github.com/repos/dominikaxx/nnUNet/git/refs/heads/DP version.json
RUN git clone -b DP https://github.com/dominikaxx/nnUNet.git
#RUN git clone -b DP --single-branch https://github.com/dominikaxx/nnUNet.git

WORKDIR /workspace/nnUNet
RUN pip install -e .
COPY nnUNet_trained_models nnUNet_trained_models/

ENV RESULTS_FOLDER=/workspace/nnUNet/nnUNet_trained_models

# Installing additional libraries
WORKDIR /workspace/
RUN pip3 install --upgrade git+https://github.com/nanohanno/hiddenlayer.git@bugfix/get_trace_graph#egg=hiddenlayer
RUN pip3 install progress
RUN pip3 install graphviz
RUN pip3 install axial_attention

# Setting up User on Image
# Match UID to be same as the one on host machine, run command 'id'
RUN useradd -u 1000 grafika
RUN chown -R grafika:grafika /workspace
USER grafika

# Git Credentials
#RUN git config --global user.name "dominikaxx"
#RUN git config --global user.email "domca.moly@gmail.com"

#ENTRYPOINT ["python3", "/workspace/nnUNet/predikuj.py"]


