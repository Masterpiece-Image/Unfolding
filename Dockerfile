# FROM conda/miniconda3:latest

# FROM tensorflow/tensorflow:1.13.1
FROM ubuntu:latest

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        python3 \
        python3-pip

# RUN pip3 install \
#         torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu \
#         typing \
#         numpy \
#         matplotlib \
#         # Dependencies for Jupyter
#         ipykernel \
#         ipywidgets
