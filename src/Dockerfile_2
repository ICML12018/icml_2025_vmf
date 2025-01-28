FROM registry.deez.re/research/python-audio-gpu-11-2:latest

# libgoogle-perftools: cf. https://stackoverflow.com/questions/57180479/tensorflow-serving-oomkilled-or-evicted-pod-with-kubernetes
# poetry: wait until poetry is installed in python-audio img
RUN apt-get update && apt-get install -y ffmpeg zip \
    curl \
    libgoogle-perftools4 \
    && mkdir -p /var/cache \
    && mkdir -p /var/probes && touch /var/probes/ready \
    && pip install --upgrade --no-cache-dir poetry \
    && apt-get install -y graphviz

# GSUTIL SDK
# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
# Installing the package
RUN mkdir -p /usr/local/gcloud \
    && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
    && /usr/local/gcloud/google-cloud-sdk/install.sh
# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

RUN gcloud components update

# PYTHON Libraries
RUN pip install google-cloud-bigquery\
    pyarrow \
    fastparquet \
    google-cloud-storage\
    google-auth \
    jupyterlab \
    notebook \
    ipython \
    hvac \
    pymysql  \
    dbutils \
    protobuf \
    setuptools \
    audioread \
    gin-config \
    tensorflow==2.9.1 \
    tensorflow-io==0.26.0 \
    torchsummary \
    torchmetrics \
    matplotlib \
    scipy==1.7.3 \
    weightwatcher \
    tqdm \
    GPUtil \
    pandas \
    einops \
    implicit \
    graphviz \
    tables \
    psutil \
    pympler \
    seaborn \
    cupy-cuda12x
RUN pip install torch==2.0.0 \
    torchvision \
    torchaudio

RUN pip install --extra-index-url https://artifacts.deez.re/repository/python-research/simple --trusted-host artifacts.deez.re deezer-audio[resampling] deezer-environment deezer-datasource

ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
ENV PYTHONPATH "${PYTHONPATH}:/home/deep_rl"
USER deezer
CMD /bin/bash
