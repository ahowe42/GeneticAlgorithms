FROM ubuntu:latest

WORKDIR /usr/src/

# install git & clone the repo - just the most recent commit to master
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && export GIT_SSL_NO_VERIFY=1 \
    && git clone --branch master --single-branch --depth 1 https://github.com/ahowe42/GeneticAlgorithms.git

# don't need this because the git clone got it
#COPY ./requirements.txt ./

# install python & the requisite packages
RUN apt-get install -y --no-install-recommends python3.8 pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r /usr/src/GeneticAlgorithms/docker/requirements.txt
 
EXPOSE 8888

CMD ["jupyter", "lab", "--no-browser", "--allow-root", "--ip=0.0.0.0", "--ServerApp.token=''", "--ServerApp.password=''"]
