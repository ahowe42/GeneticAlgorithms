FROM ubuntu:latest
#FROM alpine:latest

WORKDIR /usr/src/

COPY ./requirements.txt ./

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.8 pip git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt \
    && export GIT_SSL_NO_VERIFY=1 \
    && git clone --branch master --single-branch --depth 1 https://github.com/ahowe42/GeneticAlgorithms.git

ENV NB_TOKEN=42
ENV NB_PASSWORD=42
 
EXPOSE 8888

CMD ["jupyter", "lab", "--no-browser", "--allow-root", "--ip=0.0.0.0", "--ServerApp.token=''", "--ServerApp.password=''"]