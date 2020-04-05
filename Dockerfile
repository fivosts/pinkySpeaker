FROM ubuntu:latest

WORKDIR /home/

RUN apt-get update &&\
    apt-get install -y git python3.7 python3-distutils curl &&\
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py &&\
    ln -s /usr/bin/python3.7 /usr/bin/python &&\
    python get-pip.py &&\
    git clone https://github.com/fivosts/pinkySpeaker.git &&\
    cd pinkySpeaker &&\
    python -m pip install -r requirements.txt &&\
    ./TfTransformer.sh
