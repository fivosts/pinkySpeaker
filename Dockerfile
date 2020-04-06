FROM ubuntu:latest

WORKDIR /home/

RUN apt-get update
RUN apt-get install -y git python3.7 python3-distutils curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN ln -s /usr/bin/python3.7 /usr/bin/python
RUN python get-pip.py
RUN git clone https://github.com/fivosts/pinkySpeaker.git

WORKDIR /home/pinkySpeaker
RUN	python -m pip install -r requirements.txt

CMD ./TfTransformer.sh
