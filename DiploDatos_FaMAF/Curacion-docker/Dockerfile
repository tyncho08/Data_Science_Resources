# Base image
FROM python:3.6-slim

# Updating repository sources
RUN apt-get update && apt-get install -y python3-pip
RUN apt-get install -y python3-setuptools
#RUN apt-get install -y python3-dev

COPY requirements.txt /tmp
ADD . /tmp
WORKDIR /tmp
RUN pip3 install cytoolz==0.8
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Installing spacy Model
RUN python -m spacy download es_core_news_md

# Creating directories
RUN mkdir /data
RUN mkdir /data/Permisos_SF
RUN mkdir /notebooks

# Setting up volumes
VOLUME ["/data", "/data/Permisos_SF", "/notebooks"]

# jupyter
EXPOSE 8888

CMD jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
