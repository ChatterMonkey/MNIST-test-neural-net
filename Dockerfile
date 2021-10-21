# syntax=docker/dockerfile:1

FROM python:3.8-buster

#FROM ubuntu:21.10

WORKDIR /app

COPY requirements.txt requirements.txt

#RUN pip3 install p5py

#RUN pip3 install pep517

#RUN pip3 install argon2-cffi

RUN pip3 install -r requirements.txt

COPY . /app


CMD [ "python3", "-u" , "phy_main.py"]

