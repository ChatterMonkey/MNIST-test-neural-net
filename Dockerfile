# syntax=docker/dockerfile:1

FROM python:3.8-buster

#FROM ubuntu:21.10

WORKDIR /app

COPY requirements.txt requirements.txt

#RUN pip3 install p5py

#RUN pip3 install pep517

#RUN pip3 install argon2-cffi

RUN pip3 install -r requirements.txt

COPY asimov_evaluation_plots /app
COPY loaded_data /app
COPY loss_graphs /app
COPY mnist /app
COPY neuralnets /app
COPY new_phy_graphs /app
COPY new_phy_nets /app
COPY new_phy_roc_curves /app
COPY non_normalized_loaded_data /app
COPY output_plots /app
COPY phy_nets /app
COPY phy_output_plots /app
COPY physicsdataset /app
COPY significance_tests /app
COPY requirements.txt /app
COPY phy_main.py /app


CMD [ "python3", "-u" , "phy_main.py"]

