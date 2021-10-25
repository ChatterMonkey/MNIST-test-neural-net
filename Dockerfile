# syntax=docker/dockerfile:1

FROM python:3.8-buster

#FROM ubuntu:21.10

WORKDIR /app

COPY requirements.txt requirements.txt

#RUN pip3 install p5py

#RUN pip3 install pep517

#RUN pip3 install argon2-cffi

RUN pip3 install -r requirements.txt

COPY asimov_evaluation_plots /app/asimov_evaluation_plots
COPY loaded_data /app/loaded_data
COPY loss_graphs /app/loss_graphs
COPY mnist /app/mnist
COPY neuralnets /app/neuralnets
COPY new_phy_graphs /app/new_phy_graphs
COPY new_phy_nets /app/new_phy_nets
COPY non_normalized_loaded_data /app/non_normalized_loaded_data
COPY output_plots /app/output_plots
COPY phy_nets /app/phy_nets
COPY phy_output_plots /app/phy_output_plots
COPY physicsdataset /app/physicsdataset
COPY significance_tests /app/significance_tests
COPY requirements.txt /app
COPY phy_main.py /app
COPY setup.sh /app

CMD [ "python3", "-u" , "phy_main.py"]

#CMD exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"

