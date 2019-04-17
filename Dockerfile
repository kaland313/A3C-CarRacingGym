# Use the parent image from tensorflow (Ubuntu 16.04. + CUDA 10 + Python 3.5 + TensorFlow 1.13.1 + Jupyter notebook already installed
FROM tensorflow/tensorflow:1.13.1-gpu-py3-jupyter

MAINTAINER kaland

RUN apt update --fix-missing
RUN apt install -y libopenmpi-dev ssh sudo nano gcc cmake git
RUN apt install -y xvfb python3-tk python-opengl

RUN pip3 install --upgrade pip
RUN pip3 install mpi4py pympler
RUN pip3 install jupyterlab
RUN pip3 install gym

COPY ./DockerSetup/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
COPY ./Scripts /tf/UserScripts

RUN echo "PermitRootLogin yes\nSubsystem sftp internal-sftp" > /etc/ssh/sshd_config
RUN echo "root:init" | chpasswd

WORKDIR /tf

CMD service ssh start
CMD jupyter lab

