# Use the parent image from tensorflow (Ubuntu 16.04. + CUDA 10 + Python 3.5 + TensorFlow 1.13.1 + Jupyter notebook already installed
FROM tensorflow/tensorflow:1.13.1-py3-jupyter

MAINTAINER kaland

RUN apt update --fix-missing
RUN apt install -y libopenmpi-dev ssh sudo nano gcc cmake git
RUN apt install -y build-essential freeglut3 freeglut3-dev libxi-dev libxmu-dev zlib1g-dev
RUN apt install -y xvfb python3-tk python-opengl

RUN pip3 install --upgrade pip
RUN pip3 install mpi4py pympler cma
RUN pip3 install gym 
RUN pip3 install gym[box2d]
RUN pip3 install gym[atari]
RUN pip3 install scikit-image
RUN pip3 install matplotlib
RUN pip3 install jupyterlab

COPY ./DockerSetup/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
COPY ./DockerSetup/jupyter_notebook_config.json /root/.jupyter/jupyter_notebook_config.json
EXPOSE 8888

COPY ./Scripts /tf/UserScripts

RUN echo "PermitRootLogin yes\nSubsystem sftp internal-sftp" > /etc/ssh/sshd_config
RUN echo "root:init" | chpasswd

WORKDIR /tf

# Add entrypoint.sh and other available files to image
# COPY ./DockerSetup/entrypoint.sh /tf

CMD bash
