# A3C-CarRacingGym
A3C deep reinforcement learning algoritm applied to the CarRacing-v0 gym running in a docker container
# Prerequisites 
* Docker installation (no gpu needed)
# Setting up the docker image
To set up the docker image it can be pulled from the Docker Hub repository or build based on this GitHub repository
## Pulling the image
`docker pull kaland/a3c-carracing-gym`
## Cloning the GitHub repo and building the image
`git clone https://github.com/kaland313/A3C-CarRacingGym.git`

`cd A3C-CarRacingGym`

`docker build --tag=a3c-carracing-gym .`
# Running the image
## If bulit from GitHub 
If the image was built from the github repo, inside the GitHub repo root folder (A3C-CarRacingGym) run:

`docker run -it --rm -v $(pwd)/Scripts:/tf/Scripts -v $(pwd)/Outputs:/tf/Outputs -p 8888:8888 --name=a3c-carracing-gym kaland/a3c-carracing-gym`

## If pulled from Docker Hub
If you didn't pull the github and you'd like to run the trained model included in the docker image, run the following command (note that int this case you can only transfer files from and to the container via Jupyter's file manager): 

`docker run -it --rm -p 8888:8888 --name=a3c-carracing-gym kaland/a3c-carracing-gym`

If you'd like to train the network first, run:

`docker run -it --rm -v $(pwd)/Outputs:/tf/Outputs -p 8888:8888 --name=a3c-carracing-gym kaland/a3c-carracing-gym`

In this case an Output folder will be created in the directory where the above docker command is executed, and the trained model will be saved there. 

## Geting a bash shell inside the container
The container should start with a bash shell, if this is not the case, execute: 

`docker exec -it a3c-carracing-gym /bin/bash`
# Training a model
In order to run train the model enter the following commands to the bash shell of the container:

`cd /tf/Scripts`

`xvfb-run -a -s "-screen 0 640x480x24" python a3c_carracing-v0.py --train --max-eps=20000` 

# Running the trained model
A trained model is included in this repository at: [Outputs/model_CarRacing-v0.h5](https://github.com/kaland313/A3C-CarRacingGym/blob/master/Outputs/model_CarRacing-v0.h5). 

The docker container doesn't have a graphical output at the moment, thus only command line output is given, when running a model: 

`cd /tf/Scripts`

`xvfb-run -a -s "-screen 0 640x480x24" python a3c_carracing-v0.py`

This will load a model from `Outputs/model_CarRacing-v0.h5` and run it for one episode (no eraly termination).

Outside the docker container with an apropriately set up python environment the graphical output of the gym can be viewed by running:

`python a3c_carracing-v0.py`

# Other code scripts for reference
## Code from oguzelibol/CarRacingA3C
`cd UserScripts/oguzelibol-CarRacingA3C`

`xvfb-run -a -s "-screen 0 640x480x24" python a3c.py `

## A3C CartPole script
`cd UserScripts/CartPole`

`xvfb-run -a -s "-screen 0 640x480x24" python a3c_cartpole.py --train`

# Copyright notice
The a3c_cartpole.py source code is based on the Medium post [Deep Reinforcement Learning: Playing CartPole through Asynchronous Advantage Actor Critic (A3C) with tf.keras and eager execution](https://medium.com/tensorflow/deep-reinforcement-learning-playing-cartpole-through-asynchronous-advantage-actor-critic-a3c-7eab2eea5296)


In the Dockerfile I used some parts from the Docker file on https://github.com/ffabi/SemesterProject


Scripts in the oguzelibol-CarRacingA3C folder are from https://github.com/oguzelibol/CarRacingA3C


The mpi_fork function is from https://github.com/garymcintire/mpi_util/
