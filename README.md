# A3C-CarRacingGym
A3C deep reinforcement learning algoritm applied to the CarRacing-v0 gym running in a docker container
# Prerequisites 
* nvidia-docker installed on the host machine
* CUDA capable GPU with CUDA 10 installed
# Setting up the docker image
To set up the docker image it can be pulled from the Docker Hub repository or build based on this GitHub repository
## Pulling the container
`docker pull kaland/a3c-carracing-gym`
## Cloning the GitHub repo and building the image
`git clone https://github.com/kaland313/A3C-CarRacingGym.git`

`cd A3C-CarRacingGym`

`docker build --tag=a3c-carracing-gym .`
# Running the image
If the image was built from the github repo, inside the GitHub repo root folder (A3C-CarRacingGym) run:

`docker run -it --rm --runtime=nvidia -v $(pwd)/Scripts:/tf/UserScripts -v $(pwd)/Outputs:/tf/Outputs -p 8888:8888 --name=a3c-carracing-gym a3c-carracing-gym`

Otherwise an Output folder will be created in the directory where the above docker command is executed (due to the `v $(pwd)/Outputs:/tf/Outputs` option). 
# Getting a bash shell in the container
The container should start with a bash shell, if this is not the case, execute: 

`docker exec -it a3c-carracing-gym /bin/bash`
# Training the network
## A3C CartPole code based CarRacing script
In order to run train the model enter the following commands to the bash shell of the container:

`cd UserScripts`

`xvfb-run -a -s "-screen 0 640x480x24" python a3c_carracing-v0.py --train`

The current sript only works on a very basic level, only one worker is spawned and the parameters of the optimizer are not selected correctly, thus it converges very slowly/dowesn't converge to the optimal model. 
It's not very meaningful to run a test of the trained model, because the graphical output can not be viewed yet, however it can be done by entering: 

`xvfb-run -a -s "-screen 0 640x480x24" python a3c_cartpole.py`

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
