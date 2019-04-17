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
In the GitHub repo root folder (A3C-CarRacingGym) run:

`docker run -it --rm --runtime=nvidia -v $(pwd)/Scripts:/tf/UserScripts -v $(pwd)/Outputs:/tf/Outputs -p 8888:8888 --name=a3c-carracing-gym a3c-carracing-gym`
# Getting a bash shell in the container
`docker exec -it a3c-carracing-gym /bin/bash`
# Running script
To the bash shell of the container enter:

`cd UserScripts`

`xvfb-run -s "-screen 0 640x480x24" python a3c_cartpole.py --train`

It's not very meaningful to run a test of the trained model, because the graphical output can not be viewed yet, it can be done by entering: 

`xvfb-run -s "-screen 0 640x480x24" python a3c_cartpole.py`

# Copyright notice
The a3c_cartpole.py source code is based on the Medium post [Deep Reinforcement Learning: Playing CartPole through Asynchronous Advantage Actor Critic (A3C) with tf.keras and eager execution](https://medium.com/tensorflow/deep-reinforcement-learning-playing-cartpole-through-asynchronous-advantage-actor-critic-a3c-7eab2eea5296)
In the Dockerfile I used some elemetns from https://github.com/ffabi/SemesterProject
