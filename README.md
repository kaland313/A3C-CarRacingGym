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

`docker build --tag=kaland/a3c-carracing-gym .`
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

`xvfb-run -a -s "-screen 0 640x480x24" python a3c_carracing-v0.py --train --max-eps=5000` 

The script can be executed with the following command-line arguments (the default values might change without changing this readme, they can be checked by running the scipt with the `--help` flag): 

```
  -h, --help            show this help message and exit
  --algorithm ALGORITHM
                        Choose between 'a3c' and 'random'. (default: a3c)
  --train               Train our model. (default: False)
  --load-model          Load a trained model for further training. (default:
                        False)
  --test                Test the trained model loaded from the saves
                        directory.Using multiple workers the model will be
                        tested for 100 episodes. (default: False)
  --save-dir SAVE_DIR   Directory in which you desire to save the model.
                        (default: ../Outputs/)
  --workers WORKERS     Number of workers used for training, excluding the
                        master managing the global model. (default: 4)
  --lr LR               Learning rate for the shared optimizer. (default:
                        0.0001)
  --max-eps MAX_EPS     Global maximum number of episodes to run. (default:
                        5000)
  --beta BETA           Entropy regularization coefficient beta. (default:
                        0.0001)
  --save-threshold SAVE_THRESHOLD
                        If a model is loaded, new one will overwrite it only
                        if the achieved score is higher than this (default:
                        300.0)
```

The command line arguments are saved to a file called `args.txt` if a training is started. 
# Demonstarting the model's behaviour (single worker testing)
A trained model is included in this repository at: [Outputs/model_CarRacing-v0.h5](https://github.com/kaland313/A3C-CarRacingGym/blob/master/Outputs/model_CarRacing-v0.h5). 

The docker container doesn't have a graphical output at the moment, thus only command line output is given, when running a model: 

`cd /tf/Scripts`

`xvfb-run -a -s "-screen 0 640x480x24" python a3c_carracing-v0.py`

This will load a model from `Outputs/model_CarRacing-v0.h5`, run it for one episode (with eraly termination) and a gif file showing the states encountered during the episode will be saved to the outputs folder.

Outside the docker container with an apropriately set up python environment the graphical output of the gym can be viewed by running:

`python a3c_carracing-v0.py`

# Test/evaluate the model
To evaluate the model the average episode reward over 100 episodes and its standard deviation is calculated. The evaluaion is carried out on multiple workers by running: 

`xvfb-run -a -s "-screen 0 640x480x24" python a3c_carracing-v0.py --test`

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
