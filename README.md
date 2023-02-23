# Sources of Variance in Deep Reinforcemnt Learning

## Description

This is the GitHub repository corresponting to the bacholor thesis
'Sources of Variance in Deep Reinforcement Learning' by Bernhard Thrainer
and Supervisors Dr. Sebastien Court and Dipl.-Ing. Jakob Hollenstein. It
contains the python files used during the training of the Soft Actor-Critic
algorithm and the data generated during said training.

## Requirements

run `$ pip install stable-baselines3[extra]` in your shell to install the depenencies.

## Structure

The `python/rl.py` file is used to train our agent and generate the data. It uses the `action_noise.py`
file during the mountain-car environment training. The training data can be found under `data/environment/seed/progress.csv`.
This data is read by the `data/stats.py` file which then generates the plots and ANOVAS used in the thesis.
