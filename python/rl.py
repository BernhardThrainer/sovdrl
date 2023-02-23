
"""Requirement: '$ pip install stable-baselines3[extra]'
"""
import os
import gym
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from action_noise import OrnsteinUhlenbeckActionNoise

# The names of our environments
envs = ["Pendulum-v1", "MountainCarContinuous-v0", "Reacher-v2"]
# number of timesteps we want to train for in each environment respectively
ts = [30_000, 75_000, 40_000]
# our batch size
bs = 64
# our 5 random seeds generated uniformly in range [0,99]
seeds = [24,62,94,19,51]

n_rew = [37,300,50]
pend_rew = np.zeros((625,37))
car_rew = np.zeros((625,300))
reacher_rew = np.zeros((625,200))
rew_pos = [8,-1,2]

# what environment we want to train for (see envs for names)
i = 0


k = 0
j = 0
env_name = envs[i]
timesteps = ts[i]
for s_act in seeds:
    for s_env in seeds:
        for s_sgd in seeds:
            for s_nn in seeds:
                k += 1
                print("####################")
                print("Run " + str(k) + " out of 625")
                print("####################")

                # create file paths
                model_path = os.path.join("training","models",
                                          env_name + "_" + str(timesteps),
                                          str(s_act) + "_" + str(s_env) + "_"
                                          + str(s_sgd) + "_" + str(s_nn))
                log_path = os.path.join("training","logs",
                                        env_name + "_" + str(timesteps),
                                        str(s_act) + "_" + str(s_env) + "_"
                                        + str(s_sgd) + "_" + str(s_nn))
                
                # create Logger
                new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])

                # create environment
                env = gym.make(env_name)
                # set seeds
                env.seed(s_env)
                np.random.seed(s_sgd)
                torch.manual_seed(s_nn)

                # wrap env
                env = Monitor(env)
                env = DummyVecEnv([lambda: env])

                # add actionnoise and create model if mountaincar env
                if i == 1:
                    n_actions = env.action_space.shape[-1]
                    action_noise = OrnsteinUhlenbeckActionNoise(mean = np.zeros(n_actions),
                                                                sigma = 0.5 * np.ones(n_actions),
                                                                seed = s_act)
                    np.random.seed(s_sgd)
                    model = SAC("MlpPolicy", env, action_noise = action_noise,
                                verbose = 1, batch_size = bs)
                # create model for other envs
                else:
                    model = SAC("MlpPolicy", env, verbose = 1, batch_size = bs)
                
                # set seeds
                model.env.seed(s_env)
                model.env.action_space.seed(s_act)
                model.action_space.seed(s_act)

                # set logger
                model.set_logger(new_logger)
                
                # training
                model.learn(total_timesteps = timesteps,
                            tb_log_name = str(s_act) + "_" + str(s_env)
                            + "_" + str(s_sgd) + "_" + str(s_nn))
                # save model
                model.save(model_path)
