import gym
import time
import numpy as np
import casadi as cs
from jycontrol.ilqr import ILQR


env = gym.make('Pendulum-v1', g=9.81)
obs = env.reset()
while True:
    action = np.array([0.])
    # time.sleep(0.1)
    obs, rewards, dones, info = env.step(action)
    env.render()