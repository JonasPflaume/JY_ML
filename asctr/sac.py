import gym
import numpy as np

from stable_baselines3 import SAC

env = gym.make("Pendulum-v1")

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000, log_interval=4)
model.save("sac_pendulum")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_pendulum")

rewards = []

for epi in range(50):
    rewards.append([])
    done = False
    env = gym.make("Pendulum-v1")
    obs = env.reset()
    while not done:
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      rewards[-1].append(reward)
      # env.render()
epi_rewards = np.array([sum(elem) for elem in rewards])
mean_reward = epi_rewards.mean()
var_reward = epi_rewards.var()
print("Mean episode reward: %.2f" % mean_reward, "Var episode reward: %.2f" % var_reward)