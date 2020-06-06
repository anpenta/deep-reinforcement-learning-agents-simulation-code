# Copyright (C) 2020 Andreas Pentaliotis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Simulations Module
# Author: Andreas Pentaliotis
# Email: anpenta01@gmail.com
# Reinforcement learning simulation functions.
# In every simulation function we start counting time steps from 0.

import time

import utility
import numpy as np


def simulate_visual_test_episode(agent, environment):
  utility.print_line()
  print("Simulating a visual test episode")
  utility.print_line()

  total_reward = 0
  total_time_steps = 0
  observation = environment.reset()
  done = False
  while not done:
    environment.render()
    time.sleep(0.05)

    action = agent.select_action(observation)
    next_observation, reward, done, _ = environment.step(action)

    total_reward += reward
    total_time_steps += 1
    observation = next_observation

  print("Total time steps: {:>4}".format(total_time_steps), sep=" ", end="", flush=True)
  print(" | Total reward gained: {:>5}".format(total_reward))
  print()


def simulate_training_episodes(agent, environment, episodes, visual_evaluation_frequency=0, verbose=False):
  utility.print_line()
  print("Simulating {} training episode(s)".format(episodes))
  utility.print_line()

  total_rewards = np.zeros(episodes)
  for i in range(episodes):
    total_reward = 0
    total_time_steps = 0
    observation = environment.reset()
    done = False
    while not done:
      action = agent.select_action(observation)
      next_observation, reward, done, _ = environment.step(action)
      agent.step(observation, action, reward, next_observation, done)

      total_reward += reward
      total_time_steps += 1
      observation = next_observation

    total_rewards[i] = total_reward

    if verbose:
      print("Episode: {:>5}".format(i + 1), sep=" ", end="", flush=True)
      print(" | Total time steps: {:>4}".format(total_time_steps), sep=" ", end="", flush=True)
      print(" | Total reward gained: {:>5}".format(total_reward))

    if visual_evaluation_frequency and (i + 1) % visual_evaluation_frequency == 0:
      print()
      print("Visually evaluating agent after episode {}".format(i + 1))
      simulate_visual_test_episode(agent, environment)

  return total_rewards
