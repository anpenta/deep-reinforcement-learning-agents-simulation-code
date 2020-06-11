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

import numpy as np

import utility


# simulate_visual_test_episode: Simulates a visual test episode where the given agent interacts with the given
# environment without learning.
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


# simulate_training_episodes: Simulates the given number of episodes where the given agent interacts with the given
# environment and learns. Returns the total reward gained from the agent in each episode.
def simulate_training_episodes(agent, environment, episodes, visual_evaluation_frequency=0, verbose=False):
  utility.print_line()
  print("Simulating {} training episode(s)".format(episodes))
  utility.print_line()

  episode_total_rewards = np.zeros(episodes)
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

    episode_total_rewards[i] = total_reward

    if verbose:
      print("Episode: {:>5}".format(i + 1), sep=" ", end="", flush=True)
      print(" | Total time steps: {:>4}".format(total_time_steps), sep=" ", end="", flush=True)
      print(" | Total reward gained: {:>5}".format(total_reward))

    if visual_evaluation_frequency and (i + 1) % visual_evaluation_frequency == 0:
      print()
      print("Visually evaluating agent after episode {}".format(i + 1))
      simulate_visual_test_episode(agent, environment)

  return episode_total_rewards


# simulate_training_experiments: Simulates the given number of experiments where each experiment is run with
# a different random seed. In each experiment, a new agent that uses the given algorithm name is created,
# interacts with the given environment, and learns for the given number of episodes. Returns the total reward
# gained from each agent in each episode of each experiment.
def simulate_training_experiments(algorithm_name, environment, experiments, episodes):
  utility.print_line()
  print("Simulating {} training experiment(s) of {} episode(s) each".format(experiments, episodes))
  utility.print_line()

  observation_space_size, action_space_size = utility.compute_environment_space_sizes(environment)

  experiment_total_rewards = np.zeros((experiments, episodes))
  for i in range(experiments):
    print("Simulating experiment: {:>5}/{}".format(i + 1, experiments))
    agent = utility.create_agent(algorithm_name, observation_space_size, action_space_size)
    utility.control_randomness(i, environment)
    experiment_total_rewards[i] = simulate_training_episodes(agent, environment, episodes)

  return experiment_total_rewards
