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

# Utility Module
# Author: Andreas Pentaliotis
# Email: anpenta01@gmail.com
# Utility functions to run deep reinforcement learning simulations.

import argparse
import os
import pathlib
import random

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

import agents

matplotlib.use("Agg")
plt.rcParams.update({"font.size": 12})


def print_line():
  print("-" * 100)


def create_environment(environment_name):
  if environment_name == "cart-pole":
    return gym.make("CartPole-v0")
  else:
    return None


def create_agent(algorithm_name, observation_space_size, action_space_size):
  if algorithm_name == "deep-q-learning":
    return agents.DeepQLearningAgent(observation_space_size, action_space_size)
  else:
    return None


def control_randomness(seed, environment):
  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  environment.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


def compute_cumulative_moving_average(value_array):
  cumulative_sum = np.cumsum(value_array, axis=None)
  cumulative_length = np.arange(1, value_array.size + 1)
  cumulative_moving_average = np.divide(cumulative_sum, cumulative_length)

  return cumulative_moving_average


def compute_summary_statistics(value_array, axis):
  mean_values = np.mean(value_array, axis)
  standard_deviation_values = np.std(value_array, axis)
  return mean_values, standard_deviation_values


def format_algorithm_name_for_plot(algorithm_name):
  if algorithm_name == "deep-q-learning":
    return "Deep Q-learning"
  else:
    return None


def save_training_experiment_plot(directory_path, mean_experiment_total_rewards,
                                  standard_deviation_experiment_total_rewards, algorithm_name):
  pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)
  print("Saving training plot | Directory path: {}".format(directory_path))

  plt.plot(mean_experiment_total_rewards)
  plt.fill_between(range(mean_experiment_total_rewards.size),
                   mean_experiment_total_rewards - standard_deviation_experiment_total_rewards,
                   mean_experiment_total_rewards + standard_deviation_experiment_total_rewards, alpha=0.5)
  plt.title(format_algorithm_name_for_plot(algorithm_name))
  plt.ylabel("Total reward")
  plt.xlabel("Episode")
  plt.savefig("{}/{}-total-reward".format(directory_path, algorithm_name))
  plt.close()


def save_network_state_dictionary(network_state_dictionary, directory_path, basename):
  pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)
  print("Saving network's state dictionary | Directory path: {}".format(directory_path))
  torch.save(network_state_dictionary, "{}/{}.pt".format(directory_path, basename))


def load_network_state_dictionary(file_path):
  print("Loading network's state dictionary | File path: {}".format(file_path))
  return torch.load(file_path)


def handle_input_argument_errors(input_arguments):
  if input_arguments.simulation_function == "training_episodes":
    if input_arguments.episodes < input_arguments.visual_evaluation_frequency:
      raise ValueError("value of visual_evaluation_frequency is greater than value of episodes")
    elif input_arguments.visual_evaluation_frequency < 0:
      raise ValueError("value of visual_evaluation_frequency is negative")


def parse_input_arguments(algorithm_name_choices=("deep-q-learning",), environment_name_choices=("cart-pole",),
                          experiment_choices=range(1, 11, 1), episode_choices=range(1000, 5001, 500),
                          seed_choices=range(1, 31, 1)):
  parser = argparse.ArgumentParser(prog="simulate", usage="runs deep reinforcement learning simulations")
  subparsers = parser.add_subparsers(dest="simulation_function", help="simulation function to run")
  subparsers.required = True

  add_training_episodes_parser(subparsers, algorithm_name_choices, environment_name_choices, episode_choices,
                               seed_choices)
  add_training_experiments_parser(subparsers, algorithm_name_choices, environment_name_choices, experiment_choices,
                                  episode_choices)

  input_arguments = parser.parse_args()
  handle_input_argument_errors(input_arguments)

  return input_arguments


def add_training_episodes_parser(subparsers, algorithm_name_choices, environment_name_choices, episode_choices,
                                 seed_choices):
  training_episodes_parser = subparsers.add_parser("training_episodes")

  training_episodes_parser.add_argument("algorithm_name", choices=algorithm_name_choices,
                                        help="name of algorithm to be used")
  training_episodes_parser.add_argument("environment_name", choices=environment_name_choices,
                                        help="name of environment to be used")
  training_episodes_parser.add_argument("episodes", type=int, choices=episode_choices,
                                        help="number of training episodes to simulate")
  training_episodes_parser.add_argument("seed", type=int, choices=seed_choices,
                                        help="seed value to control the randomness and get reproducible results")
  training_episodes_parser.add_argument("visual_evaluation_frequency", type=int,
                                        help="episode frequency for visually evaluating agent; should not be greater"
                                             " than value of episodes or negative; 0 suppresses visualization")


def add_training_experiments_parser(subparsers, algorithm_name_choices, environment_name_choices, experiment_choices,
                                    episode_choices):
  training_episodes_parser = subparsers.add_parser("training_experiments")

  training_episodes_parser.add_argument("algorithm_name", choices=algorithm_name_choices,
                                        help="name of algorithm to be used")
  training_episodes_parser.add_argument("environment_name", choices=environment_name_choices,
                                        help="name of environment to be used")
  training_episodes_parser.add_argument("experiments", type=int, choices=experiment_choices,
                                        help="number of training experiments to simulate; each experiment will be"
                                             " simulated using a different random seed; random seed values start from"
                                             " 0 and increase by 1 before a next experiment is simulated")
  training_episodes_parser.add_argument("episodes", type=int, choices=episode_choices,
                                        help="number of training episodes to simulate in each training experiment")
  training_episodes_parser.add_argument("output_path", help="path to directory in which to save output in; directory"
                                                            " will be created if it does not exist")
