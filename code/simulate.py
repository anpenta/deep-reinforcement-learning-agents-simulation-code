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

# Simulate Module
# Author: Andreas Pentaliotis
# Email: anpenta01@gmail.com
# Module to run deep reinforcement learning simulations.

import simulations
import utility
import agents
import torch

gamma = 0.99
max_epsilon = 1
min_epsilon = 0.01
epsilon_decay_steps = 100000
replay_memory_capacity = 100000
learning_rate = 0.001
batch_size = 128
target_network_update_frequency = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_arguments = utility.parse_input_arguments()

environment = utility.create_environment(input_arguments.environment_name)
observation_space_size = environment.observation_space.shape[0]
action_space_size = environment.action_space.n
agent = agents.Agent(observation_space_size, action_space_size, gamma, max_epsilon, min_epsilon, epsilon_decay_steps,
                     replay_memory_capacity, learning_rate, batch_size, target_network_update_frequency, device)

if input_arguments.simulation_function == "training_episodes":
  utility.control_randomness(input_arguments.seed, environment)

  total_rewards = simulations.simulate_training_episodes(agent, environment, input_arguments.episodes,
                                                         input_arguments.visual_evaluation_frequency,
                                                         input_arguments.verbose)
  total_reward_per_episode = utility.compute_cumulative_moving_average(total_rewards)

  utility.save_training_plot(input_arguments.output_path, total_reward_per_episode, input_arguments.algorithm_name)
