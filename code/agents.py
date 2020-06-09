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

# Agents Module
# Author: Andreas Pentaliotis
# Email: anpenta01@gmail.com
# Models of deep reinforcement learning agents.

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import epsilon_decay_process
import experience_preprocessor
import neural_networks
import replay_memory
import hyperparameters


class Agent:

  def __init__(self, observation_space_size, action_space_size):
    self._observation_space_size = observation_space_size
    self._action_space_size = action_space_size
    self._hyperparameters = hyperparameters.Hyperparameters()
    self._epsilon_decay_process = epsilon_decay_process.EpsilonDecayProcess(self._hyperparameters.max_epsilon,
                                                                            self._hyperparameters.min_epsilon,
                                                                            self._hyperparameters.epsilon_decay_steps)
    self._replay_memory = replay_memory.ReplayMemory(self._hyperparameters.replay_memory_capacity,
                                                     observation_space_size)
    self._experience_preprocessor = experience_preprocessor.ExperiencePreprocessor(self._hyperparameters.device)
    self._step_counter = 0

  def _update_target_network(self):
    raise NotImplementedError

  def _optimize_online_network(self, observation_batch, action_batch, reward_batch, next_observation_batch, done_batch):
    raise NotImplementedError

  def _compute_greedy_action(self, preprocessed_observation):
    raise NotImplementedError

  def select_action(self, observation):
    if np.random.rand() <= self._epsilon_decay_process.epsilon:
      return np.random.randint(self._action_space_size)
    else:
      preprocessed_observation = self._experience_preprocessor.preprocess_observation(observation)
      return self._compute_greedy_action(preprocessed_observation)

  def step(self, observation, action, reward, next_observation, done):
    self._step_counter += 1

    self._replay_memory.store_experience(observation, action, reward, next_observation, done)

    if len(self._replay_memory) >= self._hyperparameters.batch_size:
      experiences = self._replay_memory.sample_experience_minibatch(self._hyperparameters.batch_size)
      preprocessed_experiences = self._experience_preprocessor.preprocess_experience_minibatch(*experiences)
      self._optimize_online_network(*preprocessed_experiences)

    if self._step_counter % self._hyperparameters.target_network_update_frequency == 0:
      self._update_target_network()

    self._epsilon_decay_process.decay_epsilon()


class DeepQLearningAgent(Agent):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._online_network = neural_networks.DQN(self._observation_space_size, self._action_space_size,
                                               self._hyperparameters.device)
    self._target_network = neural_networks.DQN(self._observation_space_size, self._action_space_size,
                                               self._hyperparameters.device)
    self._target_network.eval()
    self._update_target_network()
    self._optimizer = optim.Adam(self._online_network.parameters(), lr=self._hyperparameters.learning_rate)

  def _update_target_network(self):
    self._target_network.load_state_dict(self._online_network.state_dict())

  def _optimize_online_network(self, observation_batch, action_batch, reward_batch, next_observation_batch, done_batch):
    state_action_values = self._online_network(observation_batch).gather(1, action_batch.unsqueeze(1))
    next_state_values = self._target_network(next_observation_batch).max(1)[0]
    next_state_values[done_batch] = 0
    update_targets = (reward_batch + self._hyperparameters.gamma * next_state_values).unsqueeze(1)

    loss = F.mse_loss(state_action_values, update_targets)
    self._optimizer.zero_grad()
    loss.backward()
    self._optimizer.step()

  def _compute_greedy_action(self, preprocessed_observation):
    self._online_network.eval()
    with torch.no_grad():
      action = self._online_network(preprocessed_observation).argmax().item()
    self._online_network.train()

    return action
