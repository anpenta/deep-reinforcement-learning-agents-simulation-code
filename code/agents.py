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

import pathlib

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import networks
import replay_memory


class Agent:

  def __init__(self, observation_space_size, action_space_size):
    self._gamma = 0.99
    self._max_epsilon = 1
    self._min_epsilon = 0.01
    self._epsilon_decay_steps = 100000
    self._epsilon_decay = (self._max_epsilon - self._min_epsilon) / self._epsilon_decay_steps
    self._replay_memory_capacity = 100000
    self._learning_rate = 0.001
    self._batch_size = 128
    self._target_network_update_frequency = 1000

    self._observation_space_size = observation_space_size
    self._action_space_size = action_space_size
    self._replay_memory = replay_memory.ReplayMemory(self._replay_memory_capacity, observation_space_size)
    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self._network = networks.DQN(observation_space_size, action_space_size).to(self._device)
    self._target_network = networks.DQN(observation_space_size, action_space_size).to(self._device)
    self._target_network.eval()
    self._update_target_network()
    self._optimizer = optim.Adam(self._network.parameters(), lr=self._learning_rate)
    self._epsilon = self._max_epsilon
    self._step_counter = 0

  def _update_target_network(self):
    self._target_network.load_state_dict(self._network.state_dict())

  def _preprocess_experiences(self, observation_batch, action_batch, reward_batch, next_observation_batch, done_batch):
    observation_batch = torch.from_numpy(observation_batch).to(self._device)
    action_batch = torch.from_numpy(action_batch).to(self._device)
    reward_batch = torch.from_numpy(reward_batch).to(self._device)
    next_observation_batch = torch.from_numpy(next_observation_batch).to(self._device)
    done_batch = torch.from_numpy(done_batch).to(self._device)

    return observation_batch, action_batch, reward_batch, next_observation_batch, done_batch

  def _compute_loss_arguments(self, observation_batch, action_batch, reward_batch, next_observation_batch, done_batch):
    state_action_values = self._network(observation_batch).gather(1, action_batch.unsqueeze(1))
    next_state_values = self._target_network(next_observation_batch).max(1)[0]
    next_state_values[done_batch] = 0
    update_targets = (reward_batch + self._gamma * next_state_values).unsqueeze(1)

    return state_action_values, update_targets

  def _optimize_network(self, state_action_values, update_targets):
    loss = F.mse_loss(state_action_values, update_targets)
    self._optimizer.zero_grad()
    loss.backward()
    self._optimizer.step()

  def select_action(self, observation):
    if np.random.rand() <= self._epsilon:
      action = np.random.randint(self._action_space_size)
    else:
      observation = torch.from_numpy(observation).float().to(self._device)
      self._network.eval()
      with torch.no_grad():
        action = self._network(observation).argmax().item()
      self._network.train()

    return action

  def step(self, observation, action, reward, next_observation, done):
    self._step_counter += 1

    self._replay_memory.store_experience(observation, action, reward, next_observation, done)

    if len(self._replay_memory) >= self._batch_size:
      experiences = self._replay_memory.sample_experiences(self._batch_size)
      experiences = self._preprocess_experiences(*experiences)
      loss_arguments = self._compute_loss_arguments(*experiences)
      self._optimize_network(*loss_arguments)

    if self._step_counter % self._target_network_update_frequency == 0:
      self._update_target_network()

    if self._epsilon > self._min_epsilon:
      self._epsilon -= self._epsilon_decay
