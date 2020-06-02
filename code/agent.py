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

# Agent Module
# Author: Andreas Pentaliotis
# Email: anpenta01@gmail.com
# Model of a deep reinforcement learning agent.

import numpy as np
import torch
import torch.optim as optim

import dqn
import epsilon_greedy_policy
import replay_memory


class Agent:

  def __init__(self, observation_space_size, action_space_size):
    self._gamma = 0.99
    self._max_epsilon = 1
    self._min_epsilon = 0.05
    self._epsilon_decay_steps = 100000
    self._replay_memory_capacity = 100000
    self._learning_rate = 0.001
    self._batch_size = 32
    self._target_update_frequency = 1000

    self._observation_space_size = observation_space_size
    self._action_space_size = action_space_size
    self._policy = epsilon_greedy_policy.EpsilonGreedyPolicy(self._max_epsilon, self._min_epsilon,
                                                             self._epsilon_decay_steps)
    self._replay_memory = replay_memory.ReplayMemory(self._replay_memory_capacity)
    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self._model = dqn.DQN(observation_space_size, action_space_size).to(self._device)
    self._target_model = dqn.DQN(observation_space_size, action_space_size).to(self._device)
    self._target_model.eval()
    self._update_target_model()
    self._optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)
    self._step = 0

  def _update_target_model(self):
    self._target_model.load_state_dict(self._model.state_dict())

  # TODO: Implement experience replay.
  def _replay_experience(self):
    minibatch = self._replay_memory.sample_minibatch(self._batch_size)

  def select_action(self, observation):
    if np.random.rand() <= self._policy.expose_epsilon():
      action = np.random.randint(self._action_space_size)
    else:
      observation = torch.from_numpy(observation).float().to(self._device)
      action = self._model(observation).argmax().item()

    self._policy.decrease_epsilon()

    return action

  def step(self, observation, action, reward, next_observation, done):
    self._step += 1

    self._replay_memory.store_experience(observation, action, reward, next_observation, done)

    if self._replay_memory.can_provide_minibatch(self._batch_size):
      self._replay_experience()

    if self._step % self._target_update_frequency == 0:
      self._update_target_model()
