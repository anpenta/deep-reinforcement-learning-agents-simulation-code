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
import torch.nn.functional as F
import torch.optim as optim

import dqn
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
    self._target_model_update_frequency = 1000

    self._observation_space_size = observation_space_size
    self._action_space_size = action_space_size
    self._replay_memory = replay_memory.ReplayMemory(self._replay_memory_capacity, observation_space_size)
    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self._model = dqn.DQN(observation_space_size, action_space_size).to(self._device)
    self._target_model = dqn.DQN(observation_space_size, action_space_size).to(self._device)
    self._target_model.eval()
    self._update_target_model()
    self._optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)
    self._epsilon = self._max_epsilon
    self._step = 0

  def _update_target_model(self):
    self._target_model.load_state_dict(self._model.state_dict())

  def _replay_experience(self):
    experiences = self._replay_memory.sample_experiences(self._batch_size)
    observation_samples, action_samples, reward_samples, next_observation_samples, done_samples = experiences

    observation_samples = torch.from_numpy(observation_samples).to(self._device)
    action_samples = torch.from_numpy(action_samples).to(self._device)
    reward_samples = torch.from_numpy(reward_samples).to(self._device)
    next_observation_samples = torch.from_numpy(next_observation_samples).to(self._device)
    done_samples = torch.from_numpy(done_samples).to(self._device)

    state_action_values = self._model(observation_samples).gather(1, action_samples.unsqueeze(1))
    next_state_values = self._target_model(next_observation_samples).max(1)[0]
    next_state_values[done_samples] = 0
    update_targets = reward_samples + self._gamma * next_state_values

    loss = F.mse_loss(state_action_values, update_targets)
    self._optimizer.zero_grad()
    loss.backward()
    self._optimizer.step()

  def select_action(self, observation):
    if np.random.rand() <= self._epsilon:
      action = np.random.randint(self._action_space_size)
    else:
      observation = torch.from_numpy(observation).float().to(self._device)
      with torch.no_grad():
        action = self._model(observation).argmax().item()

    return action

  def step(self, observation, action, reward, next_observation, done):
    self._step += 1

    self._replay_memory.store_experience(observation, action, reward, next_observation, done)

    if len(self._replay_memory) >= self._batch_size:
      self._replay_experience()

    if self._step % self._target_model_update_frequency == 0:
      self._update_target_model()

    if self._epsilon > self._min_epsilon:
      self._epsilon -= self._epsilon_decay
