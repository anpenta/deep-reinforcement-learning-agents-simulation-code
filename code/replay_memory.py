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

# Replay Memory Module
# Author: Andreas Pentaliotis
# Email: anpenta01@gmail.com
# Model of an experience replay memory.

import random

import numpy as np


class ReplayMemory:

  def __init__(self, memory_capacity, observation_space_size):
    self._memory_capacity = memory_capacity
    self._observation_memory = np.zeros((memory_capacity, observation_space_size), dtype=np.float32)
    self._action_memory = np.zeros(memory_capacity, dtype=np.int64)
    self._reward_memory = np.zeros(memory_capacity, dtype=np.float32)
    self._next_observation_memory = np.zeros((memory_capacity, observation_space_size), dtype=np.float32)
    self._done_memory = np.zeros(memory_capacity, dtype=np.bool)
    self._memory_counter = 0
    self._memory_index = 0

  def __len__(self):
    return min(self._memory_counter, self._memory_capacity)

  def store_experience(self, observation, action, reward, next_observation, done):
    self._memory_index = self._memory_counter % self._memory_capacity

    self._observation_memory[self._memory_index] = observation
    self._action_memory[self._memory_index] = action
    self._reward_memory[self._memory_index] = reward
    self._next_observation_memory[self._memory_index] = next_observation
    self._done_memory[self._memory_index] = done

    self._memory_counter += 1

  def sample_experiences(self, batch_size):
    memory_indices = min(self._memory_counter, self._memory_capacity)
    batch_indices = np.random.choice(memory_indices, batch_size, replace=False)

    observation_samples = self._observation_memory[batch_indices]
    action_samples = self._action_memory[batch_indices]
    reward_samples = self._reward_memory[batch_indices]
    next_observation_samples = self._next_observation_memory[batch_indices]
    done_samples = self._done_memory[batch_indices]

    return observation_samples, action_samples, reward_samples, next_observation_samples, done_samples
