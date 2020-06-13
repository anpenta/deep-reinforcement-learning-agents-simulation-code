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

# Replay Memories Module
# Author: Andreas Pentaliotis
# Email: anpenta01@gmail.com
# Models of experience replay memories.

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
    self._memory_index = None
    self._batch_indices = None

  def _update_memory_index(self):
    self._memory_index = self._memory_counter % self._memory_capacity
    self._memory_counter += 1

  def _sample_batch_indices(self, batch_size):
    # Sample the batch indices uniformly.
    memory_size = min(self._memory_counter, self._memory_capacity)
    self._batch_indices = np.random.choice(memory_size, batch_size, replace=False)

  def store_experience(self, observation, action, reward, next_observation, done):
    # Update the memory index and store the given experience in memory.
    self._update_memory_index()
    self._observation_memory[self._memory_index] = observation
    self._action_memory[self._memory_index] = action
    self._reward_memory[self._memory_index] = reward
    self._next_observation_memory[self._memory_index] = next_observation
    self._done_memory[self._memory_index] = done

  def sample_experience_batch(self, batch_size):
    # Sample the batch indices and experience batch.
    self._sample_batch_indices(batch_size)
    observation_batch = self._observation_memory[self._batch_indices]
    action_batch = self._action_memory[self._batch_indices]
    reward_batch = self._reward_memory[self._batch_indices]
    next_observation_batch = self._next_observation_memory[self._batch_indices]
    done_batch = self._done_memory[self._batch_indices]

    return observation_batch, action_batch, reward_batch, next_observation_batch, done_batch

  def __len__(self):
    return min(self._memory_counter, self._memory_capacity)


class PrioritizedReplayMemory(ReplayMemory):

  def __init__(self, memory_capacity, observation_space_size, priority_alpha):
    super().__init__(memory_capacity, observation_space_size)
    self._priority_alpha = priority_alpha
    self._priorities = np.zeros(memory_capacity, dtype=np.float32)
    self._max_priority = 1

  def _sample_batch_indices(self, batch_size):
    # Sample the batch indices using proportional prioritization.
    memory_size = min(self._memory_counter, self._memory_capacity)
    exponentiated_priorities = self._priorities[:memory_size] ** self._priority_alpha
    probabilities = exponentiated_priorities / np.sum(exponentiated_priorities)
    self._batch_indices = np.random.choice(memory_size, batch_size, replace=False, p=probabilities)

  def store_experience(self, observation, action, reward, next_observation, done):
    super().store_experience(observation, action, reward, next_observation, done)

    # Assign maximum priority to the given experience because it is new.
    self._priorities[self._memory_index] = self._max_priority

  def update_priorities(self, batch_priorities):
    self._priorities[self._batch_indices] = batch_priorities
    self._max_priority = max(self._max_priority, np.max(self._priorities[self._batch_indices]))
