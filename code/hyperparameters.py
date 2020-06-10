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

# Hyperparameters Module
# Author: Andreas Pentaliotis
# Email: anpenta01@gmail.com
# Hyperparameters for a deep reinforcement learning agent.

import torch


class Hyperparameters:

  def __init__(self):
    self._gamma = 0.99
    self._max_epsilon = 1
    self._min_epsilon = 0.01
    self._epsilon_decay_steps = 50000
    self._replay_memory_capacity = 50000
    self._learning_rate = 0.001
    self._batch_size = 32
    self._target_network_update_frequency = 1000
    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  @property
  def gamma(self):
    return self._gamma

  @property
  def max_epsilon(self):
    return self._max_epsilon

  @property
  def min_epsilon(self):
    return self._min_epsilon

  @property
  def epsilon_decay_steps(self):
    return self._epsilon_decay_steps

  @property
  def replay_memory_capacity(self):
    return self._replay_memory_capacity

  @property
  def learning_rate(self):
    return self._learning_rate

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def target_network_update_frequency(self):
    return self._target_network_update_frequency

  @property
  def device(self):
    return self._device
