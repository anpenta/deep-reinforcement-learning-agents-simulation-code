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
# Hyperparameters for deep reinforcement learning agents.

import torch


class Hyperparameters:

  def __init__(self):
    # Initialize shared hyperparameters.
    self._gamma = 0.99
    self._learning_rate = 0.001
    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize hyperparameters for deep Q-learning variants.
    self._max_epsilon = 0.5
    self._min_epsilon = 0.001
    self._epsilon_decay_steps = 10000
    self._memory_capacity = 50000
    self._batch_size = 32
    self._target_network_update_frequency = 500

    # Initialize hyperparameters for deep Q-learning with prioritized experience replay.
    self._priority_alpha = 0.4
    self._priority_epsilon = 0.00001
    self._min_priority_beta = 0.8
    self._max_priority_beta = 1
    self._priority_beta_growth_steps = 5000

    # Initialize hyperparameters for advantage actor critic (A2C).
    self._network_optimization_frequency = 30
    self._entropy_beta = 0.01

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
  def memory_capacity(self):
    return self._memory_capacity

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

  @property
  def priority_alpha(self):
    return self._priority_alpha

  @property
  def priority_epsilon(self):
    return self._priority_epsilon

  @property
  def min_priority_beta(self):
    return self._min_priority_beta

  @property
  def max_priority_beta(self):
    return self._max_priority_beta

  @property
  def priority_beta_growth_steps(self):
    return self._priority_beta_growth_steps

  @property
  def network_optimization_frequency(self):
    return self._network_optimization_frequency

  @property
  def entropy_beta(self):
    return self._entropy_beta
