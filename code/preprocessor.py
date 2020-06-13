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

# Preprocessor Module
# Author: Andreas Pentaliotis
# Email: anpenta01@gmail.com
# Preprocessor for deep reinforcement learning agents.

import torch


class Preprocessor:

  def __init__(self, device):
    self._device = device

  def preprocess_numpy_array(self, numpy_array, dtype=None):
    return torch.from_numpy(numpy_array).to(self._device, dtype=dtype)

  def preprocess_experience_batch(self, observation_batch, action_batch, reward_batch, next_observation_batch,
                                  done_batch):
    # Transform the given experience batch from numpy arrays to torch tensors without changing their dtype.
    preprocessed_observation_batch = self.preprocess_numpy_array(observation_batch)
    preprocessed_action_batch = self.preprocess_numpy_array(action_batch)
    preprocessed_reward_batch = self.preprocess_numpy_array(reward_batch)
    preprocessed_next_observation_batch = self.preprocess_numpy_array(next_observation_batch)
    preprocessed_done_batch = self.preprocess_numpy_array(done_batch)

    return (preprocessed_observation_batch, preprocessed_action_batch, preprocessed_reward_batch,
            preprocessed_next_observation_batch, preprocessed_done_batch)
