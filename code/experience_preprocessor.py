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

# Experience Preprocessor Module
# Author: Andreas Pentaliotis
# Email: anpenta01@gmail.com
# Experience preprocessor for a deep reinforcement learning agent.

import torch


class ExperiencePreprocessor:

  def __init__(self, device):
    self._device = device

  def preprocess_experience_minibatch(self, observation_batch, action_batch, reward_batch, next_observation_batch,
                                      done_batch):
    preprocessed_observation_batch = torch.from_numpy(observation_batch).to(self._device)
    preprocessed_action_batch = torch.from_numpy(action_batch).to(self._device)
    preprocessed_reward_batch = torch.from_numpy(reward_batch).to(self._device)
    preprocessed_next_observation_batch = torch.from_numpy(next_observation_batch).to(self._device)
    preprocessed_done_batch = torch.from_numpy(done_batch).to(self._device)

    return (preprocessed_observation_batch, preprocessed_action_batch, preprocessed_reward_batch,
            preprocessed_next_observation_batch, preprocessed_done_batch)

  def preprocess_observation(self, observation):
    preprocessed_observation = torch.from_numpy(observation).float().to(self._device)

    return preprocessed_observation
