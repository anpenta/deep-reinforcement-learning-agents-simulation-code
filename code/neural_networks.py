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

# Neural Networks Module
# Models of deep reinforcement learning neural networks.

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# DQN: Neural network to approximate action values.
class DQN(nn.Module):

  def __init__(self, observation_space_size, action_space_size, learning_rate, device):
    super().__init__()
    self._hidden_1 = nn.Linear(observation_space_size, 128)
    self._hidden_2 = nn.Linear(128, 128)
    self._output = nn.Linear(128, action_space_size)

    self._optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    self.to(device)

  # The shape of the input to the forward function should be batch size by observation space size.
  def forward(self, observation_batch):
    hidden_batch = F.relu(self._hidden_1(observation_batch))
    hidden_batch = F.relu(self._hidden_2(hidden_batch))
    action_value_batch = self._output(hidden_batch)

    return action_value_batch

  @property
  def optimizer(self):
    return self._optimizer


# DuelingDQN: Neural network to approximate action values through state value and advantage streams.
class DuelingDQN(nn.Module):

  def __init__(self, observation_space_size, action_space_size, learning_rate, device):
    super().__init__()
    self._hidden = nn.Linear(observation_space_size, 128)

    # Split the network into the state value and advantage streams.
    self._state_value_hidden = nn.Linear(128, 64)
    self._state_value_output = nn.Linear(64, 1)
    self._advantage_hidden = nn.Linear(128, 64)
    self._advantage_output = nn.Linear(64, action_space_size)

    self._optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    self.to(device)

  # The shape of the input to the forward function should be batch size by observation space size.
  def forward(self, observation_batch):
    hidden_batch = F.relu(self._hidden(observation_batch))

    state_value_hidden_batch = F.relu(self._state_value_hidden(hidden_batch))
    state_value_batch = self._state_value_output(state_value_hidden_batch)

    advantage_hidden_batch = F.relu(self._advantage_hidden(hidden_batch))
    advantage_batch = self._advantage_output(advantage_hidden_batch)

    # Combine the output of the two streams.
    action_value_batch = state_value_batch + (advantage_batch - advantage_batch.mean(dim=1, keepdim=True))

    return action_value_batch

  @property
  def optimizer(self):
    return self._optimizer


# PolicyNetwork: Neural network to approximate a policy.
class PolicyNetwork(nn.Module):

  def __init__(self, observation_space_size, action_space_size, learning_rate, device):
    super().__init__()
    self._hidden = nn.Linear(observation_space_size, 128)
    self._output = nn.Linear(128, action_space_size)

    self._optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    self.to(device)

  # The shape of the input to the forward function should be batch size by observation space size.
  def forward(self, observation_batch):
    hidden_batch = F.relu(self._hidden(observation_batch))
    action_probability_batch = F.softmax(self._output(hidden_batch), dim=1)

    return action_probability_batch

  @property
  def optimizer(self):
    return self._optimizer


# StateValueNetwork: Neural network to approximate state values.
class StateValueNetwork(nn.Module):

  def __init__(self, observation_space_size, learning_rate, device):
    super().__init__()
    self._hidden = nn.Linear(observation_space_size, 128)
    self._output = nn.Linear(128, 1)

    self._optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    self.to(device)

  # The shape of the input to the forward function should be batch size by observation space size.
  def forward(self, observation_batch):
    hidden_batch = F.relu(self._hidden(observation_batch))
    state_value_batch = self._output(hidden_batch)

    return state_value_batch

  @property
  def optimizer(self):
    return self._optimizer
