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

# DQN Module
# Author: Andreas Pentaliotis
# Email: anpenta01@gmail.com
# Model of a deep Q network.

import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

  def __init__(self, state_space_size, action_space_size):
    super().__init__()
    self._linear_1 = nn.Linear(state_space_size, 256)
    self._linear_2 = nn.Linear(256, 256)
    self._output = nn.Linear(256, action_space_size)

  def forward(self, x):
    x = F.relu(self._linear_1(x))
    x = F.relu(self._linear_2(x))
    x = F.relu(self._output(x))

    return x
