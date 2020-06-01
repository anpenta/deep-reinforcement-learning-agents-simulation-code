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

  def __init__(self):
    super().__init__()
    self._convolutional_1 = nn.Conv2d(1, 32, 8, stride=4)
    self._convolutional_2 = nn.Conv2d(32, 64, 8, stride=2)
    self._convolutional_3 = nn.Conv2d(64, 64, 3, stride=1)
    self._linear = nn.Linear(64 * 5 * 5, 512)
    self._output = nn.Linear(512, 4)

  def forward(self, x):
    x = x.view(-1, 1, 84, 84)
    x = F.relu(self._convolutional_1(x))
    x = F.relu(self._convolutional_2(x))
    x = F.relu(self._convolutional_3(x))

    x = x.view(-1)
    x = F.relu(self._linear(x))
    x = F.relu(self._output(x))

    return x
