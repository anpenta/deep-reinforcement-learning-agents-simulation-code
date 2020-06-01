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


import collections
import random


class ReplayMemory:

  def __init__(self, memory_capacity):
    self._memory = collections.deque(maxlen=memory_capacity)

  def store_experience(self, state, action, reward, next_state, done):
    self._memory.append((state, action, reward, next_state, done))

  def can_provide_minibatch(self, batch_size):
    return len(self._memory) >= batch_size

  def sample_minibatch(self, batch_size):
    return random.sample(self._memory, batch_size)
