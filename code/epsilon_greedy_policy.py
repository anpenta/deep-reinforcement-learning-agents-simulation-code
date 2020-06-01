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

# Epsilon-Greedy Policy Module
# Author: Andreas Pentaliotis
# Email: anpenta01@gmail.com
# Model of an epsilon-greedy policy.


class EpsilonGreedyPolicy:

  def __init__(self, max_epsilon, min_epsilon, epsilon_decay_steps):
    self._max_epsilon = max_epsilon
    self._min_epsilon = min_epsilon
    self._epsilon_decay_steps = epsilon_decay_steps
    self._epsilon_decay = (self._max_epsilon - self._min_epsilon) / self._epsilon_decay_steps

  def compute_epsilon(self, step):
    epsilon = self._max_epsilon - step * self._epsilon_decay
    return max(epsilon, self._min_epsilon)
