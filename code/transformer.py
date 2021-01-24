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

# Transformer Module
# Transformer for deep reinforcement learning agents.

import torch


class Transformer:

  def __init__(self, device):
    self._device = device

  def transform_array_to_tensor(self, array, torch_dtype=None):
    return torch.from_numpy(array).to(self._device, dtype=torch_dtype)

  def transform_batch_arrays_to_batch_tensors(self, *batch_arrays):
    # Transform the given batches from numpy arrays to torch tensors without changing their dtype.
    batch_tensors = tuple([self.transform_array_to_tensor(x) for x in batch_arrays])

    return batch_tensors

  @staticmethod
  def transform_tensor_to_array(tensor, numpy_dtype=None):
    if numpy_dtype:
      return tensor.detach().numpy().astype(numpy_dtype)
    else:
      return tensor.detach().numpy()
