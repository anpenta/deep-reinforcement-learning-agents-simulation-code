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

# Utility Module
# Author: Andreas Pentaliotis
# Email: anpenta01@gmail.com
# Utility functions to run deep reinforcement learning simulations.

import os
import pathlib
import random

import numpy as np
import torch


def print_line():
  print("-" * 100)


# TODO: Debug control_randomness
def control_randomness(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model_state_dictionary(model_state_dictionary, directory_path, basename):
  pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)
  print("Saving model's state dictionary | Directory path: {}".format(directory_path))
  torch.save(model_state_dictionary, "{}/{}.pt".format(directory_path, basename))


def load_model_state_dictionary(file_path):
  print("Loading model's state dictionary | File path: {}".format(file_path))
  model_state_dictionary = torch.load(file_path)
  return model_state_dictionary
