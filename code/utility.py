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

import pathlib

import torch


def save_model_parameters(model, directory_path, basename):
  pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)
  print("Saving model's parameters | Directory path: {}".format(directory_path))
  torch.save(model.state_dict(), "{}/{}.pt".format(directory_path, basename))


def load_model_parameters(model_parameter_path):
  print("Loading model's parameters | Model parameter path: {}".format(model_parameter_path))
  model_parameters = torch.load(model_parameter_path)
  return model_parameters
