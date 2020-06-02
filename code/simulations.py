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

# Simulations Module
# Author: Andreas Pentaliotis
# Email: anpenta01@gmail.com
# Helper functions to run reinforcement learning simulations.
# In every simulation function we start counting time steps from 0.

import utility


def simulate_training_episodes(agent, environment, episodes, visualize=False):
  utility.print_line()
  print("Simulating {} training episode(s)".format(episodes))
  utility.print_line()

  for episode in range(1, episodes + 1):
    total_reward = 0
    total_time_steps = 0
    state = environment.reset()
    done = False
    while not done:
      if visualize:
        environment.render()

      action = agent.select_action(state)
      next_state, reward, done, _ = environment.step(action)
      agent.step(state, action, reward, next_state, done)

      total_reward += reward
      total_time_steps += 1
      state = next_state

    print("Episode: {:>5}".format(episode), sep=" ", end="", flush=True)
    print(" | Total time steps: {:>4}".format(total_time_steps), sep=" ", end="", flush=True)
    print(" | Total reward gained: {:>5}".format(total_reward))
