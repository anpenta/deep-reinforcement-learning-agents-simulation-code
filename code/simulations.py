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

import time

import utility


def simulate_visual_test_episode(agent, environment):
  utility.print_line()
  print("Simulating a visual test episode")
  utility.print_line()

  total_reward = 0
  total_time_steps = 0
  observation = environment.reset()
  done = False
  while not done:
    environment.render()
    time.sleep(0.1)

    action = agent.select_action(observation)
    next_observation, reward, done, _ = environment.step(action)

    total_reward += reward
    total_time_steps += 1
    observation = next_observation

  print("Total time steps: {:>4}".format(total_time_steps), sep=" ", end="", flush=True)
  print(" | Total reward gained: {:>5}".format(total_reward))
  print()


def simulate_training_episodes(agent, environment, episodes, visual_evaluation_frequency=0):
  utility.print_line()
  print("Simulating {} training episode(s)".format(episodes))
  utility.print_line()

  for episode in range(1, episodes + 1):
    total_reward = 0
    total_time_steps = 0
    observation = environment.reset()
    done = False
    while not done:
      action = agent.select_action(observation)
      next_observation, reward, done, _ = environment.step(action)
      agent.step(observation, action, reward, next_observation, done)

      total_reward += reward
      total_time_steps += 1
      observation = next_observation

    print("Episode: {:>5}".format(episode), sep=" ", end="", flush=True)
    print(" | Total time steps: {:>4}".format(total_time_steps), sep=" ", end="", flush=True)
    print(" | Total reward gained: {:>5}".format(total_reward))

    if visual_evaluation_frequency and episode % visual_evaluation_frequency == 0:
      print()
      print("Visually evaluating agent after episode {}".format(episode))
      simulate_visual_test_episode(agent, environment)
