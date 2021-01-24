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

# Agents Module
# Models of deep reinforcement learning agents.

import torch
import torch.nn.functional as F

import annealing_processes
import hyperparameters
import memories
import neural_networks
import transformer


# DeepQLearningAgent: Agent that uses standard deep Q-learning.
class DeepQLearningAgent:

  def __init__(self, observation_space_size, action_space_size):
    self._hyperparameters = hyperparameters.Hyperparameters()
    self._epsilon_decay_process = annealing_processes.EpsilonDecayProcess(self._hyperparameters.max_epsilon,
                                                                          self._hyperparameters.min_epsilon,
                                                                          self._hyperparameters.epsilon_decay_steps)
    self._replay_memory = memories.ReplayMemory(self._hyperparameters.memory_capacity, observation_space_size)
    self._transformer = transformer.Transformer(self._hyperparameters.device)
    self._online_network = neural_networks.DQN(observation_space_size, action_space_size,
                                               self._hyperparameters.learning_rate, self._hyperparameters.device)
    self._target_network = neural_networks.DQN(observation_space_size, action_space_size,
                                               self._hyperparameters.learning_rate, self._hyperparameters.device)
    self._target_network.eval()
    self._update_target_network()
    self._action_space_size = action_space_size
    self._step_counter = 0

  def _update_target_network(self):
    self._target_network.load_state_dict(self._online_network.state_dict())

  def _compute_greedy_action(self, transformed_observation):
    self._online_network.eval()
    with torch.no_grad():
      # Reshape the transformed observation to a batch of one observation before providing it to the network.
      transformed_observation_batch = transformed_observation.unsqueeze(0)
      action = self._online_network(transformed_observation_batch).argmax(1).item()
    self._online_network.train()

    return action

  def _compute_loss_arguments(self, transformed_observation_batch, transformed_action_batch, transformed_reward_batch,
                              transformed_next_observation_batch, transformed_done_batch):
    # Compute the predictions.
    state_value_batch = self._online_network(transformed_observation_batch)
    state_action_value_batch = state_value_batch.gather(1, transformed_action_batch.unsqueeze(1)).squeeze(1)

    # Compute the update targets.
    with torch.no_grad():
      next_state_action_value_batch = self._target_network(transformed_next_observation_batch).max(1)[0]
      next_state_action_value_batch[transformed_done_batch] = 0
      update_target_batch = transformed_reward_batch + self._hyperparameters.gamma * next_state_action_value_batch

    return state_action_value_batch, update_target_batch

  def _optimize_online_network(self, state_action_value_batch, update_target_batch):
    loss = F.mse_loss(state_action_value_batch, update_target_batch)
    self._online_network.optimizer.zero_grad()
    loss.backward()
    self._online_network.optimizer.step()

  def select_action(self, observation):
    # Use an epsilon-greedy policy for action selection based on the epsilon decay process.
    if torch.rand(1).item() <= self._epsilon_decay_process.epsilon:
      return torch.randint(self._action_space_size, (1,)).item()
    else:
      transformed_observation = self._transformer.transform_array_to_tensor(observation, torch_dtype=torch.float32)
      return self._compute_greedy_action(transformed_observation)

  def step(self, observation, action, reward, next_observation, done):
    self._step_counter += 1

    self._replay_memory.store_experience(observation, action, reward, next_observation, done)

    if len(self._replay_memory) >= self._hyperparameters.batch_size:
      experience_batch = self._replay_memory.sample_experience_batch(self._hyperparameters.batch_size)
      transformed_experience_batch = self._transformer.transform_batch_arrays_to_batch_tensors(*experience_batch)
      loss_arguments = self._compute_loss_arguments(*transformed_experience_batch)
      self._optimize_online_network(*loss_arguments)

    if self._step_counter % self._hyperparameters.target_network_update_frequency == 0:
      self._update_target_network()

    self._epsilon_decay_process.decay_epsilon()


# DoubleDeepQLearningAgent: Agent that uses double deep Q-learning.
class DoubleDeepQLearningAgent(DeepQLearningAgent):

  def __init__(self, observation_space_size, action_space_size):
    super().__init__(observation_space_size, action_space_size)

  def _compute_loss_arguments(self, transformed_observation_batch, transformed_action_batch, transformed_reward_batch,
                              transformed_next_observation_batch, transformed_done_batch):
    # Compute the predictions.
    state_value_batch = self._online_network(transformed_observation_batch)
    state_action_value_batch = state_value_batch.gather(1, transformed_action_batch.unsqueeze(1)).squeeze(1)

    # Compute the update targets. Determine the maximizing actions for the next state using the online network
    # and evaluate them using the target network.
    self._online_network.eval()
    with torch.no_grad():
      next_state_action_batch = self._online_network(transformed_next_observation_batch).argmax(1)
      next_state_value_batch = self._target_network(transformed_next_observation_batch)
      next_state_action_value_batch = next_state_value_batch.gather(1, next_state_action_batch.unsqueeze(1)).squeeze(1)
      next_state_action_value_batch[transformed_done_batch] = 0
      update_target_batch = transformed_reward_batch + self._hyperparameters.gamma * next_state_action_value_batch
    self._online_network.train()

    return state_action_value_batch, update_target_batch


# PrioritizedDeepQLearningAgent: Agent that uses deep Q-learning with prioritized experience replay.
class PrioritizedDeepQLearningAgent(DeepQLearningAgent):

  def __init__(self, observation_space_size, action_space_size):
    super().__init__(observation_space_size, action_space_size)
    self._priority_beta_growth_process = annealing_processes.PriorityBetaGrowthProcess(
      self._hyperparameters.min_priority_beta,
      self._hyperparameters.max_priority_beta,
      self._hyperparameters.priority_beta_growth_steps)
    self._replay_memory = memories.PrioritizedReplayMemory(self._hyperparameters.memory_capacity,
                                                           observation_space_size,
                                                           self._hyperparameters.priority_alpha)

  def _compute_batch_priorities(self, state_action_values, update_targets):
    with torch.no_grad():
      batch_priorities = torch.abs(state_action_values - update_targets) + self._hyperparameters.priority_epsilon

    return batch_priorities

  def _optimize_online_network(self, state_action_value_batch, update_target_batch):
    # Compute the normalized importance sampling weights of the most recently sampled experience batch
    # and use them to compute the loss.
    priority_beta = self._priority_beta_growth_process.priority_beta
    normalized_batch_weights = self._replay_memory.compute_normalized_importance_sampling_batch_weights(priority_beta)
    transformed_normalized_batch_weights = self._transformer.transform_array_to_tensor(normalized_batch_weights)
    loss_batch = F.mse_loss(state_action_value_batch, update_target_batch, reduction="none")
    loss = (transformed_normalized_batch_weights * loss_batch).mean()

    self._online_network.optimizer.zero_grad()
    loss.backward()
    self._online_network.optimizer.step()

  def step(self, observation, action, reward, next_observation, done):
    self._step_counter += 1

    self._replay_memory.store_experience(observation, action, reward, next_observation, done)

    if len(self._replay_memory) >= self._hyperparameters.batch_size:
      experience_batch = self._replay_memory.sample_experience_batch(self._hyperparameters.batch_size)
      transformed_experience_batch = self._transformer.transform_batch_arrays_to_batch_tensors(*experience_batch)
      loss_arguments = self._compute_loss_arguments(*transformed_experience_batch)
      self._optimize_online_network(*loss_arguments)

      # Compute the batch priorities and use them to update the priorities in the prioritized replay memory.
      batch_priorities = self._compute_batch_priorities(*loss_arguments)
      self._replay_memory.update_priorities(batch_priorities)

      # Grow priority beta in each learning step.
      self._priority_beta_growth_process.grow_priority_beta()

    if self._step_counter % self._hyperparameters.target_network_update_frequency == 0:
      self._update_target_network()

    self._epsilon_decay_process.decay_epsilon()


# DuelingDeepQLearningAgent: Agent that uses dueling deep Q-learning.
class DuelingDeepQLearningAgent(DeepQLearningAgent):

  def __init__(self, observation_space_size, action_space_size):
    super().__init__(observation_space_size, action_space_size)
    self._online_network = neural_networks.DuelingDQN(observation_space_size, action_space_size,
                                                      self._hyperparameters.learning_rate,
                                                      self._hyperparameters.device)
    self._target_network = neural_networks.DuelingDQN(observation_space_size, action_space_size,
                                                      self._hyperparameters.learning_rate,
                                                      self._hyperparameters.device)
    self._target_network.eval()
    self._update_target_network()


# ZeroBaselineVanillaPolicyGradientAgent: Agent that uses vanilla policy gradient (REINFORCE) with zero as a baseline.
class ZeroBaselineVanillaPolicyGradientAgent:

  def __init__(self, observation_space_size, action_space_size):
    self._hyperparameters = hyperparameters.Hyperparameters()
    self._trajectory_memory = memories.TrajectoryMemory()
    self._transformer = transformer.Transformer(self._hyperparameters.device)
    self._policy_network = neural_networks.PolicyNetwork(observation_space_size, action_space_size,
                                                         self._hyperparameters.learning_rate,
                                                         self._hyperparameters.device)

  def _compute_action_probabilities(self, transformed_observation):
    self._policy_network.eval()
    with torch.no_grad():
      # Reshape the transformed observation to a batch of one observation before providing it to the network.
      transformed_observation_batch = transformed_observation.unsqueeze(0)
      action_probabilities = self._policy_network(transformed_observation_batch).squeeze(0)
    self._policy_network.train()

    return action_probabilities

  def _compute_discounted_return_batch(self, transformed_reward_batch):
    transformed_reward_batch_size = transformed_reward_batch.size(0)
    discounted_return_batch = torch.empty(transformed_reward_batch_size)
    discounted_return = 0
    for i in torch.flip(torch.arange(transformed_reward_batch_size), dims=(0,)):
      discounted_return = transformed_reward_batch[i] + self._hyperparameters.gamma * discounted_return
      discounted_return_batch[i] = discounted_return

    return discounted_return_batch

  def _compute_loss_arguments(self, transformed_observation_batch, transformed_action_batch, transformed_reward_batch):
    # Compute the predictions.
    state_probability_batch = self._policy_network(transformed_observation_batch)
    state_action_probability_batch = state_probability_batch.gather(1, transformed_action_batch.unsqueeze(1)).squeeze(1)
    state_action_log_probability_batch = torch.log(state_action_probability_batch)

    # Compute the discounted returns.
    discounted_return_batch = self._compute_discounted_return_batch(transformed_reward_batch)

    return state_action_log_probability_batch, discounted_return_batch

  def _optimize_policy_network(self, state_action_log_probability_batch, discounted_return_batch):
    loss = -torch.sum(state_action_log_probability_batch * discounted_return_batch)
    self._policy_network.optimizer.zero_grad()
    loss.backward()
    self._policy_network.optimizer.step()

  def select_action(self, observation):
    # Use the action probabilities of the current policy for action selection.
    transformed_observation = self._transformer.transform_array_to_tensor(observation, torch_dtype=torch.float32)
    action_probabilities = self._compute_action_probabilities(transformed_observation)
    action_probability_distribution = torch.distributions.Categorical(action_probabilities)
    action = action_probability_distribution.sample().item()

    return action

  # Keep a placeholder parameter in the step function for compatibility with the other agents.
  def step(self, observation, action, reward, _, done):
    self._trajectory_memory.update_trajectory_batch(observation, action, reward)

    if done:
      trajectory_batch = self._trajectory_memory.retrieve_trajectory_batch()
      transformed_trajectory_batch = self._transformer.transform_batch_arrays_to_batch_tensors(*trajectory_batch)
      loss_arguments = self._compute_loss_arguments(*transformed_trajectory_batch)
      self._optimize_policy_network(*loss_arguments)
      self._trajectory_memory.delete_trajectory_batch()


# StateValueBaselineVanillaPolicyGradientAgent: Agent that uses vanilla policy gradient (REINFORCE) with the
# state value as a baseline.
class StateValueBaselineVanillaPolicyGradientAgent(ZeroBaselineVanillaPolicyGradientAgent):

  def __init__(self, observation_space_size, action_space_size):
    super().__init__(observation_space_size, action_space_size)
    self._state_value_network = neural_networks.StateValueNetwork(observation_space_size,
                                                                  self._hyperparameters.learning_rate,
                                                                  self._hyperparameters.device)

  def _compute_loss_arguments(self, transformed_observation_batch, transformed_action_batch, transformed_reward_batch):
    # Compute the predictions for both networks.
    state_probability_batch = self._policy_network(transformed_observation_batch)
    state_action_probability_batch = state_probability_batch.gather(1, transformed_action_batch.unsqueeze(1)).squeeze(1)
    state_action_log_probability_batch = torch.log(state_action_probability_batch)
    state_value_batch = self._state_value_network(transformed_observation_batch).squeeze(1)

    # Compute the discounted returns and advantage values.
    self._state_value_network.eval()
    with torch.no_grad():
      discounted_return_batch = self._compute_discounted_return_batch(transformed_reward_batch)
      baseline_state_value_batch = self._state_value_network(transformed_observation_batch).squeeze(1)
      advantage_value_batch = discounted_return_batch - baseline_state_value_batch
    self._state_value_network.train()

    return state_action_log_probability_batch, state_value_batch, advantage_value_batch, discounted_return_batch

  def _optimize_networks(self, state_action_log_probability_batch, state_value_batch, advantage_value_batch,
                         discounted_return_batch):
    # Optimize the policy network.
    policy_loss = -torch.sum(state_action_log_probability_batch * advantage_value_batch)
    self._policy_network.optimizer.zero_grad()
    policy_loss.backward()
    self._policy_network.optimizer.step()

    # Optimize the state value network.
    state_value_loss = F.mse_loss(state_value_batch, discounted_return_batch)
    self._state_value_network.optimizer.zero_grad()
    state_value_loss.backward()
    self._state_value_network.optimizer.step()

  # Keep a placeholder parameter in the step function for compatibility with the other agents.
  def step(self, observation, action, reward, _, done):
    self._trajectory_memory.update_trajectory_batch(observation, action, reward)

    if done:
      trajectory_batch = self._trajectory_memory.retrieve_trajectory_batch()
      transformed_trajectory_batch = self._transformer.transform_batch_arrays_to_batch_tensors(*trajectory_batch)
      loss_arguments = self._compute_loss_arguments(*transformed_trajectory_batch)
      self._optimize_networks(*loss_arguments)
      self._trajectory_memory.delete_trajectory_batch()


# AdvantageActorCriticAgent: Agent that uses advantage actor critic (A2C) with only one actor-learner thread.
class AdvantageActorCriticAgent:

  def __init__(self, observation_space_size, action_space_size):
    self._hyperparameters = hyperparameters.Hyperparameters()
    self._trajectory_memory = memories.TrajectoryMemory()
    self._transformer = transformer.Transformer(self._hyperparameters.device)
    self._policy_network = neural_networks.PolicyNetwork(observation_space_size, action_space_size,
                                                         self._hyperparameters.learning_rate,
                                                         self._hyperparameters.device)
    self._state_value_network = neural_networks.StateValueNetwork(observation_space_size,
                                                                  self._hyperparameters.learning_rate,
                                                                  self._hyperparameters.device)
    self._step_counter = 0

  def _compute_action_probabilities(self, transformed_observation):
    self._policy_network.eval()
    with torch.no_grad():
      # Reshape the transformed observation to a batch of one observation before providing it to the network.
      transformed_observation_batch = transformed_observation.unsqueeze(0)
      action_probabilities = self._policy_network(transformed_observation_batch).squeeze(0)
    self._policy_network.train()

    return action_probabilities

  def _compute_discounted_return_batch(self, transformed_reward_batch, next_state_value):
    transformed_reward_batch_size = transformed_reward_batch.size(0)
    discounted_return_batch = torch.empty(transformed_reward_batch_size)
    discounted_return = next_state_value
    for i in torch.flip(torch.arange(transformed_reward_batch_size), dims=(0,)):
      discounted_return = transformed_reward_batch[i] + self._hyperparameters.gamma * discounted_return
      discounted_return_batch[i] = discounted_return

    return discounted_return_batch

  def _compute_loss_arguments(self, transformed_observation_batch, transformed_action_batch, transformed_reward_batch,
                              transformed_next_observation, done):
    # Compute the predictions for both networks.
    state_probability_batch = self._policy_network(transformed_observation_batch)
    state_action_probability_batch = state_probability_batch.gather(1, transformed_action_batch.unsqueeze(1)).squeeze(1)
    state_action_log_probability_batch = torch.log(state_action_probability_batch)
    state_entropy_batch = -torch.sum(state_probability_batch * torch.log(state_probability_batch), dim=1)
    state_value_batch = self._state_value_network(transformed_observation_batch).squeeze(1)

    # Compute the discounted returns and advantage values.
    self._state_value_network.eval()
    with torch.no_grad():
      if done:
        next_state_value = 0
      else:
        # Reshape the transformed next observation to a batch of one observation before providing it to the network.
        transformed_next_observation_batch = transformed_next_observation.unsqueeze(0)
        next_state_value = self._state_value_network(transformed_next_observation_batch).squeeze(0)
      discounted_return_batch = self._compute_discounted_return_batch(transformed_reward_batch, next_state_value)
      baseline_state_value_batch = self._state_value_network(transformed_observation_batch).squeeze(1)
      advantage_value_batch = discounted_return_batch - baseline_state_value_batch
    self._state_value_network.train()

    return (state_action_log_probability_batch, state_value_batch, state_entropy_batch, advantage_value_batch,
            discounted_return_batch)

  def _optimize_networks(self, state_action_log_probability_batch, state_value_batch, state_entropy_batch,
                         advantage_value_batch, discounted_return_batch):
    # Optimize the policy network.
    entropy_regularization = self._hyperparameters.entropy_beta * state_entropy_batch
    policy_loss = -torch.sum(state_action_log_probability_batch * advantage_value_batch + entropy_regularization)
    self._policy_network.optimizer.zero_grad()
    policy_loss.backward()
    self._policy_network.optimizer.step()

    # Optimize the state value network.
    state_value_loss = F.mse_loss(state_value_batch, discounted_return_batch)
    self._state_value_network.optimizer.zero_grad()
    state_value_loss.backward()
    self._state_value_network.optimizer.step()

  def select_action(self, observation):
    # Use the action probabilities of the current policy for action selection.
    transformed_observation = self._transformer.transform_array_to_tensor(observation, torch_dtype=torch.float32)
    action_probabilities = self._compute_action_probabilities(transformed_observation)
    action_probability_distribution = torch.distributions.Categorical(action_probabilities)
    action = action_probability_distribution.sample().item()

    return action

  def step(self, observation, action, reward, next_observation, done):
    self._step_counter += 1

    self._trajectory_memory.update_trajectory_batch(observation, action, reward)

    if done or (self._step_counter % self._hyperparameters.network_optimization_frequency == 0):
      trajectory_batch = self._trajectory_memory.retrieve_trajectory_batch()
      transformed_trajectory_batch = self._transformer.transform_batch_arrays_to_batch_tensors(*trajectory_batch)
      transformed_next_observation = self._transformer.transform_array_to_tensor(next_observation,
                                                                                 torch_dtype=torch.float32)
      loss_arguments = self._compute_loss_arguments(*transformed_trajectory_batch, transformed_next_observation, done)
      self._optimize_networks(*loss_arguments)
      self._trajectory_memory.delete_trajectory_batch()
