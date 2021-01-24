# Deep Reinforcement Learning Agents Simulation Code

This repository contains code that can be used to run simulations with various deep reinforcement learning agents that interact with OpenAI Gym environments.

## Installation

It is recommended to install conda and then create an environment for the simulation software using the ```environment.yaml``` file. A suggestion on how to install the simulation software and activate the environment is provided below.

```bash
git clone https://github.com/anpenta/deep-reinforcement-learning-agents-simulation-code.git
cd deep-reinforcement-learning-agents-simulation-code
conda env create -f environment.yaml
conda activate deep-reinforcement-learning-agents-simulation-code
```

## Running the simulations

To run the simulations you can provide commands through the terminal using the ```simulate``` module. An example is given below.

```bash
python3 simulate.py training_episodes prioritized-deep-q-learning cart-pole 5000 22 1000
```
This will run the ```simulate_training_episodes``` function with an agent that uses prioritized deep Q-learning and the cart-pole environment. The agent will interact with the environment for 5000 episodes, the random seed will be set to 22, and a visual test episode will run every 1000 training episodes. An example of how to see the arguments for each simulation function is provided below.

```bash
python3 simulate.py training_episodes --help
```

## Results

As an example, below are some experimental results with the cart-pole environment. The dark lines are averages over ten experiments with ten different random seeds, and the shaded areas represent the standard deviations.

<p float="left">
<img src=./experimental-results/cart-pole/deep-q-learning-total-reward.png height="320" width="420">
<img src=./experimental-results/cart-pole/double-deep-q-learning-total-reward.png height="320" width="420">
<img src=./experimental-results/cart-pole/prioritized-deep-q-learning-total-reward.png height="320" width="420">
<img src=./experimental-results/cart-pole/dueling-deep-q-learning-total-reward.png height="320" width="420">
<img src=./experimental-results/cart-pole/zero-baseline-vanilla-policy-gradient-total-reward.png height="320" width="420">
<img src=./experimental-results/cart-pole/state-value-baseline-vanilla-policy-gradient-total-reward.png height="320" width="420">
<img src=./experimental-results/cart-pole/advantage-actor-critic-total-reward.png height="320" width="420">
</p>

## Sources
* Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.
* Mnih, Volodymyr, et al. "Asynchronous methods for deep reinforcement learning." International conference on machine learning. 2016.
* Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).
* Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning with double q-learning." Thirtieth AAAI conference on artificial intelligence. 2016.
* Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." arXiv preprint arXiv:1511.06581 (2015).
* Williams, Ronald J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
