"""Generic agent class for dynamic foraging
"""

import numpy as np

from aind_behavior_gym.dynamic_foraging.task import DynamicForagingTaskBase


class DynamicForagingAgentBase:
    """Generic agent class for dynamic foraging"""

    def __init__(
        self,
        seed=None,
    ):
        """Init the agent

        If a task is provided, the agent will be initialized with that task.
        Otherwise, the user must call `set_task(task)` before the agent can perform any actions.
        """
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Resets the agent's internal state. Override this if your agent has an internal state."""
        pass

    def perform(
        self,
        task: DynamicForagingTaskBase,
    ):
        """Perform one session (eposide) of the dynamic foraging task while learning."""
        self.task = task
        self.n_actions = task.action_space.n

        # --- Main task loop ---
        observation, info = self.task.reset()  # Get the initial observation
        done = False
        while not done:
            action = self.act(observation)
            next_observation, reward, done, truncated, info = task.step(action)
            self.learn(observation, action, reward, next_observation, done)
            observation = next_observation

    def act(self, observation):
        """
        Chooses an action based on the current observation.

        Args:
            observation: The current observation from the environment.

        Returns:
            action: The action chosen by the agent.
        """
        raise NotImplementedError("The 'act' method should be overridden by subclasses.")

    def learn(self, observation, action, reward, next_observation, done):
        """
        Updates the agent's knowledge or policy based on the last action and its outcome.

        This is the core method that should be implemented by all non-trivial agents.
        It could be Q-learning, policy gradients, neural networks, etc.

        Args:
            observation: The observation before the action was taken.
            action: The action taken by the agent.
            reward: The reward received after taking the action.
            next_observation: The next observation after the action.
            done: Whether the episode has ended.
        """
        raise NotImplementedError("The 'learn' method should be overridden by subclasses.")

    def fit(self, data):
        """
        Fit the parameters of the agent to data.

        Args:
            data: Either from animal data (model fitting) or simulated data (model recovery)
        """
        raise NotImplementedError("The 'fit' method should be overridden in order to fit data.")

    def save(self, filepath):
        """
        Saves the agent's current state or learned parameters to a file.

        Args:
            filepath (str): The path to the file where the agent's state will be saved.
        """
        raise NotImplementedError("The 'save' method should be overridden by subclasses.")

    def load(self, filepath):
        """
        Loads the agent's state or learned parameters from a file.

        Args:
            filepath (str): The path to the file from which the agent's state will be loaded.
        """
        raise NotImplementedError("The 'load' method should be overridden by subclasses.")
