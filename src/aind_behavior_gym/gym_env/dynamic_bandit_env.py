"""A general gymnasium environment for dynamic foraging tasks in AIND.

First coded by Han for the project in Neuromatch Academy: Deep Learning
https://github.com/hanhou/meta_rl/blob/bd9b5b1d6eb93d217563ff37608aaa2f572c08e6/han/environment/dynamic_bandit_env.py
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..dynamic_foraging_tasks.base import DynamicBanditTask

L = 0
R = 1
IGNORE = 2


class DynamicBanditEnv(gym.Env):
    """
    A general gymnasium environment for dynamic bandit task

    - To use the environment, you need to define a task object that determines
      the dynamics of reward contingencies.

        For example:

        from dynamic_foraging_tasks.coupled_block_task import CoupledBlockTask

        task = CoupledBlockTask(block_min=40, block_max=80, block_beta=20)
        env = DynamicBanditEnv(task, num_trials=1000)
        agent = # define your agent here

        observation, info = env.reset()
        done = False

        while not done:  # Trial loop
            # Choose an action
            action = agent.act(observation)

            # Take the action and observe the next observation and reward
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Move to the next observation
            observation = next_observation

    - Overwrite the _get_obs() method to define the observation space.

    ---
    task:
        A task object (like CoupledBlockTask) that defines the dynamics of reward contingencies
        in the environment.
    num_trials:
        Number of trials in the session

    Adapted from https://github.com/thinkjrs/gym-bandit-environments/blob/master/gym_bandits/bandit.py  # noqa E501
    """

    def __init__(
        self,
        task: DynamicBanditTask,  # Receive an instance of the task object
        num_arms: int = 2,  # Number of arms in the bandit
        allow_ignore: bool = False,  # Allow the agent to ignore the task
        num_trials: int = 1000,  # Number of trials in the session
    ):
        """Init"""
        self.task = task
        self.num_trials = num_trials
        self.allow_ignore = allow_ignore

        # State space
        # - Time (trial number) is the only observable state to the agent
        self.observation_space = spaces.Dict(
            {
                "trial": spaces.Box(low=0, high=self.num_trials, dtype=np.int64),
            }
        )

        # Action space
        num_actions = num_arms + int(allow_ignore)  # Add the last action as ignore if allowed
        self.action_space = spaces.Discrete(num_actions)

    def _get_obs(self):
        """Return the observation"""
        return {"trial": self.task.trial}

    def _get_info(self):
        """
        Info about the environment that the agents is not supposed to know.
        For instance, info can reveal the index of the optimal arm,
        or the value of prior parameter.
        Can be useful to evaluate the agent's perfomance
        """
        return {
            "trial": self.trial,
            "task_object": self.task,
        }

    def reset(self, seed=None, options={}):
        """
        The reset method will be called to initiate a new episode.
        You may assume that the `step` method will not be called before `reset` has been called.
        Moreover, `reset` should be called whenever a done signal has been issued.
        This should *NOT* automatically reset the task! Resetting the task is
        handled in the wrapper.
        """
        # Seed the random number generator and pass it to the task as well
        self.rng = np.random.default_rng(seed)

        # Reset the task
        self.task.reset(seed=seed)  # Using the same seed if provided
        self.trial = self.task.trial

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Execute one step in the environment.
        Should return: (observation, reward, terminated, truncated, info)
        If terminated or truncated is true, the user needs to call reset().
        """
        # Action should be type integer in [0, k_bandits-1]
        assert self.action_space.contains(action)

        # Compute reward
        # TODO: add baiting here
        reward = 0
        ignored = self.allow_ignore and action == self.action_space.n - 1

        if not ignored and self.rng.uniform(0, 1) < self.task.trial_p_reward[-1][action]:
            reward = 1

        # Decide termination before trial += 1
        terminated = bool((self.trial == self.num_trials - 1))  # self.trial starts from 0

        # State transition if not terminated (trial += 1 here)
        if not terminated:
            self.task.add_action(
                action
            )  # Task state transition may depend on the action (like in Uncoupled task)
            self.task.next_trial()
            self.trial = self.task.trial

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info


class DynamicBanditEnvHistoryAsState(DynamicBanditEnv):
    """Use history as state

    self._get_obs() returns a tuple containing the recent history
    [action(t), reward(t), action(t-1), reward(t-1), ..., action(t - history_length + 1), reward(t - history_length + 1)]  # noqa E501

    This will be the input to the DQN as "state"
    """

    def __init__(self, task, num_trials=1000, history_length=50):
        """Override the init method to add history_length"""
        super().__init__(task, num_trials)
        self.history_length = history_length
        self.history = []

        # Define the observation space based on the history length and the
        # original observation space
        self.observation_space = spaces.Box(
            low=0, high=self.num_trials, shape=(history_length * 2,), dtype=np.float32
        )

    def _get_obs(self):
        """Flatten the history into a single vector and return it as the observation"""
        if len(self.history) < self.history_length:
            padding = [(0, 0)] * (self.history_length - len(self.history))
            history = self.history[::-1] + padding
        else:
            history = self.history[-self.history_length :][::-1]  # noqa E203
        return np.array(history, dtype=np.float32).flatten()

    def reset(self, seed=None, options={}):
        """Override the reset method to initialize the history"""
        observation, info = super().reset(seed=seed, options=options)
        self.history = [(0, 0)] * self.history_length  # Initialize with zeros
        return self._get_obs(), info

    def step(self, action):
        """Override the step method to update the history"""
        observation, reward, terminated, truncated, info = super().step(action)
        self.history.append((action, reward))
        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode="human"):
        """Trivial render method"""
        return super().render(mode)

    def close(self):
        """Trivial close method"""
        return super().close()
