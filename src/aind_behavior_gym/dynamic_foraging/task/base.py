"""A general gymnasium environment for dynamic foraging tasks in AIND.

Adapted from Han's code for the project in Neuromatch Academy: Deep Learning
https://github.com/hanhou/meta_rl/blob/bd9b5b1d6eb93d217563ff37608aaa2f572c08e6/han/environment/dynamic_bandit_env.py

See also Po-Chen Kuo's implementation:
https://github.com/pckuo/meta_rl/blob/main/environments/bandit/bandit.py
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aind_behavior_gym.task.base import DynamicBanditTask

L = 0
R = 1
IGNORE = 2


class DynamicForagingTaskBase(gym.Env):
    """
    A general gymnasium environment for dynamic bandit task

    Adapted from https://github.com/thinkjrs/gym-bandit-environments/blob/master/gym_bandits/bandit.py  # noqa E501
    """

    def __init__(
        self,
        num_arms: int = 2,  # Number of arms in the bandit
        allow_ignore: bool = False,  # Allow the agent to ignore the task
        num_trials: int = 1000,  # Number of trials in the session
    ):
        """Init"""
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

    def reset(self, seed=None, options={}):
        """
        The reset method will be called to initiate a new episode.
        You may assume that the `step` method will not be called before `reset` has been called.
        Moreover, `reset` should be called whenever a done signal has been issued.
        This should *NOT* automatically reset the task! Resetting the task is
        handled in the wrapper.
        """
        # Seed the random number generator of the env
        self.rng = np.random.default_rng(seed)

        # Some mandatory initialization for any dynamic foraging task
        self.trial = -1
        self.trial_p_reward = []
        self.actions = []
        self.rewards = []
        
        self.generate_next_trial()  # Generate next p_reward

        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        Execute one step in the environment.
        Should return: (observation, reward, terminated, truncated, info)
        If terminated or truncated is true, the user needs to call reset().
        """
        # Action should be type integer in [0, k_bandits-1]
        assert self.action_space.contains(action)
        self.actions.append(action)

        # Generate reward
        reward = self.generate_reward(action)
        self.rewards.append(reward)
        
        # Decide termination before trial += 1
        terminated = bool((self.trial == self.num_trials - 1))  # self.trial starts from 0

        # State transition if not terminated (trial += 1 here)
        if not terminated:
            self.generate_next_trial()

        return self._get_obs(), reward, terminated, False, self._get_info()
    
    def generate_reward(self, action):
        """Compute reward, could be overridden by subclasses for more complex reward structures"""
        # TODO: add baiting here
        reward = 0
        ignored = self.allow_ignore and action == self.action_space.n - 1

        if not ignored and self.rng.uniform(0, 1) < self.trial_p_reward[-1][action]:
            reward = 1
        return reward
    
    def generate_next_trial(self):
        """Generate a new trial and increment the trial number

        # Following lines are mandatory in the overridden method
        self.trial_p_reward.append(...)
        self.trial += 1
        """
        raise NotImplementedError("generate_next_trial() should be overridden by subclasses")
    
    def get_choice_history(self):
        return np.array(self.actions)
    
    def get_reward_history(self):
        return np.array(self.rewards)
    
    def get_p_reward(self):
        return np.array(self.trial_p_reward).T

    def _get_obs(self):
        """Return the observation"""
        return {"trial": self.trial}

    def _get_info(self):
        """
        Info about the environment that the agents is not supposed to know.
        For instance, info can reveal the index of the optimal arm,
        or the value of prior parameter.
        Can be useful to evaluate the agent's perfomance
        """
        return {
            "trial": self.trial,
            "task_object": self,
        }
