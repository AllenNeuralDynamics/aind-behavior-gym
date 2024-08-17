"""A general gymnasium environment for dynamic foraging tasks in AIND.

Adapted from Han's code for the project in Neuromatch Academy: Deep Learning
https://github.com/hanhou/meta_rl/blob/bd9b5b1d6eb93d217563ff37608aaa2f572c08e6/han/environment/dynamic_bandit_env.py

See also Po-Chen Kuo's implementation:
https://github.com/pckuo/meta_rl/blob/main/environments/bandit/bandit.py
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

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
        reward_baiting: bool = False,  # Whether the reward is baited
        allow_ignore: bool = False,  # Allow the agent to ignore the task
        num_arms: int = 2,  # Number of arms in the bandit
        num_trials: int = 1000,  # Number of trials in the session
        seed=None,
    ):
        """Init"""
        self.num_trials = num_trials
        self.reward_baiting = reward_baiting
        self.num_arms = num_arms
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

        # Random seed
        self.rng = np.random.default_rng(seed)

    def reset(self, options={}):
        """
        The reset method will be called to initiate a new episode.
        You may assume that the `step` method will not be called before `reset` has been called.
        Moreover, `reset` should be called whenever a done signal has been issued.
        This should *NOT* automatically reset the task! Resetting the task is
        handled in the wrapper.
        """
        # Some mandatory initialization for any dynamic foraging task
        self.trial = 0
        self.trial_p_reward = np.empty((self.num_trials, self.num_arms))
        self.reward_assigned_before_action = np.zeros_like(
            self.trial_p_reward
        )  # Whether the reward exists in a certain trial before action
        self.reward_assigned_after_action = np.zeros_like(
            self.trial_p_reward
        )  # Whether the reward exists in a certain trial after action
        self.random_numbers = np.empty_like(
            self.trial_p_reward
        )  # Cache the generated random numbers

        self.action = np.empty(self.num_trials, dtype=int)
        self.reward = np.empty(self.num_trials)

        self.generate_new_trial()  # Generate a new p_reward for the first trial

        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        Execute one step in the environment.
        Should return: (observation, reward, terminated, truncated, info)
        If terminated or truncated is true, the user needs to call reset().
        """
        # Action should be type integer in [0, num_arms-1] if not allow_ignore else [0, num_arms]
        assert self.action_space.contains(action)
        self.action[self.trial] = action

        # Generate reward
        reward = self.generate_reward(action)
        self.reward[self.trial] = reward

        # Generate observation and info before self.trial += 1
        observation = self._get_obs()
        info = self._get_info()

        # Decide termination before trial += 1
        terminated = bool((self.trial == self.num_trials - 1))  # self.trial starts from 0

        # State transition if not terminated (trial += 1 here)
        if not terminated:
            self.trial += 1  # tick time here
            self.generate_new_trial()

        return observation, reward, terminated, False, info

    def generate_reward(self, action):
        """Compute reward, could be overridden by subclasses for more complex reward structures"""

        # -- Refilling rewards on this trial --
        self.random_numbers[self.trial] = self.rng.uniform(0, 1, size=self.num_arms)
        reward_assigned = (
            self.random_numbers[self.trial] < self.trial_p_reward[self.trial]
        ).astype(float)

        # -- Reward baited from the last trial --
        if self.reward_baiting and self.trial > 0:
            reward_assigned = np.logical_or(
                reward_assigned, self.reward_assigned_after_action[self.trial - 1]
            ).astype(float)

        # Cache the reward assignment
        self.reward_assigned_before_action[self.trial] = reward_assigned
        self.reward_assigned_after_action[self.trial] = reward_assigned

        # -- Reward delivery --
        if action == IGNORE:
            # Note that reward may be still refilled even if the agent ignores the trial
            return 0

        # Clear up the reward_assigned_after_action slot and return the reward
        self.reward_assigned_after_action[self.trial, action] = 0
        return reward_assigned[action]

    def generate_new_trial(self):
        """Generate p_reward for a new trial
        Note that self.trial already increased by 1 here
        """
        raise NotImplementedError("generate_next_trial() should be overridden by subclasses")

    def get_choice_history(self):
        """Return the history of actions in format that is compatible with other library such as
        aind_dynamic_foraging_basic_analysis
        """
        actions = self.action.astype(float)
        actions[actions == IGNORE] = np.nan
        return actions

    def get_reward_history(self):
        """Return the history of rewards in format that is compatible with other library such as
        aind_dynamic_foraging_basic_analysis
        """
        return self.reward

    def get_p_reward(self):
        """Return the reward probabilities for each arm in each trial which is compatible with
        other library such as aind_dynamic_foraging_basic_analysis
        """
        return self.trial_p_reward.T

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
            "task_object": self,  # Return the whole task object for debugging
        }
