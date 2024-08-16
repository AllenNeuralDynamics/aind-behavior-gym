"""Couple block task for dynamic bandit environment
This is very close to the task used in mice training.

First coded by Han for the project in Neuromatch Academy: Deep Learning
https://github.com/hanhou/meta_rl/blob/bd9b5b1d6eb93d217563ff37608aaa2f572c08e6/han/environment/dynamic_bandit_env.py
"""

from typing import List

import numpy as np

from aind_behavior_gym.dynamic_foraging.task import DynamicForagingTaskBase, L, R


class CoupledBlockTask(DynamicForagingTaskBase):
    """Coupled block task for dynamic foraging

    This default setting roughly matches what has been used in this paper:
    https://www.sciencedirect.com/science/article/pii/S089662731930529X
    """

    def __init__(
        self,
        block_min: int = 40,  # Min block length
        block_max: int = 80,  # Max block length
        block_beta: int = 20,  # Time constant of exponential distribution (the larger the flatter)
        p_reward_pairs: List[List[float]] = None,  # List of reward probability pairs
        **kwargs,
    ):
        """Init"""
        super().__init__(**kwargs)

        if p_reward_pairs is None:
            p_reward_pairs = [
                [0.225, 0.225],  # 1:1
                [0.45 / 4 * 1, 0.45 / 4 * 3],  # 1:3
                [0.45 / 7 * 1, 0.45 / 7 * 6],  # 1:6
                [0.05, 0.40],  # 1:8
            ]

        self.block_min = block_min
        self.block_max = block_max
        self.block_beta = block_beta
        self.p_reward_pairs = [sorted(ps) for ps in p_reward_pairs]  # Always sort the input ps

    def reset(self):
        """Reset the task"""

        # Add more initialization specific to this task
        self.block_starts = [0]  # Start of each block. The first block always starts at trial 0
        self.block_lens = []  # Lengths of each block
        self.block_p_reward = []  # Rwd prob of each block

        # Call the base class reset at the end
        return super().reset()

    def generate_new_trial(self):
        """Override the base class method to generate the next trial for coupled block task."""
        # Start a new block if necessary
        if self.trial == self.block_starts[-1]:
            self._next_block()

        # Append the current block's reward probability
        # Note that self.trial already increased by 1 here
        self.trial_p_reward[self.trial, :] = self.block_p_reward[-1]
        return self.trial_p_reward[-1, :]

    def _next_block(self):
        """
        Generate the next block
        """
        # Generate the block length
        self.block_lens.append(
            int(
                generate_trunc_exp(
                    self.block_min,
                    self.block_max,
                    self.block_beta,
                    rng=self.rng,
                )[0]
            )
        )
        self.block_starts.append(self.block_starts[-1] + self.block_lens[-1])

        # Generate the reward probability
        self.block_p_reward.append(self._generate_block_p_reward())
        return

    def _generate_block_p_reward(self):
        """
        Generate the reward probability for the next block.
        """
        # If it is the first block, randomly choose a pair and the side
        if len(self.block_p_reward) == 0:
            p_reward = self.rng.choice(self.p_reward_pairs)
            p_reward = self._flip_side(p_reward, None)
            return p_reward

        # Else, generate a new p_reward based on the current p_reward
        # 1. if current p_L == p_R, randomly choose a p_reward_pair (excluding p_L == p_R)
        #    and make sure the new block is flipped compare
        #    to the one before the equal-probability block
        # 2. else, randomly choose a p_reward_pair and always flip the side
        if self.block_p_reward[-1][L] == self.block_p_reward[-1][R]:
            # Cannot be p_L == p_R again
            valid_pairs = [p for p in self.p_reward_pairs if p[L] != p[R]]
            # Randomly choose from the valid pairs
            p_reward = self.rng.choice(valid_pairs)
            # If there is a block before the equal-probability block, flip relative to it
            # otherwise, randomly choose
            p_reward = self._flip_side(
                p_reward, self.block_p_reward[-2] if len(self.block_p_reward) > 1 else None
            )
        else:
            # Randomly choose from any pairs
            p_reward = self.rng.choice(self.p_reward_pairs)
            # Make sure the side is flipped
            p_reward = self._flip_side(p_reward, self.block_p_reward[-1])

        return p_reward

    def _flip_side(self, p_reward_new, p_reward_old=None):
        """
        Make sure the new block is flipped compare to the one before the equal-probability block.
        If old is None, flip it with a 0.5 probability.
        """
        should_flip = p_reward_old is None and self.rng.random() < 0.5
        if p_reward_old is not None:
            should_flip = (p_reward_new[L] < p_reward_new[R]) == (p_reward_old[L] < p_reward_old[R])

        return p_reward_new[::-1] if should_flip else p_reward_new


def generate_trunc_exp(lower, upper, beta, n=1, rng=None):
    """
    Generate n samples from a truncated exponential distribution
    """
    if rng is None:
        rng = np.random.default_rng()

    x = lower + rng.exponential(beta, n)
    x[x > upper] = upper
    return x
