"""Test the CoupledBlockTask with a random agent"""

import unittest
import numpy as np

from aind_behavior_gym.dynamic_foraging_tasks.coupled_block_task import CoupledBlockTask
from aind_behavior_gym.gym_env.dynamic_bandit_env import DynamicBanditEnv
from aind_behavior_gym.gym_env.dynamic_bandit_env import L, R


class TestCoupledBlockTask(unittest.TestCase):
    """Test the CoupledBlockTask with a random agent"""

    def setUp(self):
        """Set up the environment and task"""
        self.task = CoupledBlockTask(block_min=40, block_max=80, block_beta=20)
        self.env = DynamicBanditEnv(self.task, num_trials=1000)
        self.rng = np.random.default_rng(seed=42)  # Random number generator

    def test_coupled_block_task(self):
        """Test the CoupledBlockTask with a random agent"""
        observation, info = self.env.reset(seed=42)
        done = False
        actions = []
        rewards = []

        while not done:  # Trial loop
            # Randomly choose between L and R
            action = self.rng.choice([L, R])

            # Take the action and observe the next observation and reward
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            actions.append(action)
            rewards.append(reward)

        # Assertions to verify the length of actions and rewards matches the number of trials
        self.assertEqual(
            self.task.block_starts,
            [
                0,
                80,
                125,
                166,
                234,
                314,
                374,
                415,
                489,
                536,
                579,
                620,
                678,
                726,
                770,
                846,
                899,
                947,
                988,
                1031,
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
