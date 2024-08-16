"""Test the RandomWalkTask by itself
"""

import unittest

import numpy as np

from aind_behavior_gym.dynamic_foraging.agent.random_agent import RandomAgent
from aind_behavior_gym.dynamic_foraging.task.random_walk_task import RandomWalkTask


class TestRandomWalkTask(unittest.TestCase):
    """Test the RandomWalkTask by itself"""

    def setUp(self):
        """Set up the environment and task"""
        self.task = RandomWalkTask(
            p_min=[0.1, 0.1],  # The lower bound of p_L and p_R
            p_max=[0.9, 0.9],  # The upper bound
            sigma=[0.1, 0.1],  # The mean of each step of the random walk
            mean=[0, 0],  # The mean of each step of the random walk
            num_trials=1000,
            allow_ignore=False,
            seed=42,
        )
        self.agent = RandomAgent(seed=42)
        self.agent.add_task(self.task)

    def test_random_walk_task(self):
        """Test the reward schedule"""
        # Agent performs the task
        self.agent.perform()

        # Call plot function and check it runs without error
        fig = self.task.plot_reward_schedule()
        fig.savefig("tests/results/test_random_walk_task.png")
        self.assertIsNotNone(fig)  # Ensure the figure is created

        np.testing.assert_array_almost_equal(
            self.task.trial_p_reward[:10, :],
            np.array(
                [
                    [0.71916484, 0.45110275],
                    [0.81322131, 0.25599923],
                    [0.82600535, 0.22437497],
                    [0.74070096, 0.31231477],
                    [0.74730403, 0.42503889],
                    [0.66137478, 0.46191397],
                    [0.74921981, 0.45692138],
                    [0.68112686, 0.57917551],
                    [0.63829407, 0.54396216],
                    [0.67483848, 0.58523542],
                ]
            ),
        )
        np.testing.assert_array_equal(
            self.task.get_reward_history()[-10:],
            np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
