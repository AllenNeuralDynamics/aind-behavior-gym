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

    def test_random_walk_task(self):
        """Test the reward schedule"""
        # Agent performs the task
        self.agent.perform(task=self.task)

        # Call plot function and check it runs without error
        fig = self.task.plot_reward_schedule()
        fig.savefig("tests/results/test_random_walk_task.png")
        self.assertIsNotNone(fig)  # Ensure the figure is created

        # Check reward assignment
        np.testing.assert_array_equal(
            np.logical_or(
                np.logical_and(
                    self.task.reward_baiting, self.task.reward_assigned_after_action[:-1]
                ),
                self.task.random_numbers[1:] < self.task.trial_p_reward[1:],
            ),
            self.task.reward_assigned_before_action[1:],
        )

        np.testing.assert_array_almost_equal(
            self.task.trial_p_reward[:10, :],
            np.array(
                [
                    [0.71916484, 0.45110275],
                    [0.52406132, 0.3208848],
                    [0.5223812, 0.23558041],
                    [0.52898427, 0.34830453],
                    [0.56585935, 0.25241627],
                    [0.54737312, 0.18432331],
                    [0.50454033, 0.14910996],
                    [0.54581359, 0.19219206],
                    [0.49458932, 0.11081479],
                    [0.48319458, 0.1],
                ]
            ),
        )
        np.testing.assert_array_equal(
            self.task.get_reward_history()[-10:],
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
