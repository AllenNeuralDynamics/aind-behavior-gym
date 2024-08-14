"""Test the RandomWalkTask by itself
"""

import unittest

import numpy as np

from aind_behavior_gym.dynamic_foraging_tasks.random_walk_task import RandomWalkTask
from aind_behavior_gym.gym_env.dynamic_bandit_env import L


class TestRandomWalkTask(unittest.TestCase):
    """Test the RandomWalkTask by itself"""

    def setUp(self):
        """Set up the environment and task"""
        self.total_trial = 1000
        self.reward_schedule = RandomWalkTask(
            p_min=[0.1, 0.1],  # The lower bound of p_L and p_R
            p_max=[0.9, 0.9],  # The upper bound
            sigma=[0.1, 0.1],  # The mean of each step of the random walk
            mean=[0, 0],  # The mean of each step of the random walk
        )
        self.reward_schedule.reset(seed=42)  # Already includes a next_trial()

    def test_reward_schedule(self):
        """Test the reward schedule"""
        while self.reward_schedule.trial < self.total_trial:
            # Replace this with the actual choice
            choice = L  # Irrelevant for random walk

            # Add choice
            self.reward_schedule.add_action(choice)

            # Next trial
            self.reward_schedule.next_trial()

        # Call plot function and check it runs without error
        fig = self.reward_schedule.plot_reward_schedule()
        fig.savefig("tests/results/test_random_walk_task.png")
        self.assertIsNotNone(fig)  # Ensure the figure is created

        np.testing.assert_array_almost_equal(
            np.array(self.reward_schedule.trial_p_reward)[:10, :],
            np.array(
                [
                    [0.71916484, 0.45110275],
                    [0.79420996, 0.54515922],
                    [0.59910644, 0.41494127],
                    [0.61189048, 0.38331701],
                    [0.61021036, 0.29801262],
                    [0.69815016, 0.37579181],
                    [0.70475323, 0.48851594],
                    [0.75150417, 0.40258669],
                    [0.78837924, 0.30669843],
                    [0.87622427, 0.30170584],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
