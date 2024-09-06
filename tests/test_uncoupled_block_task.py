"""Test the dynamic bandit environment with a random agent performing an uncoupled block task"""

import unittest

import numpy as np

from aind_behavior_gym.dynamic_foraging.agent.random_agent import RandomAgentBiasedIgnore
from aind_behavior_gym.dynamic_foraging.task.uncoupled_block_task import (
    IGNORE,
    L,
    R,
    UncoupledBlockTask,
)


class TestUncoupledTask(unittest.TestCase):
    """Test the dynamic bandit environment with a random agent
    performing an uncoupled block task
    """

    def setUp(self):
        """Set up the environment and task"""

        self.task = UncoupledBlockTask(
            rwd_prob_array=[0.1, 0.5, 0.9],
            block_min=20,
            block_max=35,
            persev_add=True,
            perseverative_limit=4,
            max_block_tally=4,
            allow_ignore=True,
            num_trials=1000,
            seed=42,
        )
        self.agent = RandomAgentBiasedIgnore(seed=42)

    def test_uncoupled_block_task(self):
        """Test the UncoupledBlockTask with a random agent"""
        # --- Agent performs the task ---
        self.agent.perform(task=self.task)

        # --- Assertions ---
        # Call plot function
        fig, _ = self.task.plot_reward_schedule()
        fig.savefig("tests/results/test_uncoupled_block_task.png")
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

        # Assertions to verify the behavior of block ends
        self.assertEqual(
            self.task.block_ends[L],
            [
                21,
                47,
                104,
                126,
                144,
                173,
                201,
                235,
                262,
                285,
                319,
                340,
                364,
                396,
                424,
                454,
                486,
                516,
                547,
                589,
                609,
                688,
                721,
                753,
                786,
                844,
                864,
                868,
                896,
                967,
                987,
                990,
                1011,
            ],
        )

        self.assertEqual(
            self.task.block_ends[R],
            [
                17,
                44,
                94,
                144,
                164,
                192,
                201,
                254,
                282,
                302,
                323,
                340,
                380,
                396,
                432,
                454,
                504,
                538,
                589,
                601,
                695,
                721,
                752,
                777,
                833,
                868,
                880,
                947,
                990,
                1011,
            ],
        )

        # Verify rewards
        self.assertEqual(
            self.task.reward[-25:].tolist(),
            [
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
            ],
        )
        self.assertEqual(self.task.reward[self.task.action == IGNORE].sum(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
