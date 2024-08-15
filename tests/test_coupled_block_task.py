"""Test the CoupledBlockTask with a random agent"""

import unittest
import numpy as np

from aind_behavior_gym.dynamic_foraging.task.coupled_block_task import CoupledBlockTask
from aind_behavior_gym.dynamic_foraging.task.base import L, R
from aind_behavior_gym.dynamic_foraging.agent.random_agent import RandomAgent

from aind_dynamic_foraging_basic_analysis import plot_foraging_session


class TestCoupledBlockTask(unittest.TestCase):
    """Test the CoupledBlockTask with a random agent"""

    def setUp(self):
        """Set up the environment and task"""
        self.task = CoupledBlockTask(allow_ignore=False, seed=42)
        self.agent = RandomAgent(task=self.task, seed=42)
        
    def test_coupled_block_task(self):
        """Test the CoupledBlockTask with a random agent"""
        # Agent performs the task
        self.agent.perform()
        
        # Call plot function and check it runs without error
        fig, _ = plot_foraging_session(
            choice_history=self.task.get_choice_history(),
            reward_history=self.task.get_reward_history(),
            p_reward=self.task.get_p_reward(),
        )
        fig.savefig("tests/results/test_coupled_block_task.png")
        self.assertIsNotNone(fig)  # Ensure the figure is created

        self.assertEqual(
            self.task.block_starts,
            [
                0,
                80,
                122,
                167,
                213,
                270,
                311,
                363,
                443,
                518,
                558,
                638,
                691,
                740,
                781,
                821,
                873,
                922,
                974,
                1018,
            ],
        )
        np.testing.assert_array_equal(
            self.task.get_choice_history()[-10:], np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 1])
        )
        np.testing.assert_array_equal(
            self.task.get_reward_history()[-10:], np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        )
        np.testing.assert_array_equal(
            self.task.get_p_reward()[:, -10:],
            np.array(
                [
                    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                    [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
