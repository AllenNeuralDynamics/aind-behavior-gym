"""Test the CoupledBlockTask with a random agent"""

import unittest

import numpy as np
from aind_dynamic_foraging_basic_analysis import plot_foraging_session

from aind_behavior_gym.dynamic_foraging.agent.random_agent import RandomAgent
from aind_behavior_gym.dynamic_foraging.task.coupled_block_task import CoupledBlockTask


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

        # Make sure block transitions are correct
        self.assertEqual(
            self.task.block_starts[:-1],
            np.where(np.diff(np.concatenate([np.array([-1]), self.task.get_p_reward()[0]])))[
                0
            ].tolist(),
        )
        self.assertEqual(
            self.task.block_starts,
            [
                0,
                80,
                122,
                181,
                222,
                270,
                320,
                386,
                466,
                529,
                575,
                630,
                689,
                732,
                812,
                857,
                909,
                989,
                1034,
            ],
        )
        np.testing.assert_array_equal(
            self.task.get_choice_history()[-10:],
            np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
        )
        np.testing.assert_array_equal(
            self.task.get_reward_history()[-10:],
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        )
        np.testing.assert_array_equal(
            self.task.get_p_reward()[:, -10:],
            np.array(
                [
                    [
                        0.3375,
                        0.3375,
                        0.3375,
                        0.3375,
                        0.3375,
                        0.3375,
                        0.3375,
                        0.3375,
                        0.3375,
                        0.3375,
                    ],
                    [
                        0.1125,
                        0.1125,
                        0.1125,
                        0.1125,
                        0.1125,
                        0.1125,
                        0.1125,
                        0.1125,
                        0.1125,
                        0.1125,
                    ],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
