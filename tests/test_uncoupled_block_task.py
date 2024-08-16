"""Test the dynamic bandit environment with a random agent performing an uncoupled block task"""

import unittest

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
        self.agent.add_task(self.task)

    def test_uncoupled_block_task(self):
        """Test the UncoupledBlockTask with a random agent"""
        # --- Agent performs the task ---
        self.agent.perform()

        # --- Assertions ---
        # Call plot function
        fig = self.task.plot_reward_schedule()
        fig.savefig("tests/results/test_uncoupled_block_task.png")
        self.assertIsNotNone(fig)  # Ensure the figure is created

        # Assertions to verify the behavior of block ends
        self.assertEqual(
            self.task.block_ends[L],
            [
                21,
                52,
                76,
                102,
                183,
                214,
                321,
                349,
                378,
                398,
                418,
                450,
                476,
                502,
                527,
                564,
                601,
                622,
                653,
                676,
                698,
                725,
                751,
                773,
                780,
                813,
                836,
                865,
                900,
                904,
                928,
                969,
                989,
                1007,
            ],
        )

        self.assertEqual(
            self.task.block_ends[R],
            [
                17,
                21,
                62,
                92,
                183,
                191,
                224,
                331,
                366,
                398,
                418,
                458,
                476,
                520,
                527,
                570,
                597,
                618,
                622,
                669,
                693,
                698,
                743,
                773,
                796,
                831,
                851,
                875,
                904,
                918,
                946,
                975,
                989,
                1022,
            ],
        )

        # Verify rewards
        self.assertEqual(
            self.task.rewards[-25:].tolist(),
            [
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
        )
        self.assertEqual(self.task.rewards[self.task.actions == IGNORE].sum(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
