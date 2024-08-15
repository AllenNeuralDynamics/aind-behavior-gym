"""A Random agent
"""

from aind_behavior_gym.dynamic_foraging.agent.base import AgentBase

class RandomAgent(AgentBase):
    """A Random agent
    """

    def act(self, observation):
        """Simply random choose in the action space"""
        return self.rng.choice(self.n_actions)

    def learn(self, observation, action, reward, next_observation, done):
        """No learning for a random agent"""
        pass