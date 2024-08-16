"""A Random agent
"""

from aind_behavior_gym.dynamic_foraging.agent import AgentBase
from aind_behavior_gym.dynamic_foraging.task import L, R, IGNORE

class RandomAgent(AgentBase):
    """A Random agent
    """

    def act(self, observation):
        """Simply random choose in the action space"""
        return self.rng.choice(self.n_actions)

    def learn(self, observation, action, reward, next_observation, done):
        """No learning for a random agent"""
        pass
    

class RandomAgentBiasedIgnore(RandomAgent):
    """A biased agent with ignores
    """
    
    def act(self, *args):
        """Random choose from three actions"""
        return [L, R, IGNORE][self.rng.choice([0] * 100 + [1] * 20 + [2] * 1)]
