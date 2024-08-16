"""Random walk task for the dynamic bandit environment.
"""

import matplotlib.pyplot as plt
import numpy as np

from aind_behavior_gym.dynamic_foraging.task import DynamicForagingTaskBase, L, R


class RandomWalkTask(DynamicForagingTaskBase):
    """
    Generate reward schedule with random walk

    (see Miller et al. 2021, https://www.biorxiv.org/content/10.1101/461129v3.full.pdf)
    """

    def __init__(
        self,
        p_min=[0, 0],  # The lower bound of p_L and p_R
        p_max=[1, 1],  # The upper bound
        sigma=[0.15, 0.15],  # The mean of each step of the random walk
        mean=[0, 0],  # The mean of each step of the random walk
        **kwargs,
    ) -> None:
        """Init"""
        super().__init__(**kwargs)

        if not isinstance(sigma, list):
            sigma = [sigma, sigma]  # Backward compatibility

        if not isinstance(p_min, list):
            p_min = [p_min, p_min]  # Backward compatibility

        if not isinstance(p_max, list):
            p_max = [p_max, p_max]  # Backward compatibility

        self.p_min, self.p_max, self.sigma, self.mean = p_min, p_max, sigma, mean

    def reset(self, seed=None):
        """Reset the task, remember to call the base class reset at the end."""
        self.hold_this_block = False

        return super().reset()

    def generate_new_trial(self):
        """Generate a new trial. Overwrite the base class method."""
        # Note that self.trial already increased by 1 here
        self.trial_p_reward[self.trial, :] = [self._generate_next_p(side) for side in [L, R]]

    def _generate_next_p(self, side):
        """Generate the p_side for the next trial."""
        if self.trial == 0:
            # Start with uniform distribution
            return self.rng.uniform(self.p_min[side], self.p_max[side])
        if self.hold_this_block:
            return self.trial_p_reward[self.trial - 1, side]

        # Else, take a random walk
        p = self.rng.normal(
            self.trial_p_reward[self.trial - 1, side] + self.mean[side], self.sigma[side]
        )
        p = min(self.p_max[side], max(self.p_min[side], p))  # Absorb at the boundary
        return p

    def plot_reward_schedule(self):
        """Plot the reward schedule and compute the auto-correlation."""
        trial_p_reward = np.array(self.trial_p_reward)

        fig, ax = plt.subplots(
            2, 2, figsize=[15, 7], sharex="col", gridspec_kw=dict(width_ratios=[4, 1], wspace=0.1)
        )

        for s, col in zip([L, R], ["r", "b"]):
            ax[0, 0].plot(trial_p_reward[:, s], col, marker=".", alpha=0.5, lw=2)
            ax[0, 1].plot(auto_corr(trial_p_reward[:, s]), col)

        ax[1, 0].plot(trial_p_reward[:, L] + trial_p_reward[:, R], label="sum")
        ax[1, 0].plot(
            trial_p_reward[:, R] / (trial_p_reward[:, L] + trial_p_reward[:, R]),
            label="R/(L+R)",
        )
        ax[1, 0].legend()

        ax[0, 1].set(title="auto correlation", xlim=[0, 100])
        ax[0, 1].axhline(y=0, c="k", ls="--")

        return fig


def auto_corr(data):
    """Util function to compute the auto-correlation of the data."""
    mean = np.mean(data)
    # Variance
    var = np.var(data)
    # Normalized data
    ndata = data - mean
    acorr = np.correlate(ndata, ndata, "full")[len(ndata) - 1 :]  # noqa E203
    acorr = acorr / var / len(ndata)
    return acorr
