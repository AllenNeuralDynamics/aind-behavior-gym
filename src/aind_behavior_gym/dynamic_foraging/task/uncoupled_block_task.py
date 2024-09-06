"""Uncoupled task for dynamic bandit environment

see /test/test_uncoupled_block_task.py for usage

"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from aind_behavior_gym.dynamic_foraging.task import IGNORE, DynamicForagingTaskBase, L, R

logger = logging.getLogger(__name__)


class UncoupledBlockTask(DynamicForagingTaskBase):
    """
    Generate uncoupled block reward schedule
    (by on-line updating)

    adapted from Cohen lab's Arduino code (with some bug fixes)
    https://github.com/JeremiahYCohenLab/arduinoLibraries/blob/master/libraries/task_operantMatchingDecoupledBait/task_operantMatchingDecoupledBait.cpp  # noqa E501

    See Grossman et al. 2022:

    In the final stage of the task, the reward probabilities assigned to each lick spout were drawn
    pseudorandomly from the set {0.1, 0.5, 0.9} in all the mice from the behavior experiments (n=46),
    all the mice from the DREADDs experiments (n=10), and half of the mice from the electrophysiology
    experiments (n=2). The other half of mice from the electrophysiology experiments (n=2) were run
    on a version of the task with probabilities drawn from the set {0.1, 0.4, 0.7}. The probabilities
    were assigned to each spout individually with block lengths drawn from a uniform distribution
    of 20–35 trials. To stagger the blocks of probability assignment for each spout, the block length
    for one spout in the first block of each session was drawn from a uniform distribution of 6–21
    trials. For each spout, probability assignments could not be repeated across consecutive blocks.
    To maintain task engagement, reward probabilities of 0.1 could not be simultaneously assigned
    to both spouts. If one spout was assigned a reward probability greater than or equal to the reward
    probability of the other spout for 3 consecutive blocks, the probability of that spout was set to
    0.1 to encourage switching behavior and limit the creation of a direction bias. If a mouse
    perseverated on a spout with a reward probability of 0.1 for 4 consecutive trials, 4 trials were
    added to the length of both blocks. This procedure was implemented to keep mice from choosing
    one spout until the reward probability became high again.

    """

    def __init__(
        self,
        rwd_prob_array=[0.1, 0.5, 0.9],
        block_min=20,
        block_max=35,
        persev_add=True,
        perseverative_limit=4,
        max_block_tally=4,  # Max number of consecutive blocks in which one side is better
        **kwargs,
    ) -> None:
        """Init"""
        super().__init__(**kwargs)

        self.rwd_prob_array = rwd_prob_array
        self.block_min = block_min
        self.block_max = block_max
        self.persev_add = persev_add
        self.perseverative_limit = perseverative_limit
        self.max_block_tally = max_block_tally

        self.block_stagger = int((round(block_max - block_min - 0.5) / 2 + block_min) / 2)

    def reset(self, seed=None):
        """Reset the task"""
        self.rwd_tally = [0, 0]  # List for 'L' and 'R'

        self.block_ends = [[], []]  # List for 'L' and 'R', Trial number on which each block ends
        self.block_rwd_prob = [[], []]  # List for 'L' and 'R', Reward probability
        self.block_ind = [
            0,
            0,
        ]  # List for 'L' and 'R', Index of current block (= len(block_end_at_trial))

        self.force_by_tally = [[], []]  # List for 'L' and 'R'
        self.force_by_both_lowest = [[], []]  # List for 'L' and 'R'

        # Anti-persev
        self.persev_consec_on_min_prob = [0, 0]  # List for 'L' and 'R'
        self.persev_add_at_trials = []

        # Manually block hold
        self.hold_this_block = False

        return super().reset()

    def generate_new_trial(self):
        """Generate a new trial. Overwrite the base class method."""
        msg = ""

        if self.trial == 0:
            self.generate_first_block()

        # Block switch?
        if not self.hold_this_block:
            for s in [L, R]:
                if self.trial >= self.block_ends[s][self.block_ind[s]]:
                    # In case a block is mannually 'held', update the actual block transition
                    self.block_ends[s][self.block_ind[s]] = self.trial

                    self.block_ind[s] += 1
                    self.block_effective_ind += 1
                    msg = (
                        self.generate_next_block(
                            s, check_higher_in_a_row=True, check_both_lowest=True
                        )
                        + "\n"
                    )

        # Fill new value
        self.trial_p_reward[self.trial, :] = [
            self.block_rwd_prob[L][self.block_ind[L]],
            self.block_rwd_prob[R][self.block_ind[R]],
        ]

        # Anti-persev
        if not self.hold_this_block and self.persev_add and self.trial > 0:
            msg = msg + self.auto_shape_perseverance()
        else:
            for s in [L, R]:
                self.persev_consec_on_min_prob[s] = 0

        assert self.block_ind[L] + 1 == len(self.block_rwd_prob[L]) == len(self.block_ends[L])
        assert self.block_ind[R] + 1 == len(self.block_rwd_prob[R]) == len(self.block_ends[R])

        return (
            [
                self.trial_p_reward[self.trial - 1, s] != self.trial_p_reward[self.trial, s]
                for s in [L, R]
            ]  # Whether block just switched
            if self.trial > 0
            else [0, 0]
        ), msg

    def generate_first_block(self):
        """Generate the first block. Note the stagger is applied."""
        for side in [L, R]:
            self.generate_next_block(side)

        # Avoid both blocks have the lowest reward prob
        while np.all([x[0] == np.min(self.rwd_prob_array) for x in self.block_rwd_prob]):
            self.block_rwd_prob[self.rng.choice([L, R])][0] = self.rng.choice(
                self.rwd_prob_array
            )  # Random change one side to another prob

        # Start with block stagger: the lower side makes the first block switch earlier
        smaller_side = np.argmin([self.block_rwd_prob[L][0], self.block_rwd_prob[R][0]])
        self.block_ends[smaller_side][0] -= self.block_stagger

        self.block_effective_ind = 1  # Effective block ind

    def generate_next_block(self, side, check_higher_in_a_row=True, check_both_lowest=True):
        """Generate the next block for both sides (yes, very complicated logic...)"""
        msg = ""
        other_side = R if side == L else L
        random_block_len = self.rng.integers(low=self.block_min, high=self.block_max + 1)

        if self.block_ind[side] == 0:  # The first block
            self.block_ends[side].append(random_block_len)
            self.block_rwd_prob[side].append(self.rng.choice(self.rwd_prob_array))

        else:  # Not the first block
            self.block_ends[side].append(
                random_block_len + self.block_ends[side][self.block_ind[side] - 1]
            )

            # If this side has higher prob for too long, force it to be the lowest
            if check_higher_in_a_row:
                # For each effective block, update number of times each side >= the other side
                this_prev = self.block_rwd_prob[side][self.block_ind[side] - 1]
                other_now = self.block_rwd_prob[other_side][self.block_ind[other_side]]
                if this_prev > other_now:
                    self.rwd_tally[side] += 1
                    self.rwd_tally[other_side] = 0
                elif this_prev == other_now:
                    self.rwd_tally[side] += 1
                    self.rwd_tally[other_side] += 1
                else:
                    self.rwd_tally[other_side] += 1
                    self.rwd_tally[side] = 0

                if (
                    self.rwd_tally[side] >= self.max_block_tally
                ):  # Only check higher-in-a-row for this side
                    msg = (
                        f"--- {self.trial}: {side} is higher for {self.rwd_tally[side]} "
                        f"eff_blocks, force {side} to lowest ---\n"
                    )
                    logger.info(msg)
                    self.block_rwd_prob[side].append(min(self.rwd_prob_array))
                    self.rwd_tally[side] = self.rwd_tally[other_side] = 0
                    self.force_by_tally[side].append(self.trial)
                else:  # Otherwise, randomly choose one
                    self.block_rwd_prob[side].append(self.rng.choice(self.rwd_prob_array))
            else:
                self.block_rwd_prob[side].append(self.rng.choice(self.rwd_prob_array))

            # Don't repeat the previous rwd prob
            # (this will not mess up with the "forced" case since the previous block cannot be
            # the lowest prob in the first place)
            while self.block_rwd_prob[side][-2] == self.block_rwd_prob[side][-1]:
                self.block_rwd_prob[side][-1] = self.rng.choice(self.rwd_prob_array)

            # If the other side is already at the lowest prob AND this side just generates the same
            # (either through "forced" case or not), push the previous lowest side to a higher prob
            if check_both_lowest and self.block_rwd_prob[side][-1] == self.block_rwd_prob[
                other_side
            ][-1] == min(self.rwd_prob_array):
                # Stagger this side
                self.block_ends[side][-1] -= self.block_stagger

                # Force block switch of the other side
                msg += f"--- {self.trial}: both side is the lowest, push {side} to higher ---"
                logger.info(msg)
                self.force_by_both_lowest[side].append(self.trial)
                self.block_ends[other_side][-1] = self.trial
                self.block_ind[
                    other_side
                ] += (
                    1  # Two sides change at the same time, no need to add block_effective_ind twice
                )
                self.generate_next_block(
                    other_side, check_higher_in_a_row=False, check_both_lowest=False
                )  # Just generate new block, no need to do checks
        return msg

    def auto_shape_perseverance(self):
        """Anti-perseverance mechanism

        See Grossman et al. 2022:

        If a mouse perseverated on a spout with reward probability of 0.1 for 4 consecutive trials,
        4 trials were added to the length of both blocks. This procedure was implemented to keep
        mice from choosing one spout until the reward probability became high again.
        """
        msg = ""
        for s in [L, R]:
            if self.action[self.trial - 1] == s:  # Note that self.trial already increased 1
                self.persev_consec_on_min_prob[1 - s] = (
                    0  # Reset other side as soon as there is an opposite choice
                )
                if self.trial_p_reward[self.trial - 1, s] == min(
                    self.rwd_prob_array
                ):  # If last choice is on side with min_prob (0.1), add counter
                    self.persev_consec_on_min_prob[s] += 1

        for s in [L, R]:
            if self.persev_consec_on_min_prob[s] >= self.perseverative_limit:
                for ss in [L, R]:
                    self.block_ends[ss][
                        -1
                    ] += self.perseverative_limit  # Add 'perseverative_limit' trials to both blocks
                    self.persev_consec_on_min_prob[ss] = 0
                msg = f"persev at side = {s}, added {self.perseverative_limit} trials to both sides"
                logger.info(msg)
                self.persev_add_at_trials.append(self.trial)
        return msg

    def plot_reward_schedule(self):
        """Plot the reward schedule with annotations showing forced block switches"""
        fig, ax = plt.subplots(2, 1, figsize=[15, 7], sharex="col")

        def annotate_block(ax):
            """Annotate block switches with special handling"""
            for s, col in zip([L, R], ["r", "b"]):
                [
                    ax.axvline(x + (0.1 if s == R else 0), 0, 1, color=col, ls="--", lw=0.5)
                    for x in self.block_ends[s]
                ]

                ax.plot(
                    self.force_by_tally[s],
                    [1.2] * len(self.force_by_tally[s]),
                    marker=">",
                    ls="",
                    color=col,
                    label="forced by tally",
                )
                ax.plot(
                    self.force_by_both_lowest[s],
                    [1.1] * len(self.force_by_both_lowest[s]),
                    marker="v",
                    ls="",
                    color=col,
                    label="forced by both lowest",
                )

            for s, col, pos, m in zip(
                [L, R, IGNORE], ["r", "b", "k"], [0, 1, 0.95], ["|", "|", "x"]
            ):
                this_choice = np.where(self.action == s)
                ax.plot(this_choice, [pos] * len(this_choice), m, color=col)

            ax.plot(
                self.persev_add_at_trials,
                [1.05] * len(self.persev_add_at_trials),
                marker="+",
                label=f"anti-persev added {self.perseverative_limit} trials",
                ls="",
                color="c",
            )

        for s, col in zip([L, R], ["r", "b"]):
            ax[0].plot(self.trial_p_reward[:, s], col, marker=".", alpha=0.5, lw=2)
        annotate_block(ax[0])

        ax[1].plot(
            self.trial_p_reward.sum(axis=1),
            label="sum",
        )
        ax[1].plot(
            self.trial_p_reward[:, R] / self.trial_p_reward.sum(axis=1),
            label="R/(L+R)",
        )
        ax[1].legend()
        annotate_block(ax[1])

        return fig, ax
