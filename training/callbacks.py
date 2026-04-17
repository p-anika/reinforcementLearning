"""
training/callbacks.py
---------------------
SB3 training callbacks for the research framework.

ResearchLoggerCallback records per-step trust scores and per-rollout
policy losses during training. These are used by analyze_results.py
to generate trust evolution and loss curves.

Trust scores are recorded every step via the VecEnv attribute interface.
Policy losses are captured from the SB3 internal logger after each
gradient update (populated after rollout for PPO, after each gradient
step for SAC). Entries are None between updates and are filtered
out during analysis.
"""

from stable_baselines3.common.callbacks import BaseCallback


class ResearchLoggerCallback(BaseCallback):
    """
    Logs trust scores and policy losses throughout a training run.

    Attributes
    ----------
    trust_scores  : list[float]        trust weight w at each environment step
    policy_losses : list[float | None] policy gradient loss (PPO) or actor loss
                                       (SAC) after each gradient update;
                                       None when no update has occurred yet
    """

    def __init__(self):
        super().__init__()
        self.trust_scores  = []
        self.policy_losses = []

    def _on_step(self) -> bool:
        # ── Trust score ────────────────────────────────────────────────────────
        # SB3 wraps training environments in a VecEnv.
        # get_attr() is the correct way to read attributes from inside a VecEnv.
        w = self.training_env.get_attr("current_w")[0]
        self.trust_scores.append(float(w))

        # ── Policy loss ────────────────────────────────────────────────────────
        # The SB3 logger is populated after gradient updates, not every step.
        # We probe it each step; it returns None in between updates.
        #   PPO key: "train/policy_gradient_loss"
        #   SAC key: "train/actor_loss"
        logger_vals = self.model.logger.name_to_value
        loss = (
            logger_vals.get("train/policy_gradient_loss")
            or logger_vals.get("train/actor_loss")
        )
        self.policy_losses.append(loss)  # None entries filtered in analyze_results.py

        return True  # Returning False would stop training early
