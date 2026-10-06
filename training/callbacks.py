"""
training/callbacks.py
---------------------
SB3 callback that logs the Bayesian trust weight and PPO policy loss.

analyze_results.py uses these logs to plot trust and loss curves.
"""

from stable_baselines3.common.callbacks import BaseCallback


class ResearchLoggerCallback(BaseCallback):
    """
    Records, at every environment step:

    trust_scores  : the wrapper's current trust weight w
    policy_losses : PPO's most recent policy-gradient loss. It is None until
                    the first update and only changes once per rollout.
    """

    def __init__(self):
        super().__init__()
        self.trust_scores = []
        self.policy_losses = []

    def _on_step(self) -> bool:
        # get_attr reads an attribute from the env inside SB3's VecEnv wrapper
        self.trust_scores.append(float(self.training_env.get_attr("current_w")[0]))
        # No `or` fallback: a loss of exactly 0.0 is still a valid value
        self.policy_losses.append(
            self.model.logger.name_to_value.get("train/policy_gradient_loss")
        )
        return True  # returning False would stop training early