# reinforcementLearning

# Full matrix (2 agents × 3 envs × 3 modes × 6 humans × 3 seeds = 324 runs)
# python run_experiment.py

# Single combo (quick test)
# python run_experiment.py --agents PPO --envs MiniGrid-Empty-8x8-v0 \
#     --modes bayesian --humans StochasticLow --seeds 42 --timesteps 500

# Preview matrix without running
# python run_experiment.py --dry-run

# Generate plots from saved results
# python analyze_results.py --results-dir results/<run-name>
