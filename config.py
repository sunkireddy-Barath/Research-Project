"""
config.py — Centralised Hyperparameter Configuration
======================================================
Single source of truth for every tunable parameter in the
Self-Healing Blockchain Security Framework.

Changing a value here propagates automatically to v3.py, baselines.py,
experiments.py, and visualize.py without touching algorithm code.

Sections
--------
  REPRODUCIBILITY  Random seeds
  DQN AGENT        Network architecture and RL hyperparameters
  REWARD FUNCTION  Multi-objective reward weights  (Rt = αCt − βOt − γFt − δLt)
  DETECTORS        Thresholds and weights for the hybrid anomaly engine
  FRAMEWORK        Anomaly gate and autoencoder pre-training settings
  ALERT POLICY     Selective propagation tier thresholds
  TRAINING         Episode counts and traffic mix
  EVALUATION       Steps per eval episode
  PATHS            Output file locations
"""

# ═══════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42

# ═══════════════════════════════════════════════════════════════════════════
# DQN AGENT
# ═══════════════════════════════════════════════════════════════════════════
#
# State vector is 7-dimensional:
#   [composite_anomaly, network_score, consensus_score, application_score,
#    tx_failure_ratio, validator_deviation, contract_risk]
#
# Per-layer scores allow the agent to distinguish layer-specific attacks
# (e.g. Sybil has high network_score but low consensus/application) from
# benign traffic where all three layer scores remain moderate.

STATE_DIM          = 7     # composite + 3 layer scores + 3 telemetry signals
ACTION_DIM         = 5     # NO_ACTION, ALERT, THROTTLE, ISOLATE, PAUSE_CONTRACT
HIDDEN_DIM         = 32    # hidden units in the 2-layer Q-network
LEARNING_RATE      = 1e-3  # SGD learning rate for policy network updates

GAMMA              = 0.95  # discount factor for future rewards
EPSILON_INIT       = 1.0   # initial exploration rate
EPSILON_MIN        = 0.05  # minimum exploration floor
EPSILON_DECAY      = 0.995 # multiplicative decay applied after each training step

BATCH_SIZE         = 32    # experience replay mini-batch size
BUFFER_CAPACITY    = 10_000  # maximum transitions stored in replay buffer
TARGET_UPDATE_FREQ = 100   # steps between policy→target network sync

# ═══════════════════════════════════════════════════════════════════════════
# REWARD FUNCTION   Rt = α·Ct − β·Ot − γ·Ft − δ·Lt
# ═══════════════════════════════════════════════════════════════════════════

REWARD_ALPHA          = 1.0  # α — containment effectiveness weight
REWARD_BETA           = 0.3  # β — operational overhead weight
REWARD_GAMMA          = 0.5  # γ — false-positive penalty weight
REWARD_DELTA          = 0.2  # δ — response latency penalty weight
BENIGN_ACTION_PENALTY = 0.8  # extra penalty for acting on normal traffic

# ═══════════════════════════════════════════════════════════════════════════
# HYBRID ANOMALY DETECTORS
# ═══════════════════════════════════════════════════════════════════════════

# Layer weights (must sum to 1.0)
WEIGHT_NETWORK     = 0.35  # L1 — Isolation Forest (network / peer-level)
WEIGHT_CONSENSUS   = 0.35  # L2 — Statistical Validator (consensus / voting)
WEIGHT_APPLICATION = 0.30  # L3 — Autoencoder (smart-contract execution)

# Isolation Forest: fires when |z-score| of any network feature exceeds this.
# Set to 3.5σ so that max(|Z1|,|Z2|,|Z3|) on i.i.d. N(0,1) gives FPR < 1%.
IF_THRESHOLD       = 3.5

# Statistical Validator rolling-window length
STAT_WINDOW        = 50

# Autoencoder architecture and training
AE_LATENT_DIM      = 2
AE_LR              = 0.01

# ═══════════════════════════════════════════════════════════════════════════
# FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════

# Anomaly gate: steps with composite score below this threshold are assigned
# NO_ACTION without invoking the RL agent.
# Calibrated at the midpoint between normal-traffic baseline (~0.42) and the
# weakest attack signal (collusion at ~0.46).
ACTION_GATE = 0.44

# Autoencoder pre-training on normal telemetry before the RL loop
AE_PRETRAIN_SAMPLES = 500
AE_PRETRAIN_EPOCHS  = 30

# ═══════════════════════════════════════════════════════════════════════════
# SELECTIVE ALERT PROPAGATION POLICY
# ═══════════════════════════════════════════════════════════════════════════
#
# Tier 1 LOCAL    score < LOCAL_THRESHOLD  or  containment > 0.7
#         → resolve silently, zero network broadcast
# Tier 2 NEIGHBOR score < NEIGHBOR_THRESHOLD
#         → propagate to adjacent validators
# Tier 3 GLOBAL   score ≥ NEIGHBOR_THRESHOLD
#         → escalate to on-chain governance

LOCAL_THRESHOLD    = 0.50
NEIGHBOR_THRESHOLD = 0.75

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

N_EPISODES        = 500   # total training episodes
STEPS_PER_EPISODE = 20    # timesteps per episode
NORMAL_FRACTION   = 0.40  # fraction of episodes drawn from NORMAL traffic
                          # (remaining 60% split across 6 attack types)

# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

EVAL_STEPS = 100  # timesteps used for each post-training evaluation episode

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════

MODEL_PATH    = "dqn_weights.npz"
RESULTS_DIR   = "results"
FIGURES_DIR   = "results/figures"
EXP_DATA_PATH = "results/experiment_results.npz"
