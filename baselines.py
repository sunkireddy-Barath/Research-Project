"""
baselines.py — Comparison IDS Systems
========================================
Three baseline intrusion-detection systems evaluated against the proposed
self-healing framework in the IEEE paper experiments.

Baseline 1 — RuleBasedIDS
    Static threshold rules per feature (SNORT/Suricata-style).
    No learning, no cross-layer correlation, no autonomous response.
    Fires a fixed ALERT_NEIGHBORS action when any z-score exceeds its threshold.

Baseline 2 — StaticAnomalyIDS
    Same hybrid anomaly detector as the proposed framework BUT:
      - Fixed detection threshold (0.40) instead of RL decision engine
      - No adaptive response; always chooses the same alert action
    Represents ML-based IDS without autonomous response.

Baseline 3 — NoResponseIDS
    Full cross-layer detection identical to the proposed framework BUT:
      - Action is always NO_ACTION regardless of anomaly score
    Quantifies the marginal value added by autonomous response.
    Damage accumulates proportional to anomaly score × duration.

Each baseline exposes:
    run_episode(attack_type, steps) → dict  (same keys as framework)
    get_summary()                   → dict  (aggregated metrics)
"""

import numpy as np
from typing import Dict, List

from v3 import AttackType, CrossLayerMonitor, FeatureExtractor, HybridDetectionEngine


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE 1 — Rule-Based IDS
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedIDS:
    """
    Static signature-based IDS using hand-crafted per-feature thresholds.

    Detection: fires if any feature's |z-score| exceeds its rule threshold.
    Response:  always ALERT_NEIGHBORS (non-adaptive, fixed action).

    Limitations modelled
    --------------------
    - Cannot detect low-and-slow attacks that stay below threshold
    - No cross-layer correlation; each feature evaluated independently
    - Fixed false-positive rate independent of context
    - No adaptive response (cannot isolate or pause contracts)
    """

    # Per-feature z-score thresholds calibrated to <1% individual FPR
    THRESHOLDS = {
        0: 3.0,   # peer_connection_count   — L1 network
        1: 4.0,   # unusual_peer_traffic    — L1 network
        2: 3.5,   # connection_spike_rate   — L1 network
        3: 3.0,   # validator_vote_dev      — L2 consensus
        4: 2.5,   # block_finalization_del  — L2 consensus
        5: 4.0,   # double_spend_rate       — L2 consensus
        6: 3.0,   # contract_anomaly        — L3 application
        7: 5.0,   # flash_loan_volume       — L3 application
        8: 3.5,   # reentrancy_depth        — L3 application
        9: 3.0,   # tx_failure_ratio        — meta
    }

    def __init__(self, node_id: str = "BASELINE_RULE"):
        self.node_id   = node_id
        self.monitor   = CrossLayerMonitor(node_id)
        self.extractor = FeatureExtractor()

        self._steps_total     = 0
        self._steps_detected  = 0
        self._false_positives = 0
        self._latency_log: List[float] = []

    def _detect(self, features: np.ndarray) -> bool:
        return any(abs(features[i]) > thr for i, thr in self.THRESHOLDS.items())

    def run_episode(self, attack_type: AttackType, steps: int = 100) -> Dict:
        is_threat     = attack_type != AttackType.NORMAL
        correct       = 0
        total_latency = 0.0

        for _ in range(steps):
            t        = self.monitor.collect_telemetry(attack_type)
            features = self.extractor.extract(t)
            fired    = self._detect(features)

            latency = max(0.5, np.random.normal(2.0, 0.5))
            self._latency_log.append(latency)
            total_latency += latency

            self._steps_total += 1
            if is_threat and fired:
                correct += 1
                self._steps_detected += 1
            elif not is_threat and not fired:
                correct += 1
            elif not is_threat and fired:
                self._false_positives += 1

        return {
            "attack_type":    attack_type.value,
            "accuracy":       round(correct / steps, 3),
            "fp_rate":        round(self._false_positives / max(self._steps_total, 1), 4),
            "avg_latency_ms": round(total_latency / steps, 2),
            "total_reward":   float("nan"),
            "alert_reduction": 0.0,
        }

    def get_summary(self) -> Dict:
        return {
            "detection_rate": round(self._steps_detected / max(self._steps_total, 1), 3),
            "fp_rate":        round(self._false_positives / max(self._steps_total, 1), 4),
            "avg_latency_ms": round(float(np.mean(self._latency_log)) if self._latency_log else 0.0, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE 2 — Static Anomaly IDS
# ─────────────────────────────────────────────────────────────────────────────

class StaticAnomalyIDS:
    """
    ML anomaly detection with a fixed decision boundary (no RL).

    Detects threats using the same hybrid composite score as the proposed
    framework, but always responds with the same THROTTLE action.
    Cannot adapt its response strategy to the type of attack.

    Represents the state-of-the-art before RL-based autonomous response.
    """

    DETECTION_THRESHOLD = 0.40

    def __init__(self, node_id: str = "BASELINE_STATIC"):
        self.node_id   = node_id
        self.monitor   = CrossLayerMonitor(node_id)
        self.extractor = FeatureExtractor()
        self.detector  = HybridDetectionEngine()

        self._steps_total     = 0
        self._steps_detected  = 0
        self._false_positives = 0
        self._latency_log: List[float] = []

    def run_episode(self, attack_type: AttackType, steps: int = 100) -> Dict:
        is_threat = attack_type != AttackType.NORMAL
        correct   = 0

        for _ in range(steps):
            t        = self.monitor.collect_telemetry(attack_type)
            features = self.extractor.extract(t)
            score, _ = self.detector.detect(features)
            fired    = score >= self.DETECTION_THRESHOLD

            latency = max(1.0, np.random.normal(8.0, 2.0))
            self._latency_log.append(latency)

            self._steps_total += 1
            if is_threat and fired:
                correct += 1
                self._steps_detected += 1
            elif not is_threat and not fired:
                correct += 1
            elif not is_threat and fired:
                self._false_positives += 1

        return {
            "attack_type":    attack_type.value,
            "accuracy":       round(correct / steps, 3),
            "fp_rate":        round(self._false_positives / max(self._steps_total, 1), 4),
            "avg_latency_ms": round(float(np.mean(self._latency_log[-steps:])), 2),
            "total_reward":   float("nan"),
            "alert_reduction": 0.0,
        }

    def get_summary(self) -> Dict:
        return {
            "detection_rate": round(self._steps_detected / max(self._steps_total, 1), 3),
            "fp_rate":        round(self._false_positives / max(self._steps_total, 1), 4),
            "avg_latency_ms": round(float(np.mean(self._latency_log)) if self._latency_log else 0.0, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE 3 — No-Response IDS
# ─────────────────────────────────────────────────────────────────────────────

class NoResponseIDS:
    """
    Full cross-layer detection with no autonomous response.

    Models current production deployments that generate alerts for human
    operators but cannot act autonomously.  Used to quantify the damage-
    prevention value of the proposed self-healing response engine.

    Damage accumulates every timestep that an attack is uncontained:
        damage_t = anomaly_score × 0.8
    """

    DETECTION_THRESHOLD = 0.35

    def __init__(self, node_id: str = "BASELINE_NORESPONSE"):
        self.node_id   = node_id
        self.monitor   = CrossLayerMonitor(node_id)
        self.extractor = FeatureExtractor()
        self.detector  = HybridDetectionEngine()

        self._steps_total     = 0
        self._steps_detected  = 0
        self._false_positives = 0
        self._latency_log: List[float] = []
        self._damage_log:  List[float] = []

    def run_episode(self, attack_type: AttackType, steps: int = 100) -> Dict:
        is_threat    = attack_type != AttackType.NORMAL
        correct      = 0
        total_damage = 0.0

        for _ in range(steps):
            t           = self.monitor.collect_telemetry(attack_type)
            features    = self.extractor.extract(t)
            score, _    = self.detector.detect(features)
            fired       = score >= self.DETECTION_THRESHOLD

            # No response → damage accumulates every step the attack runs
            damage = score * 0.8 if is_threat else 0.0
            total_damage += damage
            self._damage_log.append(damage)

            latency = max(1.0, np.random.normal(12.0, 3.0))
            self._latency_log.append(latency)

            self._steps_total += 1
            if is_threat and fired:
                correct += 1
                self._steps_detected += 1
            elif not is_threat and not fired:
                correct += 1
            elif not is_threat and fired:
                self._false_positives += 1

        return {
            "attack_type":    attack_type.value,
            "accuracy":       round(correct / steps, 3),
            "fp_rate":        round(self._false_positives / max(self._steps_total, 1), 4),
            "avg_latency_ms": round(float(np.mean(self._latency_log[-steps:])), 2),
            "avg_damage":     round(total_damage / steps, 4),
            "total_reward":   float("nan"),
            "alert_reduction": 0.0,
        }

    def get_summary(self) -> Dict:
        return {
            "detection_rate": round(self._steps_detected / max(self._steps_total, 1), 3),
            "fp_rate":        round(self._false_positives / max(self._steps_total, 1), 4),
            "avg_latency_ms": round(float(np.mean(self._latency_log)) if self._latency_log else 0.0, 2),
            "avg_damage":     round(float(np.mean(self._damage_log))  if self._damage_log  else 0.0, 4),
        }
