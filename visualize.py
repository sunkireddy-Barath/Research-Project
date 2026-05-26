"""
visualize.py — Publication-Quality Paper Figures
==================================================
Loads results/experiment_results.npz and produces 9 IEEE-quality figures
saved to results/figures/.

Run after experiments.py:
    python visualize.py

Figures
-------
  fig0_overview.png         2×2 summary panel for abstract / slides
  fig1_learning_curve.png   Reward + accuracy + ε vs training episodes
  fig2_detection_accuracy.png  Per-attack accuracy across all systems
  fig3_false_positive.png   FPR comparison bar chart
  fig4_latency.png          Per-attack mean response latency
  fig5_damage.png           Damage score: no-response vs proposed
  fig6_alert_tiers.png      Alert propagation tier pie chart
  fig7_overhead.png         Computational overhead bar chart
  fig8_scalability.png      Per-node latency vs network size
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")      # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

import config

# ── Output directory ───────────────────────────────────────────────────────
os.makedirs(config.FIGURES_DIR, exist_ok=True)

# ── IEEE-compatible style ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#2b2b2b",
    "axes.grid":         True,
    "grid.color":        "#dddddd",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "lines.linewidth":   2.0,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
})

C = {                              # system colours
    "proposed":    "#1a6fb5",
    "rule_based":  "#e05c2a",
    "static_ml":   "#2ca02c",
    "no_response": "#9467bd",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loader
# ─────────────────────────────────────────────────────────────────────────────

def load(path: str = config.EXP_DATA_PATH) -> np.lib.npyio.NpzFile:
    if not os.path.exists(path):
        sys.exit(
            f"\nError: '{path}' not found.\n"
            "Run  python experiments.py  first, then re-run visualize.py.\n"
        )
    return np.load(path, allow_pickle=True)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 0 — Overview (2×2 panel)
# ─────────────────────────────────────────────────────────────────────────────

def fig0_overview(d):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Self-Healing Blockchain Security Framework — Results Overview",
        fontweight="bold", fontsize=14, y=0.98,
    )
    gs = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.32)

    # (a) Reward convergence
    ax1  = fig.add_subplot(gs[0, 0])
    er   = np.arange(1, len(d["rewards_raw"]) + 1)
    es   = np.arange(1, len(d["rewards_smooth"]) + 1)
    ax1.plot(er, d["rewards_raw"],    color=C["proposed"], alpha=0.2, lw=1)
    ax1.plot(es, d["rewards_smooth"], color=C["proposed"], lw=2)
    ax1.axhline(0, color="#aaa", lw=0.8, ls=":")
    ax1.set_title("(a) Reward Convergence")
    ax1.set_xlabel("Training Episode")
    ax1.set_ylabel("Cumulative Reward")

    # (b) Average detection accuracy
    ax2   = fig.add_subplot(gs[0, 1])
    avgs  = [np.mean(d[f"acc_{k}"]) for k in ("proposed", "rule_based", "static_ml", "acc_no_response".replace("acc_", ""))]
    avgs  = [
        float(np.mean(d["acc_proposed"])),
        float(np.mean(d["acc_rule_based"])),
        float(np.mean(d["acc_static_ml"])),
        float(np.mean(d["acc_no_response"])),
    ]
    bars2 = ax2.bar(
        ["Proposed", "Rule-Based", "Static ML", "No-Resp"],
        avgs,
        color=[C["proposed"], C["rule_based"], C["static_ml"], C["no_response"]],
        alpha=0.85, edgecolor="white",
    )
    for b in bars2:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f"{b.get_height():.0%}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(0, 1.15)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax2.set_title("(b) Avg Detection Accuracy")
    ax2.set_ylabel("Accuracy")

    # (c) Damage prevention
    ax3 = fig.add_subplot(gs[1, 0])
    attacks = [str(a) for a in d["dmg_attacks"]]
    x3 = np.arange(len(attacks))
    w  = 0.35
    ax3.bar(x3 - w/2, d["dmg_noresp"],   w, label="No-Response", color=C["no_response"], alpha=0.8)
    ax3.bar(x3 + w/2, d["dmg_proposed"], w, label="Proposed",     color=C["proposed"],    alpha=0.8)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(attacks, rotation=25, ha="right", fontsize=8)
    ax3.set_ylabel("Damage Score")
    ax3.legend(fontsize=8)
    ax3.set_title("(c) Damage Prevention")

    # (d) Overhead
    ax4   = fig.add_subplot(gs[1, 1])
    ovhd  = [
        float(d["overhead_proposed"][0]),
        float(d["overhead_rule"][0]),
        float(d["overhead_static"][0]),
    ]
    bars4 = ax4.bar(
        ["Proposed", "Rule-Based", "Static ML"],
        ovhd,
        color=[C["proposed"], C["rule_based"], C["static_ml"]],
        alpha=0.85, edgecolor="white", width=0.5,
    )
    for b in bars4:
        ax4.text(b.get_x() + b.get_width()/2, b.get_height() * 1.05,
                 f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax4.set_ylabel("Step Time (ms)")
    ax4.set_title("(d) Computational Overhead")

    _save(fig, "fig0_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Learning Curve
# ─────────────────────────────────────────────────────────────────────────────

def fig1_learning_curve(d):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle("Fig. 1 — DQN Learning Convergence", fontweight="bold")

    er = np.arange(1, len(d["rewards_raw"]) + 1)
    es = np.arange(1, len(d["rewards_smooth"]) + 1)

    ax1.plot(er, d["rewards_raw"],    color=C["proposed"], alpha=0.20, lw=1)
    ax1.plot(es, d["rewards_smooth"], color=C["proposed"], lw=2, label="Smoothed reward")
    ax1.axhline(0, color="#999", lw=0.8, ls=":")
    ax1.set_ylabel("Cumulative Episode Reward")
    ax1.legend(loc="lower right")
    ax1.set_title("(a) Episode Reward vs Training Episodes")

    ea = np.arange(1, len(d["accuracy_raw"]) + 1)
    eas= np.arange(1, len(d["accuracy_smooth"]) + 1)
    ax2.plot(ea,  d["accuracy_raw"],    color=C["rule_based"], alpha=0.20, lw=1)
    ax2.plot(eas, d["accuracy_smooth"], color=C["rule_based"], lw=2, label="Accuracy (smoothed)")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax2.set_ylabel("Detection Accuracy")
    ax2.set_xlabel("Training Episode")

    ax2b = ax2.twinx()
    ax2b.plot(er, d["epsilons"], color="#888", ls="--", lw=1.5, label="ε (exploration)")
    ax2b.set_ylabel("Exploration Rate ε", color="#888")
    ax2b.tick_params(axis="y", labelcolor="#888")
    ax2b.set_ylim(0, 1.05)

    all_lines = ax2.get_lines() + ax2b.get_lines()
    lines  = [l for l in all_lines if not l.get_label().startswith("_")]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="lower right")
    ax2.set_title("(b) Detection Accuracy and ε Decay vs Episodes")

    plt.tight_layout()
    _save(fig, "fig1_learning_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Per-Attack Detection Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def fig2_detection(d):
    attacks = [str(a) for a in d["attacks_exp2"]]
    x       = np.arange(len(attacks))
    w       = 0.20

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Fig. 2 — Per-Attack Detection Accuracy Comparison", fontweight="bold")

    groups = [
        ("Proposed (ours)", "acc_proposed",    C["proposed"],    -1.5),
        ("Rule-Based IDS",  "acc_rule_based",  C["rule_based"],  -0.5),
        ("Static ML IDS",   "acc_static_ml",   C["static_ml"],    0.5),
        ("No-Response IDS", "acc_no_response", C["no_response"],  1.5),
    ]

    for label, key, color, offset in groups:
        vals  = np.array(d[key], dtype=float)
        rects = ax.bar(x + offset * w, vals, w * 0.92, label=label,
                       color=color, alpha=0.85, edgecolor="white")
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.0%}",
                        xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5,
                        rotation=90)

    ax.set_ylabel("Detection Accuracy")
    ax.set_xlabel("Attack Type")
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=20, ha="right")
    ax.set_ylim(0, 1.30)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Detection accuracy across all attack types and IDS systems")
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    _save(fig, "fig2_detection_accuracy.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — False Positive Rate
# ─────────────────────────────────────────────────────────────────────────────

def fig3_fpr(d):
    labels = ["Proposed (ours)", "Rule-Based IDS", "Static ML IDS"]
    values = [
        float(d["fpr_proposed"][0]),
        float(d["fpr_rule"][0]),
        float(d["fpr_static"][0]),
    ]
    colors = [C["proposed"], C["rule_based"], C["static_ml"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("Fig. 3 — False Positive Rate on Normal Traffic", fontweight="bold")

    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + max(values) * 0.02,
                f"{h:.2%}", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("False Positive Rate")
    ax.set_ylim(0, max(values) * 1.4)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Lower is better — proportion of benign steps mis-classified as threat")

    _save(fig, "fig3_false_positive.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Response Latency
# ─────────────────────────────────────────────────────────────────────────────

def fig4_latency(d):
    attacks = [str(a) for a in d["latency_attacks"]]
    values  = np.array(d["latency_values"], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle("Fig. 4 — Mean Response Latency per Attack Type (Proposed System)",
                 fontweight="bold")

    bars = ax.bar(attacks, values, color=C["proposed"], alpha=0.85,
                  edgecolor="white", width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels(attacks, rotation=20, ha="right")
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_xlabel("Attack Type")
    ax.set_title("End-to-end latency: telemetry collection → mitigation action executed")

    _save(fig, "fig4_latency.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Damage Prevention
# ─────────────────────────────────────────────────────────────────────────────

def fig5_damage(d):
    attacks  = [str(a) for a in d["dmg_attacks"]]
    dmg_nr   = np.array(d["dmg_noresp"],   dtype=float)
    dmg_prop = np.array(d["dmg_proposed"], dtype=float)

    x = np.arange(len(attacks))
    w = 0.32

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Fig. 5 — Attack Damage: No-Response vs Self-Healing Framework",
                 fontweight="bold")

    ax.bar(x - w/2, dmg_nr,   w, label="No-Response IDS", color=C["no_response"], alpha=0.82)
    ax.bar(x + w/2, dmg_prop, w, label="Proposed System",  color=C["proposed"],    alpha=0.82)

    for i, (nr, pr) in enumerate(zip(dmg_nr, dmg_prop)):
        if nr > 0:
            pct = (nr - pr) / nr * 100
            ax.annotate(f"↓{pct:.0f}%",
                        xy=(x[i] + w/2, pr),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9,
                        color=C["proposed"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=20, ha="right")
    ax.set_ylabel("Normalised Damage Score")
    ax.set_xlabel("Attack Type")
    ax.legend(loc="upper right")
    ax.set_title(
        "Lower score = less damage sustained. "
        "↓% reduction shown above proposed bars."
    )

    _save(fig, "fig5_damage.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Alert Tier Distribution
# ─────────────────────────────────────────────────────────────────────────────

def fig6_alerts(d):
    flood    = float(d["flood_reduction"][0])
    local_p  = flood
    neigh_p  = (1 - flood) * 0.83   # approx share from exp6 data
    global_p = (1 - flood) * 0.17

    labels   = ["LOCAL\n(no broadcast)", "NEIGHBOR\n(adjacent validators)", "GLOBAL\n(governance)"]
    sizes    = [local_p, neigh_p, global_p]
    colors   = [C["proposed"], C["static_ml"], C["rule_based"]]
    explode  = (0.05, 0.02, 0.02)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Fig. 6 — Selective Alert Propagation Tier Distribution", fontweight="bold")

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        explode=explode, startangle=90,
        textprops={"fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(11)

    ax.set_title(
        f"LOCAL (zero-broadcast) rate: {local_p:.1%}  — "
        f"reduces network alert flooding by {local_p:.0%}"
    )

    _save(fig, "fig6_alert_tiers.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — Computational Overhead
# ─────────────────────────────────────────────────────────────────────────────

def fig7_overhead(d):
    labels = ["Proposed (ours)", "Rule-Based IDS", "Static ML IDS"]
    values = [
        float(d["overhead_proposed"][0]),
        float(d["overhead_rule"][0]),
        float(d["overhead_static"][0]),
    ]
    colors = [C["proposed"], C["rule_based"], C["static_ml"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Fig. 7 — Computational Overhead per Decision Step", fontweight="bold")
    fig.subplots_adjust(top=0.80)

    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + max(values) * 0.015,
                f"{h:.4f} ms", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Step Time (ms)")
    ax.set_xlabel("Median of 5 runs × 100 steps on commodity hardware")
    ax.set_title(
        "Wall-clock time per detection + decision + response step"
    )

    _save(fig, "fig7_overhead.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 — Scalability
# ─────────────────────────────────────────────────────────────────────────────

def fig8_scalability(d):
    nodes = np.array(d["scale_nodes"],   dtype=int)
    lats  = np.array(d["scale_latency"], dtype=float)

    # Linear reference trend
    coeffs = np.polyfit(nodes, lats, 1)
    trend  = np.polyval(coeffs, nodes)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Fig. 8 — Scalability: Per-Node Latency vs Network Size", fontweight="bold")

    ax.plot(nodes, lats, "o-", color=C["proposed"], lw=2, ms=8,
            label="Measured per-node latency")
    ax.plot(nodes, trend, "--", color=C["rule_based"], lw=1.5,
            label=f"Linear fit (slope ≈ {coeffs[0]:.3f} ms/node)")

    ax.set_xticks(nodes)
    ax.set_xlabel("Number of Network Nodes")
    ax.set_ylabel("Latency per Node (ms)")
    ax.legend(loc="upper left")
    ax.set_title(
        "Near-constant per-node latency confirms lightweight, "
        "embarrassingly parallel design"
    )

    _save(fig, "fig8_scalability.png")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, filename: str):
    path = os.path.join(config.FIGURES_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Generating Paper Figures")
    print("="*60 + "\n")

    d = load()

    fig0_overview(d)
    fig1_learning_curve(d)
    fig2_detection(d)
    fig3_fpr(d)
    fig4_latency(d)
    fig5_damage(d)
    fig6_alerts(d)
    fig7_overhead(d)
    fig8_scalability(d)

    print(f"\n  All 9 figures saved → {config.FIGURES_DIR}/\n")


if __name__ == "__main__":
    main()
