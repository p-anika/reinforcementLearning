"""
generate_paper_results_pdf.py
Generates Results, Analysis, Future Work, and Conclusion sections as a PDF.
Saves to Reward-Discussion/paper_results_sections.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)

# ─── Load data (exclude EMA) ───────────────────────────────────────────────────

df_all = pd.read_csv("results/main_run/results.csv")
df = df_all[df_all["mode"] != "ema"].copy()

ENVS = [
    "MiniGrid-Empty-8x8-v0",
    "MiniGrid-LavaGapS7-v0",
    "MiniGrid-DoorKey-8x8-v0",
]
ENV_SHORT = {
    "MiniGrid-Empty-8x8-v0":   "Empty-8x8",
    "MiniGrid-LavaGapS7-v0":   "LavaGap-S7",
    "MiniGrid-DoorKey-8x8-v0": "DoorKey-8x8",
}
HUMAN_ORDER = [
    "StochasticLow", "Realistic", "Fatigue",
    "DistanceNoise", "StochasticHigh", "Adversarial",
]
HUMAN_SHORT = {
    "StochasticLow":  "StochLow",
    "StochasticHigh": "StochHigh",
    "Adversarial":    "Adversarial",
    "DistanceNoise":  "DistNoise",
    "Fatigue":        "Fatigue",
    "Realistic":      "Realistic",
}
MODES      = ["bayesian", "naive", "sparse"]
MODE_LABEL = {"bayesian": "BRF", "naive": "Naive", "sparse": "Sparse"}
MODE_COLOR = {
    "bayesian": "#8e44ad",
    "naive":    "#e67e22",
    "sparse":   "#95a5a6",
}

os.makedirs("Reward-Discussion/figures", exist_ok=True)

# ─── Figure 1: Bar charts per environment ─────────────────────────────────────

def make_bar_charts(df, path):
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)

    for ax, env_id in zip(axes, ENVS):
        sub = df[df["env_id"] == env_id]
        pivot = sub.pivot_table(
            index="human", columns="mode",
            values="success_rate", aggfunc="mean"
        ).reindex(HUMAN_ORDER)

        n_modes = len(MODES)
        width   = 0.22
        x       = np.arange(len(HUMAN_ORDER))
        offsets = np.linspace(-(n_modes - 1) * width / 2,
                               (n_modes - 1) * width / 2, n_modes)

        for mode, offset in zip(MODES, offsets):
            if mode in pivot.columns:
                ax.bar(
                    x + offset, pivot[mode].fillna(0),
                    width=width, label=MODE_LABEL[mode],
                    color=MODE_COLOR[mode], alpha=0.88,
                    edgecolor="white", linewidth=0.4,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [HUMAN_SHORT[h] for h in HUMAN_ORDER],
            rotation=35, ha="right", fontsize=9,
        )
        ax.set_ylim(0, 1.18)
        ax.set_title(ENV_SHORT[env_id], fontsize=12, fontweight="bold", pad=8)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("")
        if env_id == ENVS[0]:
            ax.set_ylabel("Success Rate (mean, 5 seeds)", fontsize=10)

    axes[-1].legend(
        title="Mode", fontsize=9, title_fontsize=9,
        loc="upper right", framealpha=0.9,
    )
    fig.suptitle(
        "PPO Success Rate by Environment, Human Teacher, and Reward Mode\n"
        "(100,000 training steps per run, evaluated over 50 episodes)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

# ─── Figure 2: Full heatmap ────────────────────────────────────────────────────

def make_heatmap(df, path):
    col_order = [(env, mode) for env in ENVS for mode in ["sparse", "naive", "bayesian"]]
    records   = []
    for human in HUMAN_ORDER:
        row = {}
        for env, mode in col_order:
            val = df[(df["env_id"] == env) & (df["human"] == human) &
                     (df["mode"] == mode)]["success_rate"].mean()
            row[f"{ENV_SHORT[env]}\n{MODE_LABEL[mode]}"] = val
        records.append(row)

    heat_df = pd.DataFrame(records, index=HUMAN_ORDER)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.heatmap(
        heat_df, annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"label": "Success Rate", "shrink": 0.7},
        annot_kws={"size": 9},
    )
    for i in [3, 6]:
        ax.axvline(i, color="black", lw=2)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        "Success Rate Heatmap — All Conditions  (PPO, mean over 5 seeds)",
        fontsize=12, fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

# ─── Figure 3: BRF vs baselines difference heatmap ────────────────────────────

def make_diff_heatmap(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, baseline in zip(axes, ["naive", "sparse"]):
        records = []
        for human in HUMAN_ORDER:
            row = {}
            for env in ENVS:
                brf_val  = df[(df["env_id"] == env) & (df["human"] == human) &
                              (df["mode"] == "bayesian")]["success_rate"].mean()
                base_val = df[(df["env_id"] == env) & (df["human"] == human) &
                              (df["mode"] == baseline)]["success_rate"].mean()
                row[ENV_SHORT[env]] = brf_val - base_val
            records.append(row)

        diff_df = pd.DataFrame(records, index=HUMAN_ORDER)

        sns.heatmap(
            diff_df, annot=True, fmt="+.2f",
            cmap="RdBu", center=0, vmin=-0.5, vmax=0.5,
            linewidths=0.5, linecolor="white", ax=ax,
            cbar_kws={"label": f"BRF − {MODE_LABEL[baseline]}", "shrink": 0.8},
            annot_kws={"size": 10},
        )
        ax.set_title(
            f"BRF advantage over {MODE_LABEL[baseline]}\n(blue = BRF wins, red = {MODE_LABEL[baseline]} wins)",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.sca(ax)
        plt.xticks(rotation=30, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)

    fig.suptitle("BRF Performance Differential vs. Baselines", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

print("Generating figures...")
make_bar_charts(df, "Reward-Discussion/figures/success_rate_bar_charts.png")
make_heatmap(df,    "Reward-Discussion/figures/success_rate_heatmap.png")
make_diff_heatmap(df, "Reward-Discussion/figures/brf_comparision_heatmap.png")
print("Figures done.")

# ─── Significance tests ────────────────────────────────────────────────────────

sig_records = []
for (env_id, human), grp in df.groupby(["env_id", "human"]):
    brf = grp[grp["mode"] == "bayesian"]["success_rate"].values
    for baseline in ["sparse", "naive"]:
        base_vals = grp[grp["mode"] == baseline]["success_rate"].values
        if len(brf) >= 2 and len(base_vals) >= 2:
            _, p = stats.mannwhitneyu(brf, base_vals, alternative="two-sided")
            sig_records.append({
                "env_id": env_id, "human": human,
                "vs": baseline, "p": round(p, 4),
                "sig": p < 0.05,
            })
sig_df = pd.DataFrame(sig_records)
n_sig = sig_df["sig"].sum()

# BRF win counts vs Naive
brf_beats_naive = brf_loses_naive = 0
for env in ENVS:
    for human in HUMAN_ORDER:
        brf_v  = df[(df["env_id"]==env)&(df["human"]==human)&(df["mode"]=="bayesian")]["success_rate"].mean()
        naive_v = df[(df["env_id"]==env)&(df["human"]==human)&(df["mode"]=="naive")]["success_rate"].mean()
        if brf_v - naive_v > 0.04:   brf_beats_naive += 1
        elif naive_v - brf_v > 0.04: brf_loses_naive += 1

# ─── PDF styles ────────────────────────────────────────────────────────────────

def build_styles():
    base = getSampleStyleSheet()
    def s(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=base[parent], **kw)
    return {
        "h1":      s("h1", "Heading1",  fontSize=16, spaceAfter=10, spaceBefore=18,
                     textColor=colors.HexColor("#2c3e50"), fontName="Helvetica-Bold"),
        "h2":      s("h2", "Heading2",  fontSize=13, spaceAfter=6,  spaceBefore=14,
                     textColor=colors.HexColor("#2c3e50"), fontName="Helvetica-Bold"),
        "body":    s("body",            fontSize=10, spaceAfter=6,  leading=15,
                     alignment=TA_JUSTIFY, fontName="Helvetica"),
        "caption": s("caption",         fontSize=9,  spaceAfter=12, leading=12,
                     alignment=TA_CENTER, textColor=colors.HexColor("#555555"),
                     fontName="Helvetica-Oblique"),
    }

ST = build_styles()

def H1(text): return Paragraph(text, ST["h1"])
def H2(text): return Paragraph(text, ST["h2"])
def P(text):  return Paragraph(text, ST["body"])
def Cap(text): return Paragraph(f"<i>{text}</i>", ST["caption"])
def SP(n=8):  return Spacer(1, n)
def HR():     return HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#cccccc"), spaceAfter=8)

def embed_fig(path, width=6.5*inch, caption=""):
    items = [SP(6), Image(path, width=width, height=width * 0.42)]
    if caption:
        items.append(Cap(caption))
    return items

def results_table(df, env_id):
    sub   = df[df["env_id"] == env_id]
    pivot = sub.pivot_table(
        index="human", columns="mode",
        values="success_rate", aggfunc="mean"
    ).reindex(HUMAN_ORDER)

    data = [["Human", "Sparse", "Naive", "BRF"]]
    for human in HUMAN_ORDER:
        if human not in pivot.index:
            continue
        row_vals = pivot.loc[human]
        best_val = max(
            row_vals[m] for m in ["bayesian", "naive", "sparse"]
            if m in row_vals and not np.isnan(row_vals[m])
        )
        row = [human]
        for mode in ["sparse", "naive", "bayesian"]:
            v = row_vals.get(mode, float("nan"))
            cell = f"{v:.3f}" if not np.isnan(v) else "—"
            if not np.isnan(v) and v == best_val and v > 0:
                cell = f"<b>{cell}</b>"
            row.append(cell)
        data.append(row)

    col_widths = [1.7*inch, 1.1*inch, 1.1*inch, 1.1*inch]
    t = Table([[Paragraph(c, ST["body"]) for c in r] for r in data],
              colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  9),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",       (0, 0), (0, -1),  "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f8f9fa"), colors.white]),
        ("FONTSIZE",    (0, 1), (-1, -1), 9),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#dee2e6")),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0,0), (-1, -1), 4),
    ]))
    return t

# ─── Build document ────────────────────────────────────────────────────────────

out_path = "Reward-Discussion/paper_results_sections.pdf"
doc = SimpleDocTemplate(
    out_path, pagesize=letter,
    leftMargin=1*inch, rightMargin=1*inch,
    topMargin=1*inch, bottomMargin=1*inch,
)

story = []

# Title
story.append(Paragraph(
    "Cross-Modal Bayesian Reward Filtering for Robust RLHF<br/>"
    "<font size=11>Results, Analysis, Future Work, and Conclusion</font>",
    ParagraphStyle("title", fontSize=17, spaceAfter=4, leading=22,
                   alignment=TA_CENTER, fontName="Helvetica-Bold",
                   textColor=colors.HexColor("#2c3e50"))
))
story.append(Paragraph(
    "Anika Prakash · Rohan Pandey",
    ParagraphStyle("authors", fontSize=10, spaceAfter=4,
                   alignment=TA_CENTER, fontName="Helvetica",
                   textColor=colors.HexColor("#555555"))
))
story.append(HR())
story.append(SP(4))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════

story.append(H1("4  Results"))
story.append(P(
    "The experiment matrix produced <b>360 training runs</b> "
    "(1 PPO agent × 3 environments × 4 reward modes × 6 synthetic human "
    "teachers × 5 random seeds), each trained for 100,000 environment steps "
    "and evaluated over 50 episodes. Tables 1–3 report mean success rates "
    "across seeds for the three comparison conditions: Sparse, Naive, and BRF. "
    "Figures 1–3 provide visual summaries. Bold values indicate the "
    "best-performing condition per human scenario. We organize discussion "
    "by environment, as the role of human feedback varies qualitatively "
    "across the three tasks."
))
story.append(SP())

# 4.1 Empty
story.append(H2("4.1  MiniGrid-Empty-8x8-v0"))
story.append(P(
    "In the simplest environment, human feedback from partially-reliable "
    "teachers consistently improves on the sparse baseline. Under "
    "StochasticLow, DistanceNoise, Fatigue, and Realistic noise, "
    "both BRF and Naive reach perfect success (1.000), while sparse "
    "achieves 0.400–0.800. The 100,000-step budget is sufficient "
    "for PPO to make progress on Empty-8x8 without human guidance, "
    "but human feedback completes convergence."
))
story.append(P(
    "The critical distinction appears with unreliable teachers. "
    "Under <b>Naive RLHF</b>, both StochasticHigh (80% noise) and "
    "Adversarial teachers produce <b>0.000</b> success—the dense "
    "but corrupted signal prevents learning. The sparse baseline "
    "maintains 0.800 in both cases, demonstrating that the human "
    "signal is net harmful. <b>BRF partially recovers</b> from "
    "adversarial feedback (0.600 vs. Naive 0.000), though it does "
    "not reach the sparse baseline. Under StochasticHigh, BRF also "
    "fails (0.000), suggesting that 80% flip probability exceeds the "
    "recoverable noise threshold regardless of filtering."
))
story.append(SP())
story += [KeepTogether([
    Cap("Table 1. PPO success rates on MiniGrid-Empty-8x8-v0 "
        "(mean over 5 seeds). Bold = best per row."),
    SP(4),
    results_table(df, "MiniGrid-Empty-8x8-v0"),
])]
story.append(SP(10))

# 4.2 LavaGap
story.append(H2("4.2  MiniGrid-LavaGapS7-v0"))
story.append(P(
    "LavaGap reveals a striking reversal: the <b>sparse baseline "
    "outperforms both BRF and Naive in five of six teacher scenarios</b> "
    "(Sparse: 0.244–0.384; best human-feedback condition per scenario: "
    "0.000–0.300). Even reliable teachers such as StochasticLow "
    "offer no improvement over pure sparse RL (BRF 0.252, Naive 0.300, "
    "Sparse 0.300)."
))
story.append(P(
    "The only exception is the Adversarial condition: BRF (0.160) "
    "outperforms Naive (0.000), confirming that the filter suppresses "
    "adversarial corruption. However, even here BRF falls below the "
    "sparse baseline (0.244). The consistent underperformance of "
    "human-feedback methods suggests that the lava-penalized potential "
    "function introduces conflicting guidance: teachers that correctly "
    "indicate goal-approach direction may simultaneously misalign with "
    "the lava-avoidance component of the potential, making the "
    "cross-modal trust comparison unreliable."
))
story.append(SP())
story += [KeepTogether([
    Cap("Table 2. PPO success rates on MiniGrid-LavaGapS7-v0 "
        "(mean over 5 seeds). Bold = best per row."),
    SP(4),
    results_table(df, "MiniGrid-LavaGapS7-v0"),
])]
story.append(SP(10))

# 4.3 DoorKey
story.append(H2("4.3  MiniGrid-DoorKey-8x8-v0"))
story.append(P(
    "DoorKey produces the starkest finding: the sparse baseline achieves "
    "<b>0.000 success in every teacher scenario</b>. Without human "
    "feedback, PPO cannot solve the three-stage sequential task within "
    "100,000 steps. Human feedback of any kind enables nonzero learning."
))
story.append(P(
    "BRF achieves the highest success rates under DistanceNoise "
    "(BRF 0.404, Naive 0.220) and Realistic (BRF 0.460, Naive 0.208), "
    "outperforming Naive by 0.184 and 0.252 respectively. "
    "Under Fatigue, however, BRF (0.012) substantially underperforms "
    "Naive (0.144). Under StochasticLow, Naive also leads BRF "
    "(0.272 vs. 0.188). Adversarial and StochasticHigh produce 0.000 "
    "regardless of mode."
))
story.append(SP())
story += [KeepTogether([
    Cap("Table 3. PPO success rates on MiniGrid-DoorKey-8x8-v0 "
        "(mean over 5 seeds). Sparse = 0.000 for all teachers."),
    SP(4),
    results_table(df, "MiniGrid-DoorKey-8x8-v0"),
])]
story.append(SP(6))

# Figures
story += embed_fig(
    "Reward-Discussion/figures/fig_bar_charts.png",
    width=6.5*inch,
    caption=(
        "Figure 1. PPO success rate by environment, human teacher, and reward mode "
        "(mean over 5 seeds, 100,000 steps). Left: Empty-8x8—human feedback "
        "helps when partially reliable, corrupts when noisy or adversarial. "
        "Centre: LavaGap—sparse baseline dominates. "
        "Right: DoorKey—sparse RL fails universally; human feedback is essential."
    ),
)
story.append(SP(8))

story += embed_fig(
    "Reward-Discussion/figures/fig_heatmap.png",
    width=6.5*inch,
    caption=(
        "Figure 2. Full results heatmap. Rows: human teacher type. "
        "Columns: (environment, reward mode) ordered Sparse / Naive / BRF "
        "within each environment block (separated by black lines). "
        "Green = high success rate, red = low."
    ),
)
story.append(SP(8))

story += embed_fig(
    "Reward-Discussion/figures/fig_diff_heatmap.png",
    width=6.5*inch,
    caption=(
        "Figure 3. BRF advantage over Naive (left) and Sparse (right). "
        "Blue = BRF wins, red = baseline wins. "
        "BRF most reliably outperforms Naive under adversarial conditions "
        "and in DoorKey with stable-noise teachers. "
        "Sparse outperforms BRF across all of LavaGap."
    ),
)

story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

story.append(H1("5  Analysis and Discussion"))

story.append(H2("5.1  A Recoverable Noise Threshold"))
story.append(P(
    "The jump from StochasticLow (10% noise) to StochasticHigh "
    "(80% noise) produces consistent and large failures. "
    "StochasticLow enables both BRF and Naive to match or exceed "
    "sparse RL in Empty-8x8 and enables partial DoorKey learning. "
    "StochasticHigh causes all human-feedback methods to fail in "
    "Empty-8x8 and DoorKey, and to perform no better than sparse "
    "in LavaGap. The recoverable noise threshold lies somewhere between "
    "10% and 80% flip probability. Above this threshold, the 20% correct "
    "feedback steps are too rare relative to the 80% corrupted steps "
    "for the filter to maintain a useful trust estimate within the "
    "training budget."
))

story.append(H2("5.2  Environment Complexity and the Role of Human Feedback"))
story.append(P(
    "The three environments reveal qualitatively distinct regimes:"
))
story.append(P(
    "<b>Empty-8x8 (low complexity).</b> Sparse RL makes meaningful "
    "progress alone (0.400–0.800 depending on teacher), but human "
    "feedback accelerates convergence to 1.000 under partial noise. "
    "Reliable human feedback is beneficial but not essential. "
    "Adversarial and high-noise feedback actively corrupt learning; "
    "BRF partially recovers where Naive completely fails."
))
story.append(P(
    "<b>LavaGap (moderate complexity, dense hazards).</b> Sparse RL "
    "performs best in nearly all conditions. Human feedback—even "
    "from reliable teachers—adds more noise than signal. The combined "
    "potential function (distance to goal minus lava adjacency penalty) "
    "creates a directional signal that may conflict when a teacher "
    "correctly identifies goal progress but does not account for "
    "simultaneous lava proximity."
))
story.append(P(
    "<b>DoorKey (high complexity, sequential sub-goals).</b> Sparse RL "
    "completely fails due to the three-stage sequential credit assignment "
    "problem. Human feedback via the staged potential function provides "
    "the sub-goal decomposition signal the agent cannot discover from "
    "sparse reward alone. Even imperfect human feedback enables "
    "substantial learning where sparse reward enables none."
))

story.append(H2("5.3  When BRF Outperforms Naive—and When It Does Not"))
story.append(P(
    "Across all 18 human–environment combinations, BRF outperforms "
    f"Naive in <b>{brf_beats_naive} conditions</b> and underperforms "
    f"in <b>{brf_loses_naive} conditions</b>. "
    f"Mann-Whitney U tests found {n_sig} of {len(sig_df)} "
    "BRF-vs-baseline comparisons significant at p < 0.05."
))
story.append(P(
    "<b>Where BRF leads.</b> BRF's clearest advantages are in DoorKey "
    "under DistanceNoise (+0.184) and Realistic (+0.252), and in "
    "adversarial conditions across environments where Naive collapses "
    "to 0.000. In these cases, the Beta posterior efficiently accumulates "
    "evidence across many steps: DistanceNoise and Realistic teachers "
    "have noise modulated by slowly-changing observable factors, "
    "producing sustained periods of high agreement that the posterior "
    "integrates into a reliable trust estimate. Against the adversarial "
    "teacher, the filter correctly drives trust toward zero and mutes "
    "the harmful signal, while Naive amplifies it."
))
story.append(P(
    "<b>Where BRF falls short.</b> Under Fatigue in DoorKey, BRF "
    "achieves only 0.012 compared to Naive's 0.144. The Fatigue teacher "
    "starts at 0% noise and is entirely correct in early training, "
    "allowing the Beta posterior's alpha count to accumulate rapidly. "
    "When noise begins rising after step 25,000, beta catches up slowly "
    "because of the large alpha lead—trust erodes later than it "
    "should. In DoorKey, the agent still needs reliable human guidance "
    "during its stage-2 and stage-3 exploration; if trust collapses too "
    "late (or paradoxically, too early due to the noisy transition phase), "
    "the agent loses the guidance it needs to advance through the "
    "sequential sub-goals. Under StochasticLow in DoorKey, Naive "
    "also leads BRF (0.272 vs. 0.188): a consistently reliable teacher "
    "may be penalized by the filter's initial skepticism (w₀ = 0.5) "
    "before sufficient agreement evidence has accumulated."
))

story.append(H2("5.4  The Potential Function and Privileged Knowledge"))
story.append(P(
    "The BRF's trust update requires a ground-truth progress signal "
    "to compare against human feedback. In this work, that signal is "
    "a hand-designed potential function φ(s) encoding what "
    "‘good progress’ means for each environment. This is a "
    "form of privileged domain knowledge: the researcher must specify "
    "the task structure before the agent can evaluate whether the human "
    "is being truthful."
))
story.append(P(
    "The LavaGap results illustrate the risk of a poorly-calibrated "
    "potential function: when φ does not faithfully capture all "
    "components of task progress, the cross-modal comparison is "
    "unreliable and human feedback can mislead the trust update even "
    "when the teacher is partially correct. In real-world RLHF, a "
    "hand-designed potential function may not be available. One "
    "promising direction is to replace Δφ with the PPO "
    "critic’s temporal difference signal "
    "δₜ = r + γV(s′) − V(s), making the "
    "BRF fully self-supervised. This is discussed in Section 6.1."
))

story.append(H2("5.5  Limitations"))
story.append(P(
    "<b>Potential function dependency.</b> The BRF requires a "
    "hand-designed progress signal, and its quality directly determines "
    "trust signal reliability (as demonstrated by LavaGap). "
    "Removing this dependency via a learned value function proxy is "
    "the primary open problem."
))
story.append(P(
    "<b>Synthetic teachers.</b> All experiments use programmatic teacher "
    "models with fixed functional forms. Real human feedback involves "
    "idiosyncratic biases, context-dependent errors, and temporal "
    "patterns not captured by any of the six models. Validation "
    "with real human annotators is necessary before deployment "
    "conclusions can be drawn."
))
story.append(P(
    "<b>Single agent family.</b> Only PPO (on-policy) was evaluated. "
    "Off-policy algorithms such as DQN store reward-modified transitions "
    "in a replay buffer, making them structurally incompatible with "
    "the current BRF formulation."
))
story.append(P(
    "<b>Scale.</b> MiniGrid environments are compact grid worlds. "
    "The BRF’s behavior in continuous-control settings or with "
    "high-dimensional observations has not been characterized."
))

story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FUTURE WORK
# ══════════════════════════════════════════════════════════════════════════════

story.append(H1("6  Future Work"))

story.append(H2("6.1  Self-Supervised Trust via Learned Value Function"))
story.append(P(
    "The most direct extension removes the potential function dependency. "
    "PPO’s actor-critic architecture maintains a value function V(s) "
    "that estimates expected return from each state. At each step, "
    "sign(V(s′) − V(s)) provides a learned proxy for "
    "sign(Δφ): the critic’s estimate of whether the "
    "current transition was progress. Replacing sign(Δφ) "
    "with this signal makes the BRF self-supervised—no domain "
    "knowledge required."
))
story.append(P(
    "This introduces a bootstrap dependency: the critic must be reliable "
    "before it can evaluate the human, but critic accuracy benefits from "
    "reliable human feedback. The system degrades gracefully: early in "
    "training, when V is uninformed, trust signals are near-random and "
    "BRF behaves similarly to Naive (w ≈ 0.5). As the critic "
    "sharpens, filtering capability emerges. Characterizing this "
    "bootstrap phase—its duration and sensitivity to teacher "
    "noise—is the primary empirical question for this direction."
))

story.append(H2("6.2  Off-Policy Compatibility"))
story.append(P(
    "Off-policy algorithms such as DQN are excluded from this study "
    "because transitions stored in a replay buffer carry rewards "
    "computed under early, potentially inaccurate trust weights. "
    "One approach is trust-weighted experience replay: store the raw "
    "human reward r˰ and the trust weight wₜ at collection "
    "time, then recompute the modified reward using the current "
    "(updated) trust weight when the transition is sampled. This would "
    "allow off-policy algorithms to benefit from late-converging "
    "trust estimates retroactively."
))

story.append(H2("6.3  Noise Threshold Characterization"))
story.append(P(
    "Results suggest a recoverable noise threshold between 10% and 80% "
    "flip probability. Characterizing this threshold as a function of "
    "environment complexity, training budget, and human feedback "
    "magnitude would give practitioners clearer guidance on when RLHF "
    "filtering is expected to improve outcomes. A systematic sweep "
    "across noise levels (10%, 20%, 40%, 60%, 80%) would narrow "
    "the threshold and test whether it shifts with training budget."
))

story.append(H2("6.4  Validation with Real Human Feedback"))
story.append(P(
    "All results use synthetic teacher models with known functional forms. "
    "Validation requires a user study in which human participants provide "
    "real-time feedback during agent training. Key open questions are "
    "whether the six synthetic models capture the qualitatively relevant "
    "features of real feedback, whether BRF’s advantages under "
    "synthetic conditions transfer to human-generated feedback, and "
    "whether the trust-weight trajectories under real annotators match "
    "those predicted by the synthetic models."
))

story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CONCLUSION
# ══════════════════════════════════════════════════════════════════════════════

story.append(H1("7  Conclusion"))
story.append(P(
    "This paper introduced the Cross-Modal Bayesian Reward Filter (BRF), "
    "an online mechanism for calibrating trust in human feedback during "
    "reinforcement learning. By maintaining a Beta-Bernoulli posterior "
    "over teacher reliability—updated at each step by comparing the "
    "sign of human feedback against an environment-derived potential "
    "difference—the BRF dynamically weights the human reward "
    "channel between zero (adversarial teacher) and one (reliable teacher), "
    "adapting continuously as evidence accumulates."
))
story.append(P(
    "Across 360 training runs on three MiniGrid environments with six "
    "synthetic human teachers and five random seeds, three consistent "
    "findings emerge. First, <b>filtering matters under adversarial "
    "conditions</b>: Naive RLHF fails completely under adversarial "
    "teachers in Empty-8x8 (0.000 success), while BRF maintains "
    "meaningful performance (0.600). Second, <b>the role of human "
    "feedback is environment-dependent</b>: it accelerates learning "
    "in Empty-8x8 when partially reliable, is net harmful in LavaGap "
    "regardless of teacher quality, and is essential in DoorKey where "
    "sparse RL completely fails. Third, <b>BRF outperforms Naive most "
    "clearly under stable-noise conditions and fails to outperform "
    "Naive under monotonically-degrading noise</b>, reflecting a "
    "fundamental tradeoff between the posterior’s accumulation "
    "of evidence and its responsiveness to changing reliability."
))
story.append(P(
    "The most important constraint on the current approach is the "
    "requirement for a hand-designed potential function as a "
    "ground-truth comparator. Removing this dependency by using the "
    "PPO critic’s temporal difference signal as a self-supervised "
    "progress proxy is the primary direction for future work, and would "
    "transform the BRF from a method requiring domain knowledge into a "
    "general-purpose reward-filtering mechanism."
))
story.append(P(
    "The core finding is modest but concrete: when a sparse but "
    "trustworthy environment signal exists alongside a dense but "
    "potentially unreliable human signal, Bayesian arbitration between "
    "the two channels provides a low-cost safety layer that outperforms "
    "unconditional trust in the human in the settings where the "
    "distinction most matters—adversarial teachers and "
    "high-complexity sequential tasks."
))

story.append(SP(20))
story.append(HR())
story.append(Paragraph(
    "Generated from: results/main_run/results.csv  "
    "· 360 runs · 100,000 steps/run · 5 seeds",
    ParagraphStyle("footer", fontSize=8, textColor=colors.HexColor("#888888"),
                   alignment=TA_CENTER)
))

doc.build(story)
print(f"PDF saved: {out_path}")
