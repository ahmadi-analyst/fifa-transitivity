"""
Generate all figures for the FIFA World Cup Transitivity project.

Figures saved to figures/
  01_transitivity_by_year.png       — bar chart, rate per tournament
  02_transitivity_by_rank_gap.png   — bar chart, rate by A-vs-C rank gap bucket
  03_rank_gap_scatter.png           — jitter scatter: rank_gap_ac vs transitive
  04_group_network_sample.png       — match network for one illustrative group
"""

import pathlib
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import logit

warnings.filterwarnings("ignore")

PROC = pathlib.Path(__file__).parent.parent / "data" / "processed"
FIG = pathlib.Path(__file__).parent.parent / "figures"
FIG.mkdir(exist_ok=True)

PALETTE = {"transitive": "#2196F3", "not_transitive": "#EF5350", "neutral": "#78909C"}
sns.set_theme(style="whitegrid", font_scale=1.1)


def load_data():
    triples = pd.read_csv(PROC / "transitivity_triples.csv")
    matches = pd.read_csv(PROC / "group_stage_matches.csv")
    app = triples[triples["is_transitivity_triple"] == 1].copy()
    app["transitive"] = app["transitive"].astype(int)
    return triples, matches, app


# ---------------------------------------------------------------------------
# Figure 1: Transitivity rate by year
# ---------------------------------------------------------------------------
def fig_by_year(app):
    tbl = (
        app.groupby("year")["transitive"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "k", "count": "n"})
        .reset_index()
    )
    tbl["rate"] = tbl["k"] / tbl["n"]
    # Wilson 95% CI
    ci_lo, ci_hi = [], []
    for _, row in tbl.iterrows():
        res = stats.binomtest(int(row["k"]), int(row["n"]), p=0.5)
        ci = res.proportion_ci(confidence_level=0.95)
        ci_lo.append(ci.low)
        ci_hi.append(ci.high)
    tbl["ci_lo"] = ci_lo
    tbl["ci_hi"] = ci_hi
    tbl["err_lo"] = tbl["rate"] - tbl["ci_lo"]
    tbl["err_hi"] = tbl["ci_hi"] - tbl["rate"]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE["transitive"] if r >= 0.5 else PALETTE["not_transitive"]
              for r in tbl["rate"]]
    bars = ax.bar(tbl["year"].astype(str), tbl["rate"], color=colors, width=0.6,
                  yerr=[tbl["err_lo"], tbl["err_hi"]], capsize=5,
                  error_kw={"elinewidth": 1.5, "ecolor": "#555"})
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="50% (chance)")
    for bar, (_, row) in zip(bars, tbl.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                f"{int(row['k'])}/{int(row['n'])}", ha="center", va="bottom",
                fontsize=9, color="#333")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("World Cup Year")
    ax.set_ylabel("Transitivity Rate")
    ax.set_title("Group-Stage Transitivity Rate by Tournament\n"
                 "(A beats B, B beats C → does A beat C?)", pad=12)
    ax.legend()
    fig.tight_layout()
    out = FIG / "01_transitivity_by_year.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Figure 2: Transitivity rate by ranking-gap bucket
# ---------------------------------------------------------------------------
def fig_by_rank_gap(app):
    sub = app.dropna(subset=["rank_gap_ac"]).copy()
    sub["rank_gap_ac"] = sub["rank_gap_ac"].astype(float)
    bins = [-200, -10, 10, 200]
    labels = ["A better\n(>=10 ranks)", "Similar\n(±10)", "C better\n(>=10 ranks)"]
    sub["bucket"] = pd.cut(sub["rank_gap_ac"], bins=bins, labels=labels)
    tbl = (
        sub.groupby("bucket", observed=True)["transitive"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "k", "count": "n", "mean": "rate"})
        .reset_index()
    )
    # Per-bucket 95% CIs and p-values (binomial, H0: rate = 0.50)
    ci_lo, ci_hi, p_vals = [], [], []
    for _, row in tbl.iterrows():
        res = stats.binomtest(int(row["k"]), int(row["n"]), p=0.5, alternative="two-sided")
        ci = res.proportion_ci(confidence_level=0.95)
        ci_lo.append(ci.low)
        ci_hi.append(ci.high)
        p_vals.append(res.pvalue)
    tbl["ci_lo"] = ci_lo
    tbl["ci_hi"] = ci_hi
    tbl["p_val"] = p_vals
    tbl["err_lo"] = tbl["rate"] - tbl["ci_lo"]
    tbl["err_hi"] = tbl["ci_hi"] - tbl["rate"]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [PALETTE["transitive"] if r >= 0.5 else PALETTE["not_transitive"]
              for r in tbl["rate"]]
    bars = ax.bar(tbl["bucket"].astype(str), tbl["rate"], color=colors, width=0.6,
                  yerr=[tbl["err_lo"], tbl["err_hi"]], capsize=5,
                  error_kw={"elinewidth": 1.5, "ecolor": "#555"})
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="50% (chance)")
    for bar, (_, row) in zip(bars, tbl.iterrows()):
        cap_top = row["rate"] + row["err_hi"]
        sig = "*" if row["p_val"] < 0.05 else ""
        # count (and significance star) just above the error bar cap
        ax.text(bar.get_x() + bar.get_width() / 2, cap_top + 0.02,
                f"{int(row['k'])}/{int(row['n'])}{sig}", ha="center", va="bottom",
                fontsize=9, color="#333")
        # 95% CI above the count
        ax.text(bar.get_x() + bar.get_width() / 2, cap_top + 0.07,
                f"[{row['ci_lo']:.2f}, {row['ci_hi']:.2f}]", ha="center", va="bottom",
                fontsize=7.5, color="#555")
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("Pre-Tournament Ranking: Team A vs Team C\n(rank_A − rank_C; negative = A ranked higher/better)")
    ax.set_ylabel("Transitivity Rate")
    ax.set_title("Transitivity Rate by Ranking Gap (A vs C)\n"
                 "Among triples where A beat B and B beat C\n"
                 "Error bars: 95% CI  |  * p < 0.05 vs 50% (binomial test)", pad=12)
    ax.legend()
    fig.tight_layout()
    out = FIG / "02_transitivity_by_rank_gap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Figure 3: Jitter scatter — rank_gap_ac vs transitive outcome
# ---------------------------------------------------------------------------
def fig_scatter(app):
    sub = app.dropna(subset=["rank_gap_ac"]).copy()
    sub["rank_gap_ac"] = sub["rank_gap_ac"].astype(float)
    sub["jitter_y"] = sub["transitive"] + np.random.default_rng(42).uniform(-0.08, 0.08, len(sub))

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sub["transitive"].map({1: PALETTE["transitive"], 0: PALETTE["not_transitive"]})
    ax.scatter(sub["rank_gap_ac"], sub["jitter_y"], c=colors, alpha=0.55, s=30, linewidths=0)

    # Logistic curve overlay
    x_range = np.linspace(sub["rank_gap_ac"].min(), sub["rank_gap_ac"].max(), 300)
    m = logit("transitive ~ rank_gap_ac", data=sub).fit(disp=False)
    y_hat = m.predict(pd.DataFrame({"rank_gap_ac": x_range}))
    ax.plot(x_range, y_hat, color="black", linewidth=2, label="Logistic fit")

    ax.axvline(0, color="#aaa", linestyle=":", linewidth=1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Not transitive (0)", "Transitive (1)"])
    ax.set_xlabel("Ranking Gap: rank_A − rank_C\n(negative = A ranked higher/better)")
    ax.set_title("Rank Gap vs Transitivity Outcome\nwith Logistic Regression Fit", pad=12)
    blue_patch = mpatches.Patch(color=PALETTE["transitive"], label="Transitive")
    red_patch = mpatches.Patch(color=PALETTE["not_transitive"], label="Not transitive")
    ax.legend(handles=[blue_patch, red_patch, plt.Line2D([0], [0], color="black", lw=2,
                                                          label="Logistic fit")])
    fig.tight_layout()
    out = FIG / "03_rank_gap_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Figure 4: Match network for one illustrative group
# ---------------------------------------------------------------------------
def fig_network(matches):
    # Pick a group with at least one intransitive triple — 2022 Group C is interesting
    target_year, target_group = 2022, "Group C"
    sub = matches[(matches["year"] == target_year) & (matches["group_name"] == target_group)]
    if sub.empty:
        target_year, target_group = 2018, "Group F"
        sub = matches[(matches["year"] == target_year) & (matches["group_name"] == target_group)]

    teams = sorted(set(sub["home_team"]) | set(sub["away_team"]))
    n = len(teams)

    # Circular layout: pos stores (cos(angle), sin(angle)) per team
    angles = [2 * np.pi * i / n for i in range(n)]
    pos = {t: (np.cos(a), np.sin(a)) for t, a in zip(teams, angles)}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw nodes — x/y are cos/sin of the team's angle, so they double as direction vectors
    for team, (x, y) in pos.items():
        ax.scatter(x, y, s=800, color=PALETTE["neutral"], zorder=3)
        ha = "left" if x > 0.3 else ("right" if x < -0.3 else "center")
        va = "bottom" if y > 0.3 else ("top" if y < -0.3 else "center")
        ax.text(x + 0.28 * x, y + 0.28 * y, team, ha=ha, va=va, fontsize=16, fontweight="bold")

    # Draw edges
    for _, row in sub.iterrows():
        h, a = row["home_team"], row["away_team"]
        hx, hy = pos[h]
        ax, ay = pos[a]
        if row["home_pts"] > 0.5:
            winner, loser = h, a
        elif row["away_pts"] > 0.5:
            winner, loser = a, h
        else:
            winner = loser = None  # draw

        wx, wy = pos[winner] if winner else ((hx + ax) / 2, (hy + ay) / 2)
        lx, ly = pos[loser] if loser else ((hx + ax) / 2, (hy + ay) / 2)

        if winner:
            color = PALETTE["transitive"]
            fig.axes[0].annotate(
                "", xy=(lx * 0.82 + wx * 0.18, ly * 0.82 + wy * 0.18),
                xytext=(wx * 0.82 + lx * 0.18, wy * 0.82 + ly * 0.18),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8),
            )
        else:
            lx2, ly2 = (hx + ax) / 2, (hy + ay) / 2
            fig.axes[0].plot([hx * 0.85 + lx2 * 0.15, ax * 0.85 + lx2 * 0.15],
                             [hy * 0.85 + ly2 * 0.15, ay * 0.85 + ly2 * 0.15],
                             color=PALETTE["neutral"], lw=1.5, linestyle="--")

        score = f"{int(row['home_score'])}–{int(row['away_score'])}"
        mx, my = (hx + ax) / 2, (hy + ay) / 2
        # Two crossing edges share midpoint (0,0); offset each perpendicularly to separate them
        if abs(mx) < 0.01 and abs(my) < 0.01:
            length = np.hypot(ax - hx, ay - hy)
            px, py = -(ay - hy) / length, (ax - hx) / length
            if py < 0 or (abs(py) < 1e-9 and px < 0):
                px, py = -px, -py
            mx += 0.13 * px
            my += 0.13 * py
        fig.axes[0].text(mx, my, score, ha="center", va="center",
                         fontsize=16, color="#555",
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    ax_obj = fig.axes[0]
    ax_obj.set_xlim(-1.9, 1.9)
    ax_obj.set_ylim(-1.9, 1.9)
    ax_obj.set_title(f"Match Results Network\n{target_year} World Cup — {target_group}",
                     pad=14, fontsize=18)
    fig.tight_layout()
    out = FIG / "04_group_network_sample.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating figures...")
    np.random.seed(42)
    _, matches, app = load_data()
    fig_by_year(app)
    fig_by_rank_gap(app)
    fig_scatter(app)
    fig_network(matches)
    print(f"Done. All figures saved to figures/")
