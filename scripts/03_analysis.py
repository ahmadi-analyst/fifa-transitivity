"""
Statistical analysis of FIFA World Cup group-stage transitivity (1994-2022).

Tests
-----
1. Binomial test: is overall transitivity rate != 0.5 (chance)?
2. Chi-square: does transitivity rate vary across tournaments?
3. Ranking-gap buckets: does A-vs-C rank gap predict transitivity?
4. Logistic regression: rank_gap_ac -> P(transitive)

Reads:  data/processed/transitivity_triples.csv
Prints: results to stdout
"""

import pathlib
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings("ignore")

PROC = pathlib.Path(__file__).parent.parent / "data" / "processed"

N_BOOT = 10_000
BOOT_SEED = 42


def _boot_overall(df, n_boot=N_BOOT, seed=BOOT_SEED):
    """Group-level bootstrap: 95% CI and p-value (H0: rate = 0.5) for overall rate."""
    rng = np.random.default_rng(seed)
    grp = df.groupby(["year", "group_name"])["transitive"].agg(["sum", "count"])
    sums, counts = grp["sum"].to_numpy(float), grp["count"].to_numpy(float)
    observed = sums.sum() / counts.sum()
    n = len(grp)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = sums[idx].sum(1) / counts[idx].sum(1)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    shifted = boot - boot.mean() + 0.5
    p_val = float(np.mean(np.abs(shifted - 0.5) >= abs(observed - 0.5)))
    return ci_lo, ci_hi, p_val


def _boot_buckets(df, bins, labels, n_boot=N_BOOT, seed=BOOT_SEED):
    """Group-level bootstrap: per-bucket 95% CIs and p-values (H0: rate = 0.5)."""
    rng = np.random.default_rng(seed)
    sub = df.dropna(subset=["rank_gap_ac"]).copy()
    sub["rank_gap_ac"] = sub["rank_gap_ac"].astype(float)
    sub["_bi"] = pd.cut(sub["rank_gap_ac"], bins=bins, labels=False)
    sub = sub.dropna(subset=["_bi"])
    sub["_bi"] = sub["_bi"].astype(int)
    grp_keys = sub[["year", "group_name"]].drop_duplicates().reset_index(drop=True)
    n_groups, n_b = len(grp_keys), len(labels)
    g_sums = np.zeros((n_groups, n_b))
    g_counts = np.zeros((n_groups, n_b))
    for gi, (_, row) in enumerate(grp_keys.iterrows()):
        chunk = sub[(sub["year"] == row["year"]) & (sub["group_name"] == row["group_name"])]
        for bi in range(n_b):
            t = chunk.loc[chunk["_bi"] == bi, "transitive"]
            g_sums[gi, bi] = t.sum()
            g_counts[gi, bi] = len(t)
    idx = rng.integers(0, n_groups, size=(n_boot, n_groups))
    boot_s = g_sums[idx].sum(1)
    boot_c = g_counts[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot_r = np.where(boot_c > 0, boot_s / boot_c, np.nan)
    results = []
    for bi, label in enumerate(labels):
        obs = sub.loc[sub["_bi"] == bi, "transitive"]
        obs_rate = obs.mean()
        rates = boot_r[:, bi][~np.isnan(boot_r[:, bi])]
        ci_lo, ci_hi = np.percentile(rates, [2.5, 97.5])
        shifted = rates - rates.mean() + 0.5
        p_val = float(np.mean(np.abs(shifted - 0.5) >= abs(obs_rate - 0.5)))
        results.append({"label": label, "k": int(obs.sum()), "n": len(obs),
                        "rate": obs_rate, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_val": p_val})
    return results


def load_applicable():
    df = pd.read_csv(PROC / "transitivity_triples.csv")
    app = df[df["is_transitivity_triple"] == 1].copy()
    app["transitive"] = app["transitive"].astype(int)
    return df, app


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------------------
# Test 1: Overall binomial test
# ---------------------------------------------------------------------------
def test_overall(app):
    section("TEST 1 — Overall transitivity rate (group-level bootstrap)")
    n = len(app)
    k = app["transitive"].sum()
    rate = k / n
    ci_lo, ci_hi, p_val = _boot_overall(app)
    print(f"  Applicable triples : {n}")
    print(f"  Transitive         : {k}  ({100*rate:.1f}%)")
    print(f"  H0: p = 0.50 (chance)")
    print(f"  Bootstrap 95% CI   : [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Bootstrap p-value  : {p_val:.4f}  (n_boot={N_BOOT}, resampling groups)")
    if p_val < 0.05:
        direction = "above" if rate > 0.5 else "below"
        print(f"  => Significant: transitivity rate is {direction} 0.5 (p < 0.05)")
    else:
        print("  => Not significant: cannot reject p = 0.5")
    return rate


# ---------------------------------------------------------------------------
# Test 2: Chi-square across years
# ---------------------------------------------------------------------------
def test_by_year(app):
    section("TEST 2 — Transitivity rate by year (chi-square)")
    tbl = (
        app.groupby("year")["transitive"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "transitive", "count": "total"})
    )
    tbl["rate"] = tbl["transitive"] / tbl["total"]
    tbl["not_transitive"] = tbl["total"] - tbl["transitive"]
    print(tbl[["total", "transitive", "rate"]].to_string())
    contingency = tbl[["transitive", "not_transitive"]].values
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    print(f"\n  Chi-square: {chi2:.3f}  df={dof}  p={p:.4f}")
    if p < 0.05:
        print("  => Significant year-to-year variation")
    else:
        print("  => No significant year-to-year variation")
    return tbl


# ---------------------------------------------------------------------------
# Test 3: Ranking-gap buckets
# ---------------------------------------------------------------------------
def test_by_rank_gap(app):
    section("TEST 3 -- Transitivity rate by ranking gap (A rank - C rank)")
    sub = app.dropna(subset=["rank_gap_ac"]).copy()
    sub["rank_gap_ac"] = sub["rank_gap_ac"].astype(float)
    # negative gap = A ranked higher (lower rank number) than C
    sub["gap_bucket"] = pd.cut(
        sub["rank_gap_ac"],
        bins=[-200, -10, 10, 200],
        labels=["A better (>=10)", "Similar (±10)", "C better (>=10)"],
    )
    tbl = (
        sub.groupby("gap_bucket", observed=True)["transitive"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "transitive", "count": "total", "mean": "rate"})
    )
    print(tbl.to_string())
    contingency = tbl[["transitive"]].assign(not_t=tbl["total"] - tbl["transitive"]).values
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    print(f"\n  Chi-square across buckets: {chi2:.3f}  df={dof}  p={p:.4f}")

    print("\n  Per-bucket bootstrap tests (H0: rate = 0.50, group-level resampling):")
    print(f"  {'Bucket':<22} {'k':>4} {'n':>4}  {'rate':>5}  {'95% CI':<16}  p-value")
    print("  " + "-" * 62)
    bucket_results = _boot_buckets(
        app, bins=[-200, -10, 10, 200],
        labels=["A better (>=10)", "Similar (+-10)", "C better (>=10)"],
    )
    for r in bucket_results:
        sig = " *" if r["p_val"] < 0.05 else ""
        print(f"  {r['label']:<22} {r['k']:>4} {r['n']:>4}  {r['rate']:>5.3f}  "
              f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  {r['p_val']:.4f}{sig}")
    print("  (* p < 0.05 vs 50%)")
    return tbl, sub


# ---------------------------------------------------------------------------
# Test 4: Logistic regression
# ---------------------------------------------------------------------------
def test_logistic(app):
    section("TEST 4 — Logistic regression: rank_gap_ac -> P(transitive)")
    sub = app.dropna(subset=["rank_gap_ac", "rank_gap_ab", "rank_gap_bc"]).copy()
    for col in ["rank_gap_ac", "rank_gap_ab", "rank_gap_bc"]:
        sub[col] = sub[col].astype(float)

    # Univariate: rank_gap_ac only
    m1 = smf.logit("transitive ~ rank_gap_ac", data=sub).fit(disp=False)
    print("\n  Model 1: transitive ~ rank_gap_ac")
    print(f"  Coef (rank_gap_ac): {m1.params['rank_gap_ac']:.4f}  "
          f"p={m1.pvalues['rank_gap_ac']:.4f}")
    odds = np.exp(m1.params["rank_gap_ac"])
    print(f"  Odds ratio per 1-rank improvement for A over C: {odds:.4f}")
    print(f"  Pseudo R²: {m1.prsquared:.4f}")

    # Multivariate — note: rank_gap_ac = rank_gap_ab + rank_gap_bc (perfect collinearity)
    # so we drop rank_gap_ac and use the two components separately
    m2 = smf.logit("transitive ~ rank_gap_ab + rank_gap_bc",
                   data=sub).fit(disp=False)
    print("\n  Model 2: transitive ~ rank_gap_ab + rank_gap_bc")
    print("  (rank_gap_ac omitted: it equals rank_gap_ab + rank_gap_bc exactly)")
    for var in ["rank_gap_ab", "rank_gap_bc"]:
        print(f"  {var}: coef={m2.params[var]:.4f}  p={m2.pvalues[var]:.4f}  "
              f"OR={np.exp(m2.params[var]):.4f}")
    print(f"  Pseudo R²: {m2.prsquared:.4f}")
    return m1, m2, sub


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("FIFA World Cup Transitivity — Statistical Analysis")
    print("Modern era: 1994–2022 | Group stage only | Draws = 0.5 pts")

    df, app = load_applicable()

    overall_rate = test_overall(app)
    year_tbl = test_by_year(app)
    gap_tbl, gap_sub = test_by_rank_gap(app)
    m1, m2, lr_sub = test_logistic(app)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Group-stage matches:       {len(df[df['is_transitivity_triple'].isin([0,1])])}")
    print(f"  Applicable triples (A>B, B>C): {len(app)}")
    print(f"  Overall transitivity rate: {100*overall_rate:.1f}%")
    print("  Run 04_visualizations.py to generate figures.")
