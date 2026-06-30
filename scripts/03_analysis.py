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
    section("TEST 1 — Overall transitivity rate (binomial test)")
    n = len(app)
    k = app["transitive"].sum()
    rate = k / n
    result = stats.binomtest(k, n, p=0.5, alternative="two-sided")
    print(f"  Applicable triples : {n}")
    print(f"  Transitive         : {k}  ({100*rate:.1f}%)")
    print(f"  H0: p = 0.50 (chance)")
    print(f"  p-value (two-sided): {result.pvalue:.4f}")
    ci = result.proportion_ci(confidence_level=0.95)
    print(f"  95% CI             : [{ci.low:.3f}, {ci.high:.3f}]")
    if result.pvalue < 0.05:
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

    print("\n  Per-bucket binomial tests (H0: rate = 0.50, two-sided):")
    print(f"  {'Bucket':<22} {'k':>4} {'n':>4}  {'rate':>5}  {'95% CI':<16}  p-value")
    print("  " + "-" * 62)
    for bucket, row in tbl.iterrows():
        k, n = int(row["transitive"]), int(row["total"])
        res = stats.binomtest(k, n, p=0.5, alternative="two-sided")
        ci = res.proportion_ci(confidence_level=0.95)
        sig = " *" if res.pvalue < 0.05 else ""
        print(f"  {str(bucket):<22} {k:>4} {n:>4}  {row['rate']:>5.3f}  "
              f"[{ci.low:.3f}, {ci.high:.3f}]  {res.pvalue:.4f}{sig}")
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
