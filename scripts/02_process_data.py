"""
Process raw data into clean group-stage match records with pre-tournament FIFA rankings.

Outputs:
  data/processed/group_stage_matches.csv   - one row per match, 1994-2022 group stage
  data/processed/teams_with_rankings.csv   - one row per (tournament, team) with pre-tournament rank
  data/processed/transitivity_triples.csv  - one row per (A,B,C) transitivity triple
"""

import csv
import pathlib
from datetime import date, datetime
from itertools import permutations

RAW = pathlib.Path(__file__).parent.parent / "data" / "raw"
PROC = pathlib.Path(__file__).parent.parent / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

MODERN_YEARS = set(range(1994, 2023, 4))  # 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {path.name}")


# ---------------------------------------------------------------------------
# Step 1: Filter matches to 1994-2022 group stage
# ---------------------------------------------------------------------------

def load_group_stage_matches():
    rows = read_csv(RAW / "matches.csv")
    out = []
    for r in rows:
        year = int(r["tournament_id"].split("-")[1])
        if year not in MODERN_YEARS:
            continue
        if r["group_stage"] != "1":
            continue
        # Replayed matches (1930-era artifact) excluded
        if r["replay"] == "1":
            continue
        home_score = int(r["home_team_score"])
        away_score = int(r["away_team_score"])
        is_draw = r["draw"] == "1"
        home_win = r["home_team_win"] == "1"
        away_win = r["away_team_win"] == "1"
        # points-style outcome (win=1, draw=0.5, loss=0)
        home_pts = 1.0 if home_win else (0.5 if is_draw else 0.0)
        away_pts = 1.0 if away_win else (0.5 if is_draw else 0.0)
        out.append({
            "tournament_id": r["tournament_id"],
            "year": year,
            "match_date": r["match_date"],
            "group_name": r["group_name"],
            "home_team": r["home_team_name"],
            "home_code": r["home_team_code"],
            "away_team": r["away_team_name"],
            "away_code": r["away_team_code"],
            "home_score": home_score,
            "away_score": away_score,
            "result": r["result"],
            "home_pts": home_pts,
            "away_pts": away_pts,
        })
    print(f"  Group-stage matches 1994-2022: {len(out)}")
    return out


# ---------------------------------------------------------------------------
# Step 2: Build pre-tournament FIFA rankings snapshot
# ---------------------------------------------------------------------------

def load_rankings():
    rows = read_csv(RAW / "fifa_rankings_historical.csv")
    # Parse dates; sort ascending
    parsed = []
    for r in rows:
        try:
            d = datetime.strptime(r["date"], "%Y-%m-%d").date()
        except ValueError:
            continue
        try:
            pts = float(r["total_points"])
        except ValueError:
            continue
        parsed.append({
            "date": d,
            "team": r["team"],
            "team_short": r["team_short"],
            "total_points": pts,
        })
    parsed.sort(key=lambda x: x["date"])
    return parsed


def get_pretournament_rankings(ranking_rows, cutoff_date):
    """
    Return dict {team_name -> (rank, points)} using the most recent
    ranking snapshot whose date is <= cutoff_date.

    FIFA publishes rankings for all active teams in each release; we find
    the latest release date on or before cutoff_date, then rank teams by
    points descending within that snapshot.
    """
    # Find all unique dates <= cutoff_date
    valid_dates = sorted(
        {r["date"] for r in ranking_rows if r["date"] <= cutoff_date},
        reverse=True,
    )
    if not valid_dates:
        return {}
    snapshot_date = valid_dates[0]
    snapshot = [r for r in ranking_rows if r["date"] == snapshot_date]
    # Rank by points descending
    snapshot.sort(key=lambda x: x["total_points"], reverse=True)
    result = {}
    for rank, r in enumerate(snapshot, start=1):
        result[r["team"]] = {
            "rank": rank,
            "points": r["total_points"],
            "team_short": r["team_short"],
            "snapshot_date": str(snapshot_date),
        }
        # Also index by short code for fuzzy matching
        result[r["team_short"]] = result[r["team"]]
    return result


# Tournament start dates (used as ranking cutoff)
TOURNAMENT_START = {
    1994: date(1994, 6, 17),
    1998: date(1998, 6, 10),
    2002: date(2002, 5, 31),
    2006: date(2006, 6, 9),
    2010: date(2010, 6, 11),
    2014: date(2014, 6, 12),
    2018: date(2018, 6, 14),
    2022: date(2022, 11, 20),
}

# Manual name mappings: FIFA ranking name -> jfjelstul team name
TEAM_NAME_MAP = {
    "Korea Republic": "South Korea",
    "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast",
    "Bosnia Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "China PR": "China",
    "Congo DR": "DR Congo",
    "Korea DPR": "North Korea",
    "Chinese Taipei": "Taiwan",
    "Trinidad & Tobago": "Trinidad and Tobago",
}


def resolve_team(name, code, ranking_dict):
    """Try to look up a team in the ranking dict by name or code."""
    if name in ranking_dict:
        return ranking_dict[name]
    mapped = TEAM_NAME_MAP.get(name)
    if mapped and mapped in ranking_dict:
        return ranking_dict[mapped]
    if code in ranking_dict:
        return ranking_dict[code]
    return None


def build_team_rankings(matches, ranking_rows):
    """For each (tournament, team) pair, attach the pre-tournament ranking."""
    pre_rankings = {}
    for year, cutoff in TOURNAMENT_START.items():
        pre_rankings[year] = get_pretournament_rankings(ranking_rows, cutoff)
        snap = list({v["snapshot_date"] for v in pre_rankings[year].values() if isinstance(v, dict)})
        print(f"    {year}: snapshot date = {snap[0] if snap else 'N/A'}")

    seen = {}  # (year, team) -> ranking info
    missing = set()
    for m in matches:
        year = m["year"]
        rdict = pre_rankings[year]
        for team, code in [(m["home_team"], m["home_code"]), (m["away_team"], m["away_code"])]:
            key = (year, team)
            if key in seen:
                continue
            info = resolve_team(team, code, rdict)
            if info is None:
                missing.add((year, team, code))
                seen[key] = {"rank": None, "points": None, "snapshot_date": None}
            else:
                seen[key] = info

    if missing:
        print(f"  WARNING: {len(missing)} team(s) not found in ranking data:")
        for y, t, c in sorted(missing):
            print(f"    {y} {t} ({c})")

    rows = []
    for (year, team), info in sorted(seen.items()):
        rows.append({
            "year": year,
            "team": team,
            "pre_tournament_rank": info["rank"],
            "pre_tournament_points": info["points"],
            "ranking_snapshot_date": info["snapshot_date"],
        })
    return rows, pre_rankings


# ---------------------------------------------------------------------------
# Step 3: Build transitivity triples
# ---------------------------------------------------------------------------

def match_outcome(team_a, team_b, match_list):
    """
    Return points earned by team_a when playing team_b:
      1.0 = win, 0.5 = draw, 0.0 = loss, None = no match found
    """
    for m in match_list:
        if m["home_team"] == team_a and m["away_team"] == team_b:
            return m["home_pts"]
        if m["home_team"] == team_b and m["away_team"] == team_a:
            return m["away_pts"]
    return None


def build_transitivity_triples(matches, pre_rankings):
    """
    For each group in each tournament, enumerate ordered triples (A, B, C)
    where A has a better-or-equal result vs B than C does.
    Transitivity holds if A also beats C at least as well.
    """
    # Group matches by (year, group)
    groups = {}
    for m in matches:
        key = (m["year"], m["group_name"])
        groups.setdefault(key, []).append(m)

    triples = []
    for (year, group_name), group_matches in sorted(groups.items()):
        teams = sorted(
            {m["home_team"] for m in group_matches}
            | {m["away_team"] for m in group_matches}
        )
        rdict = pre_rankings[year]

        # All ordered triples (A, B, C) where A != B != C != A
        for a, b, c in permutations(teams, 3):
            ab = match_outcome(a, b, group_matches)
            bc = match_outcome(b, c, group_matches)
            ac = match_outcome(a, c, group_matches)

            if ab is None or bc is None or ac is None:
                continue  # skip if any match wasn't played

            # Transitivity condition: A beats B AND B beats C → A beats C
            # We use a strict definition: A has pts > 0.5 vs B, B has pts > 0.5 vs C
            a_beats_b = ab > 0.5   # strict win
            b_beats_c = bc > 0.5

            # Check transitivity
            transitive = None
            if a_beats_b and b_beats_c:
                transitive = 1 if ac > 0.5 else 0

            # Rankings
            ra = resolve_team(a, "", rdict)
            rb = resolve_team(b, "", rdict)
            rc = resolve_team(c, "", rdict)

            rank_a = ra["rank"] if ra else None
            rank_b = rb["rank"] if rb else None
            rank_c = rc["rank"] if rc else None

            triples.append({
                "year": year,
                "group_name": group_name,
                "team_a": a,
                "team_b": b,
                "team_c": c,
                "ab_pts_for_a": ab,
                "bc_pts_for_b": bc,
                "ac_pts_for_a": ac,
                "a_beats_b": int(a_beats_b),
                "b_beats_c": int(b_beats_c),
                "is_transitivity_triple": int(a_beats_b and b_beats_c),
                "transitive": transitive,  # 1=yes, 0=no, None=not applicable
                "rank_a": rank_a,
                "rank_b": rank_b,
                "rank_c": rank_c,
                "rank_gap_ab": (rank_a - rank_b) if (rank_a and rank_b) else None,
                "rank_gap_bc": (rank_b - rank_c) if (rank_b and rank_c) else None,
                "rank_gap_ac": (rank_a - rank_c) if (rank_a and rank_c) else None,
            })

    return triples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Step 1: Loading group-stage matches...")
    matches = load_group_stage_matches()

    print("Step 2: Loading and snapshotting FIFA rankings...")
    ranking_rows = load_rankings()
    team_ranking_rows, pre_rankings = build_team_rankings(matches, ranking_rows)

    print("Step 3: Building transitivity triples...")
    triples = build_transitivity_triples(matches, pre_rankings)

    applicable = [t for t in triples if t["is_transitivity_triple"] == 1]
    transitive_count = sum(1 for t in applicable if t["transitive"] == 1)
    print(f"  Total ordered triples evaluated: {len(triples)}")
    print(f"  Transitivity triples (A beats B, B beats C): {len(applicable)}")
    print(f"  Of those, A also beats C: {transitive_count} ({100*transitive_count/len(applicable):.1f}%)")

    print("Writing outputs...")
    write_csv(
        PROC / "group_stage_matches.csv",
        matches,
        ["tournament_id", "year", "match_date", "group_name",
         "home_team", "home_code", "away_team", "away_code",
         "home_score", "away_score", "result", "home_pts", "away_pts"],
    )
    write_csv(
        PROC / "teams_with_rankings.csv",
        team_ranking_rows,
        ["year", "team", "pre_tournament_rank", "pre_tournament_points", "ranking_snapshot_date"],
    )
    write_csv(
        PROC / "transitivity_triples.csv",
        triples,
        ["year", "group_name", "team_a", "team_b", "team_c",
         "ab_pts_for_a", "bc_pts_for_b", "ac_pts_for_a",
         "a_beats_b", "b_beats_c", "is_transitivity_triple", "transitive",
         "rank_a", "rank_b", "rank_c",
         "rank_gap_ab", "rank_gap_bc", "rank_gap_ac"],
    )
    print("Done.")
