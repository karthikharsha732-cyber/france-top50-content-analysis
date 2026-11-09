"""Shared data + KPI logic for the France Top 50 content-analysis project."""
import numpy as np
import pandas as pd

DUR_BINS = [0, 2.5, 3.5, 100]
DUR_LABELS = ["Short (<2.5 min)", "Medium (2.5-3.5 min)", "Long (>3.5 min)"]
SIZE_BINS = [0, 1, 6, 12, 18, 1000]
SIZE_LABELS = ["Single (1)", "EP-size (2-6)", "Mid (7-12)", "Large (13-18)", "Deluxe (19+)"]


def load_and_clean(path, report=False):
    raw = pd.read_csv(path)
    log = {"raw_rows": len(raw)}
    d = raw.copy()
    d["date"] = pd.to_datetime(d["date"], format="%d-%m-%Y")
    n = len(d); d = d.drop_duplicates(); log["exact_duplicate_rows_removed"] = n - len(d)
    bad = d["song"].isna() | (d["duration_ms"] <= 0)
    log["corrupt_rows_removed"] = int(bad.sum()); d = d[~bad]
    d["album_type"] = d["album_type"].str.strip().str.lower()
    d["album_type"] = d["album_type"].replace({"albums": "album", "singles": "single"})
    d["is_explicit"] = d["is_explicit"].astype(str).str.upper().map({"TRUE": True, "FALSE": False}).astype(bool)
    d["duration_min"] = d["duration_ms"] / 60000
    d["duration_bucket"] = pd.cut(d["duration_min"], DUR_BINS, labels=DUR_LABELS, right=False)
    d["size_band"] = pd.cut(d["total_tracks"], SIZE_BINS, labels=SIZE_LABELS)
    d["content"] = np.where(d["is_explicit"], "Explicit", "Clean")
    # popularity of 0 = score not returned by the API -> treat as missing
    log["zero_popularity_set_to_nan"] = int((d["popularity"] == 0).sum())
    d.loc[d["popularity"] == 0, "popularity"] = np.nan
    d["track_id"] = d["song"] + " | " + d["artist"]
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    d = d.sort_values(["date", "position"]).reset_index(drop=True)
    cnt = d.groupby("date").size()
    log["days"] = int(cnt.size)
    log["days_with_50"] = int((cnt == 50).sum())
    log["days_not_50"] = {str(k.date()): int(v) for k, v in cnt[cnt != 50].items()}
    full = pd.date_range(d.date.min(), d.date.max())
    log["missing_calendar_days"] = [str(x.date()) for x in full.difference(d.date.unique())]
    flag = d.groupby("track_id")["is_explicit"].nunique()
    log["tracks_with_inconsistent_explicit_flag"] = list(flag[flag > 1].index)
    log["clean_rows"] = len(d); log["unique_tracks"] = int(d.track_id.nunique())
    return (d, log) if report else d


def tier(d, n):
    return d[d["position"] <= n]


def weights(pos):
    """Rank weight: #1 = 1.0 ... #50 = 0.02 (linear)."""
    return (51 - pos) / 50


def acceptance_score(d, col):
    """Content Acceptance Score: rank-weighted share / raw share x 100 (>100 = over-indexes near the top)."""
    w = weights(d["position"])
    raw = d[col].value_counts(normalize=True)
    wt = w.groupby(d[col]).sum() / w.sum()
    return (wt / raw * 100).rename("score").dropna()


def kpis(d):
    exp = d["is_explicit"].mean()
    alb = d[d.album_type != "single"]
    small = alb[alb.total_tracks.between(2, 12)]["popularity"].mean()
    large = alb[alb.total_tracks >= 13]["popularity"].mean()
    cas = acceptance_score(d, "content")
    out = {
        "Explicit Content Share (%)": exp * 100,
        "Clean Content Dominance Ratio (clean:explicit)": (1 - exp) / exp if exp else np.nan,
        "Single vs Album Track Ratio (single:album)": (d.album_type == "single").sum() / max((d.album_type != "single").sum(), 1),
        "Average Song Duration (min)": d["duration_min"].mean(),
        "Album Size Impact Index": (large / small * 100) if small == small and large == large else np.nan,
        "Content Acceptance Score - Explicit": cas.get("Explicit", np.nan),
        "Content Acceptance Score - Clean": cas.get("Clean", np.nan),
    }
    return out
