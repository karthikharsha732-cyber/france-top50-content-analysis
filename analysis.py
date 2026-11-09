"""Shared analysis logic for the France Top 50 content-compliance project.
Used by both the Streamlit dashboard (app.py) and the report builder (build_reports.py)."""
import numpy as np
import pandas as pd

DUR_BINS = [0, 2.5, 3.5, np.inf]
DUR_LABELS = ["Short (<2:30)", "Medium (2:30-3:30)", "Long (>3:30)"]
SIZE_BINS = [0, 1, 7, 12, 17, np.inf]
SIZE_LABELS = ["Single (1 track)", "EP / short (2-7)", "Standard (8-12)", "Full album (13-17)", "Deluxe / large (18+)"]
LARGE_ALBUM_CUTOFF = 10  # releases with > 10 tracks count as "large"


def load_clean(path):
    """Validate and prepare the raw file. Returns (clean_df, validation_log dict)."""
    raw = pd.read_csv(path)
    log = {"raw_rows": len(raw)}
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    # 1) one snapshot per day, exactly 50 ranks
    counts = df.groupby("date").size()
    log["days_with_data"] = int(counts.size)
    log["days_not_50"] = {str(k.date()): int(v) for k, v in counts[counts != 50].items()}
    df = df.drop_duplicates(["date", "position"], keep="first")
    log["rows_after_dedup"] = len(df)
    log["days_not_50_after"] = int((df.groupby("date").size() != 50).sum())
    full = pd.date_range(df.date.min(), df.date.max())
    log["calendar_days"] = len(full)
    log["missing_days"] = [str(d.date()) for d in full.difference(df.date.unique())]
    # 2) types / labels
    df["album_type"] = df["album_type"].astype(str).str.strip().str.title()
    log["album_types"] = df["album_type"].value_counts().to_dict()
    df["format_group"] = np.where(df.album_type == "Single", "Single", "Album track")
    # 3) missing values
    log["missing_song_rows"] = int(df.song.isna().sum())
    df["song"] = df["song"].fillna("(unknown title)")
    log["zero_duration_rows"] = int((df.duration_ms <= 0).sum())
    log["zero_popularity_rows"] = int((df.popularity <= 0).sum())
    df["duration_min"] = df.duration_ms.where(df.duration_ms > 0) / 60000
    df["popularity_valid"] = df.popularity.where(df.popularity > 0)
    # 4) explicit flag consistency
    df["is_explicit"] = df["is_explicit"].astype(bool)
    inc = df.groupby(["song", "artist"]).is_explicit.nunique()
    log["tracks_with_inconsistent_flag"] = int((inc > 1).sum())
    # majority-vote the flag for tracks that flip, so a track is one thing throughout
    maj = df.groupby(["song", "artist"]).is_explicit.agg(lambda s: s.mean() >= 0.5)
    key = list(zip(df.song, df.artist))
    df["is_explicit_raw"] = df["is_explicit"]
    df["is_explicit"] = pd.Series(key, index=df.index).map(maj).astype(bool)
    log["flag_rows_changed"] = int((df.is_explicit != df.is_explicit_raw).sum())
    df["content"] = np.where(df.is_explicit, "Explicit", "Clean")
    # derived
    df["duration_bucket"] = pd.cut(df.duration_min, DUR_BINS, labels=DUR_LABELS, right=False)
    df["size_bucket"] = pd.cut(df.total_tracks, SIZE_BINS, labels=SIZE_LABELS)
    df["rank_tier"] = np.select([df.position <= 10, df.position <= 25], ["Top 10", "Top 11-25"], "Top 26-50")
    df["month"] = df.date.dt.to_period("M").dt.to_timestamp()
    df = df.drop(columns=["is_explicit_raw"]).sort_values(["date", "position"]).reset_index(drop=True)
    return df, log


def tier(df, n):
    return df[df.position <= n]


def rank_weight(df):
    return (51 - df.position) / 50.0


def acceptance_score(df, col, value):
    """Content Acceptance Score: rank-weighted share / plain share x 100. >100 = over-represented at the top."""
    m = (df[col] == value).astype(float)
    w = rank_weight(df)
    plain = m.mean()
    if plain == 0:
        return np.nan
    return 100 * (np.average(m, weights=w) / plain)


def kpis(df):
    ex = df.is_explicit.mean()
    single = (df.format_group == "Single").sum()
    album = (df.format_group == "Album track").sum()
    big = df[df.total_tracks > LARGE_ALBUM_CUTOFF].popularity_valid.mean()
    small = df[df.total_tracks <= LARGE_ALBUM_CUTOFF].popularity_valid.mean()
    return {
        "explicit_share": ex * 100,
        "clean_dominance_ratio": (1 - ex) / ex if ex > 0 else np.nan,
        "single_album_ratio": single / album if album else np.nan,
        "avg_duration_min": df.duration_min.mean(),
        "album_size_index": 100 * big / small if small and small == small else np.nan,
        "acceptance_explicit": acceptance_score(df, "content", "Explicit"),
        "acceptance_clean": acceptance_score(df, "content", "Clean"),
        "acceptance_single": acceptance_score(df, "format_group", "Single"),
        "avg_popularity": df.popularity_valid.mean(),
    }


def explicit_by_position(df):
    return df.groupby("position").is_explicit.mean().mul(100)


def tier_table(df):
    rows = []
    for n in (10, 25, 50):
        t = tier(df, n)
        row = {"Tier": f"Top {n}", "Explicit %": t.is_explicit.mean() * 100,
               "Single %": (t.format_group == "Single").mean() * 100,
               "Avg duration (min)": t.duration_min.mean(),
               "Avg album size": t.total_tracks.mean(),
               "Avg popularity": t.popularity_valid.mean()}
        for lab in DUR_LABELS:
            row[lab + " %"] = (t.duration_bucket == lab).mean() * 100
        rows.append(row)
    return pd.DataFrame(rows).set_index("Tier")


def profiles(df, n=10, top=6):
    t = tier(df, n)
    p = (t.assign(profile=t.content + " | " + t.format_group + " | " + t.duration_bucket.astype(str))
         .profile.value_counts(normalize=True).mul(100).head(top).rename("Share of slots %"))
    return p.reset_index().rename(columns={"profile": "Content profile"})
