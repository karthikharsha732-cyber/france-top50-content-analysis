"""Streamlit dashboard - France Top 50: Audience Sensitivity, Content Compliance & Format Preference.
Run:  streamlit run app.py
"""
import numpy as np, pandas as pd, plotly.express as px, streamlit as st
from fr_top50 import load_and_clean, tier, acceptance_score, kpis, DUR_LABELS, SIZE_LABELS

st.set_page_config(page_title="France Top 50 - Content Intelligence", layout="wide")
COL = {"Explicit": "#C8102E", "Clean": "#1F4E9C", "single": "#E08A00", "album": "#2A7F62", "compilation": "#888888"}

@st.cache_data
def get_data(path):
    return load_and_clean(path, report=True)

st.sidebar.title("Filters")
src = st.sidebar.text_input("Data file", "Atlantic_France.csv")
up = st.sidebar.file_uploader("...or upload CSV", type="csv")
try:
    if up is not None:
        import tempfile, os
        tmp = os.path.join(tempfile.gettempdir(), "up.csv"); open(tmp, "wb").write(up.getvalue()); df, log = get_data(tmp)
    else:
        df, log = get_data(src)
except Exception as e:
    st.error(f"Could not load data: {e}"); st.stop()

dmin, dmax = df.date.min().date(), df.date.max().date()
rng = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(rng, tuple) and len(rng) == 2: d0, d1 = rng
else: d0, d1 = dmin, dmax
tier_n = st.sidebar.radio("Rank tier", [10, 25, 50], index=2, format_func=lambda x: f"Top {x}", horizontal=True)
exp_opt = st.sidebar.radio("Explicit content", ["All", "Explicit only", "Clean only"])
types = st.sidebar.multiselect("Album type", sorted(df.album_type.unique()), default=sorted(df.album_type.unique()))

f = df[(df.date.dt.date >= d0) & (df.date.dt.date <= d1) & (df.position <= tier_n) & df.album_type.isin(types)]
if exp_opt == "Explicit only": f = f[f.is_explicit]
elif exp_opt == "Clean only": f = f[~f.is_explicit]
if f.empty: st.warning("No rows match the current filters."); st.stop()

st.title("France Top 50 - Audience Sensitivity, Content Compliance & Format Preference")
st.caption(f"{len(f):,} chart slots | {f.date.nunique()} days | {f.track_id.nunique()} unique tracks | Top {tier_n} | {d0} to {d1}")

K = kpis(f)
c = st.columns(6)
c[0].metric("Explicit share", f"{K['Explicit Content Share (%)']:.1f}%")
c[1].metric("Clean:Explicit ratio", f"{K['Clean Content Dominance Ratio (clean:explicit)']:.2f}")
c[2].metric("Single:Album ratio", f"{K['Single vs Album Track Ratio (single:album)']:.2f}")
c[3].metric("Avg duration", f"{K['Average Song Duration (min)']:.2f} min")
v = K["Album Size Impact Index"]; c[4].metric("Album Size Impact Index", "n/a" if v != v else f"{v:.0f}", help="Mean popularity of 13+ track albums / 2-12 track albums x 100. Below 100 = larger albums dilute track popularity.")
v = K["Content Acceptance Score - Explicit"]; c[5].metric("Acceptance: Explicit", "n/a" if v != v else f"{v:.0f}", help="Rank-weighted share / raw share x 100. Above 100 = over-represented near the top.")

tabs = st.tabs(["Compliance summary", "Explicit vs clean", "Format", "Album structure", "Duration", "Rank-tier profiles", "Data quality"])

with tabs[0]:
    ex = f.is_explicit.mean() * 100
    cas = acceptance_score(f, "content"); sg = (f.album_type == "single").mean() * 100
    st.subheader("Content compliance summary")
    a, b = st.columns(2)
    a.markdown(f"- **{ex:.1f}%** of slots are explicit; **{100-ex:.1f}%** clean.\n- Mean popularity - explicit **{f[f.is_explicit].popularity.mean():.1f}** vs clean **{f[~f.is_explicit].popularity.mean():.1f}**.\n- Mean rank - explicit **{f[f.is_explicit].position.mean():.1f}** vs clean **{f[~f.is_explicit].position.mean():.1f}** (lower = better).")
    b.markdown(f"- **{sg:.1f}%** singles / **{100-sg:.1f}%** album tracks.\n- Average duration **{f.duration_min.mean():.2f} min**; median **{f.duration_min.median():.2f} min**.\n- Explicit acceptance score **{cas.get('Explicit', float('nan')):.0f}**, clean **{cas.get('Clean', float('nan')):.0f}** (100 = neutral).")
    wk = f.groupby("date").is_explicit.mean().mul(100).rolling(14, min_periods=1).mean().reset_index()
    st.plotly_chart(px.line(wk, x="date", y="is_explicit", title="Explicit share, 14-day rolling average (%)", labels={"is_explicit": "%"}).update_traces(line_color=COL["Explicit"]), width='stretch')

with tabs[1]:
    a, b = st.columns(2)
    bands = pd.cut(f.position, [0, 5, 10, 20, 30, 40, 50], labels=["1-5", "6-10", "11-20", "21-30", "31-40", "41-50"])
    eb = f.groupby(bands, observed=True).is_explicit.mean().mul(100).reset_index()
    a.plotly_chart(px.bar(eb, x="position", y="is_explicit", title="Explicit share by rank band (%)", labels={"position": "Rank band", "is_explicit": "%"}, color_discrete_sequence=[COL["Explicit"]]), width='stretch')
    b.plotly_chart(px.box(f.dropna(subset=["popularity"]), x="content", y="popularity", color="content", color_discrete_map=COL, title="Popularity: explicit vs clean"), width='stretch')
    m = f.groupby(["month", "content"]).size().unstack(fill_value=0)
    st.plotly_chart(px.bar(m.reset_index().melt("month"), x="month", y="value", color="content", color_discrete_map=COL, title="Monthly chart slots by content type", barmode="stack"), width='stretch')
    t = f.groupby("track_id").agg(days=("date", "nunique"), content=("content", "first"))
    st.plotly_chart(px.box(t, x="content", y="days", color="content", color_discrete_map=COL, title="Track longevity (days on chart)"), width='stretch')

with tabs[2]:
    a, b = st.columns(2)
    ft = f.album_type.value_counts().reset_index(); ft.columns = ["album_type", "slots"]
    a.plotly_chart(px.pie(ft, names="album_type", values="slots", hole=.45, color="album_type", color_discrete_map=COL, title="Format share of chart slots"), width='stretch')
    bands = pd.cut(f.position, [0, 5, 10, 20, 30, 40, 50], labels=["1-5", "6-10", "11-20", "21-30", "31-40", "41-50"])
    fb = f.groupby([bands, "album_type"], observed=True).size().unstack(fill_value=0)
    fb = fb.div(fb.sum(1), axis=0).mul(100).reset_index().melt("position")
    b.plotly_chart(px.bar(fb, x="position", y="value", color="album_type", color_discrete_map=COL, title="Format mix by rank band (%)", labels={"value": "%", "position": "Rank band"}), width='stretch')
    st.plotly_chart(px.box(f.dropna(subset=["popularity"]), x="album_type", y="popularity", color="album_type", color_discrete_map=COL, title="Popularity by album type"), width='stretch')

with tabs[3]:
    a, b = st.columns(2)
    ua = f.drop_duplicates("album_cover_url")
    a.plotly_chart(px.histogram(ua, x="total_tracks", nbins=40, title="Album size distribution (unique releases)"), width='stretch')
    ss = f.groupby("size_band", observed=True).agg(popularity=("popularity", "mean"), slots=("song", "size")).reset_index()
    b.plotly_chart(px.bar(ss, x="size_band", y="popularity", title="Mean popularity by album size band", range_y=[60, 85]), width='stretch')
    al = f[f.album_type == "album"].dropna(subset=["popularity"])
    if len(al):
        sc = al.groupby(["size_band", "content"], observed=True).popularity.mean().reset_index()
        st.plotly_chart(px.line(sc, x="size_band", y="popularity", color="content", markers=True, color_discrete_map=COL, title="Album tracks only: size effect split by explicit/clean (controls for content mix)"), width='stretch')
    g = f[f.album_type == "album"].groupby(["date", "album_cover_url"]).agg(k=("song", "size"), pop=("popularity", "mean"), pos=("position", "mean")).reset_index()
    if len(g):
        gg = g.groupby(pd.cut(g.k, [0, 1, 2, 4, 100], labels=["1", "2", "3-4", "5+"]), observed=True).agg(album_days=("k", "size"), mean_popularity=("pop", "mean"), mean_rank=("pos", "mean")).round(1)
        st.markdown("**Dilution vs concentration** - tracks charting simultaneously from the same album:"); st.dataframe(gg)

with tabs[4]:
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(f, x="duration_min", nbins=60, title="Song duration (minutes)"), width='stretch')
    db = f.groupby("duration_bucket", observed=True).agg(slots=("song", "size"), popularity=("popularity", "mean"), mean_rank=("position", "mean"), explicit_share=("is_explicit", "mean")).round(3)
    b.plotly_chart(px.bar(db.reset_index(), x="duration_bucket", y="popularity", title="Mean popularity by duration bucket", range_y=[70, 82]), width='stretch')
    st.dataframe(db)
    st.plotly_chart(px.density_heatmap(f.dropna(subset=["popularity"]), x="duration_min", y="position", nbinsx=40, nbinsy=25, range_x=[1, 6], title="Duration vs rank density"), width='stretch')

with tabs[5]:
    base = df[(df.date.dt.date >= d0) & (df.date.dt.date <= d1) & df.album_type.isin(types)]
    attrs = {"Explicit": base.is_explicit, "Single": base.album_type == "single", "Large album (13+)": base.total_tracks >= 13,
             "Long (>3.5 min)": base.duration_bucket.astype(str).str.startswith("Long"), "Short (<2.5 min)": base.duration_bucket.astype(str).str.startswith("Short")}
    dens = pd.DataFrame({f"Top {n}": {k: v[base.position <= n].mean() * 100 for k, v in attrs.items()} for n in (10, 25, 50)})
    st.plotly_chart(px.imshow(dens.round(1), text_auto=".0f", color_continuous_scale="Blues", title="Attribute density (% of slots) by rank tier", aspect="auto"), width='stretch')
    base = base.assign(profile=base.content + " / " + np.where(base.album_type == "single", "Single", "Album") + " / " + base.duration_bucket.astype(str).str.split(" ").str[0])
    pf = pd.DataFrame({f"Top {n}": base[base.position <= n].profile.value_counts(normalize=True).mul(100).round(1) for n in (10, 25, 50)}).fillna(0).sort_values("Top 50", ascending=False).head(8)
    st.markdown("**Preferred content profiles (% of slots)**"); st.dataframe(pf)

with tabs[6]:
    st.json({k: v for k, v in log.items()})
    st.download_button("Download filtered data (CSV)", f.drop(columns=["month"]).to_csv(index=False), "france_top50_filtered.csv")
