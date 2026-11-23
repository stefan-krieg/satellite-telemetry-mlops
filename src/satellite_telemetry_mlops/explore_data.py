import pandas as pd

df = pd.read_csv("data/segments.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(5))

# Check data types and timestamp parsing
print("\nUnique channels:", df["channel"].unique())
print("Train values:", df["train"].value_counts(dropna=False).to_dict())
print("Anomaly values:", df["anomaly"].value_counts(dropna=False).to_dict())
print("Sampling values:", df["sampling"].value_counts(dropna=False).to_dict())

# Parse timestamps
df["ts"] = pd.to_datetime(df["timestamp"], format="mixed", dayfirst=True)
df = df.sort_values(["channel", "ts"]).reset_index(drop=True)

# Check segment sizes
seg_sizes = (
    df.groupby(["channel", "segment", "train", "sampling", "anomaly"])
    .size()
    .reset_index(name="len")
)
print("\nTotal segments:", len(seg_sizes))
print("Segment length summary:\n", seg_sizes["len"].describe())
print(
    "Segments shorter than 50:", (seg_sizes["len"] < 50).sum(), "out of", len(seg_sizes)
)
print("Distribution of segment lengths:\n", seg_sizes["len"].value_counts().head(10))

# Check time gaps within each channel and within segments
df["dt"] = df.groupby("channel")["ts"].diff().dt.total_seconds()
df["seg_change"] = df["segment"] != df.groupby("channel")["segment"].shift(1)
df["samp_change"] = df["sampling"] != df.groupby("channel")["sampling"].shift(1)

print("\nTime gaps (dt) within the SAME segment:")
same_seg = df[~df["seg_change"] & df["dt"].notna()]
print(same_seg.groupby("sampling")["dt"].describe())
print(
    "Gaps > sampling within same segment:",
    (same_seg["dt"] > same_seg["sampling"] * 1.5).sum(),
)

print("\nTime gaps (dt) BETWEEN segments (where seg_change is True and dt is notna):")
diff_seg = df[df["seg_change"] & df["dt"].notna()]
print(diff_seg["dt"].describe())
print(
    "Exact matches (dt == sampling) across segment boundaries:",
    (diff_seg["dt"] == diff_seg["sampling"]).sum(),
    "out of",
    len(diff_seg),
)
print("Gaps > 5s across segment boundaries:", (diff_seg["dt"] > 5).sum())

# Let's inspect dt values within segments and across segments in detail
print("Top dt values within same segment (sampling=1):")
print(same_seg[same_seg["sampling"] == 1]["dt"].value_counts().head(15))

print("\nTop dt values within same segment (sampling=5):")
print(same_seg[same_seg["sampling"] == 5]["dt"].value_counts().head(15))

print("\nTop dt values BETWEEN segments:")
print(diff_seg["dt"].value_counts().head(15))

# Check how continuous runs behave if we break ONLY on large gaps (e.g. > 60s) or on > sampling*1.5
for thresh_name, cond in [
    (
        "dt > sampling * 1.5 or samp_change",
        (df["dt"] > df["sampling"] * 1.5) | df["samp_change"],
    ),
    ("dt > 10s or samp_change", (df["dt"] > 10) | df["samp_change"]),
    ("dt > 60s or samp_change", (df["dt"] > 60) | df["samp_change"]),
    ("Only segment boundary", df["seg_change"]),
]:
    run_id = cond.fillna(True).cumsum()
    run_lens = df.groupby(run_id).size()
    print(f"\nRule: {thresh_name}")
    print(
        f"  Total runs: {len(run_lens)}, Runs < 50: {(run_lens < 50).sum()} ({(run_lens < 50).mean() * 100:.1f}%), Rows lost if dropped: {run_lens[run_lens < 50].sum()} / {len(df)} ({run_lens[run_lens < 50].sum() / len(df) * 100:.1f}%)"
    )

# Check timestamp overlap across channels and train/test splits across channels!
print("\nChannel time ranges and train/test breakdown:")
for ch, g in df.groupby("channel"):
    tr1 = g[g["train"] == 1]
    tr0 = g[g["train"] == 0]
    print(
        f"{ch}: total={len(g)}, train=1 ({len(tr1)} rows, {tr1['ts'].min()} -> {tr1['ts'].max()}), train=0 ({len(tr0)} rows, {tr0['ts'].min()} -> {tr0['ts'].max()})"
    )

# Check if timestamps across channels align at all
ts_per_ch = df.groupby("ts")["channel"].nunique()
print("\nNumber of channels sharing the exact same timestamp:")
print(ts_per_ch.value_counts())

# Verify if train or anomaly or sampling ever changes inside a single segment
seg_purity = df.groupby("segment")[
    ["train", "anomaly", "sampling", "channel"]
].nunique()
print("Max unique values per segment:")
print(seg_purity.max())

# Check how many runs < 50 exist if we break on: train_change | samp_change | (dt > 60)
cond_clean = (
    (df["channel"] != df["channel"].shift(1))
    | (df["train"] != df["train"].shift(1))
    | (df["sampling"] != df["sampling"].shift(1))
    | (df["dt"] > 60)
)
df["run_id"] = cond_clean.cumsum()
run_stats = df.groupby(["train", "run_id"]).size().reset_index(name="len")
for tr in [1, 0]:
    sub = run_stats[run_stats["train"] == tr]
    print(
        f"train={tr}: total runs={len(sub)}, runs < 50={(sub['len'] < 50).sum()}, rows in runs < 50={sub[sub['len'] < 50]['len'].sum()} / {sub['len'].sum()}"
    )
