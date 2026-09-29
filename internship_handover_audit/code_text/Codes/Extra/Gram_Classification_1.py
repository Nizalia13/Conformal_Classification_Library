
# ORIGINAL NOTEBOOK CELL 1 (zero-based)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# ORIGINAL NOTEBOOK CELL 3 (zero-based)
SCORES_PATH = "../Data/host_type_all_preds.csv"
METADATA_PATH = "../Data/metadata.csv"

print("Loading scores...")
t0 = time.time()
scores_df = pd.read_csv(SCORES_PATH, index_col="proteinID")
print(f"  scores shape : {scores_df.shape}  ({time.time()-t0:.1f}s)")

print("Loading metadata...")
t0 = time.time()
meta_df = pd.read_csv(METADATA_PATH, index_col="proteinID")
print(f"  metadata shape : {meta_df.shape}  ({time.time()-t0:.1f}s)")

meta_df = meta_df[["accession", "host_type", "split"]]

# ORIGINAL NOTEBOOK CELL 5 (zero-based)
# scores file
print(scores_df.head())

# ORIGINAL NOTEBOOK CELL 6 (zero-based)
# metadata file
print(meta_df.head(10))

# ORIGINAL NOTEBOOK CELL 8 (zero-based)
combined = scores_df.join(meta_df, how="inner")
score_cols = [c for c in scores_df.columns if c.startswith("host_type_")]

t0 = time.time()
phage_scores = combined.groupby("accession")[score_cols].mean()
phage_meta = combined.groupby("accession")[["host_type", "split"]].first()
phage_df = phage_scores.join(phage_meta)

print(f"Phage-level shape: {phage_df.shape}  ({time.time()-t0:.1f}s)")
print("\nGram-type distribution:")
print(phage_df["host_type"].value_counts())
print("\nSplit distribution:")
print(phage_df["split"].value_counts().sort_index())

# ORIGINAL NOTEBOOK CELL 10 (zero-based)
nan_rates = phage_df[score_cols].isna().mean().sort_values(ascending=False)
NAN_THRESHOLD = 0.50
valid_cols = nan_rates[nan_rates <= NAN_THRESHOLD].index.tolist()
dropped = nan_rates[nan_rates  > NAN_THRESHOLD].index.tolist()
K = len(valid_cols)
print(f"Columns kept   : {K} / {len(score_cols)}")
print(f"Columns dropped: {len(dropped)}")
print(f"  Dropped: {dropped}")
print(f"\nFinal K = {K}")

# ORIGINAL NOTEBOOK CELL 12 (zero-based)
train_df = phage_df[phage_df["split"] != 0].copy()
test_df = phage_df[phage_df["split"] == 0].copy()

train_df = train_df[train_df["host_type"].isin(["gram-neg", "gram-pos"])]
test_df = test_df [test_df ["host_type"].isin(["gram-neg", "gram-pos"])]

print(f"Train phages : {len(train_df)}")
print(f"Test  phages : {len(test_df)}")

# ORIGINAL NOTEBOOK CELL 14 (zero-based)
train_medians = train_df[valid_cols].median()
safe_impute_values = train_medians.copy()
safe_impute_values[safe_impute_values > 0.6] = 0.5

train_df[valid_cols] = train_df[valid_cols].fillna(safe_impute_values)
test_df [valid_cols] = test_df [valid_cols].fillna(safe_impute_values)
train_df[valid_cols] = train_df[valid_cols].fillna(0.5)
test_df [valid_cols] = test_df [valid_cols].fillna(0.5)

print(f"\nResidual NaNs — train: {train_df[valid_cols].isna().sum().sum()}")
print(f"Residual NaNs — test : {test_df[valid_cols].isna().sum().sum()}")

# ORIGINAL NOTEBOOK CELL 16 (zero-based)
print("Mean score per gram type (averaged across all K columns):")
for gt in ["gram-neg", "gram-pos"]:
    mask = train_df["host_type"] == gt
    mean_score = train_df.loc[mask, valid_cols].mean().mean()
    median_score = train_df.loc[mask, valid_cols].median().median()
    print(f"  {gt}: mean={mean_score:.4f}, median={median_score:.4f}, n={mask.sum()}")

print("\nPer-column mean scores (gram-neg vs gram-pos):")
col_means = pd.DataFrame({
    "gram-neg": train_df[train_df["host_type"]=="gram-neg"][valid_cols].mean(),
    "gram-pos": train_df[train_df["host_type"]=="gram-pos"][valid_cols].mean(),
})

col_means["gram-neg > gram-pos"] = col_means["gram-neg"] > col_means["gram-pos"]

print(col_means.to_string())
print(f"\nColumns where gram-neg scores higher: {col_means['gram-neg > gram-pos'].sum()} / {len(valid_cols)}")

# ORIGINAL NOTEBOOK CELL 19 (zero-based)
train_raw = train_df[valid_cols].values
train_labels = train_df["host_type"].values

neg_mask = train_labels == "gram-neg"
pos_mask = train_labels == "gram-pos"

test_raw     = test_df[valid_cols].values
test_labels  = test_df["host_type"].values

# ORIGINAL NOTEBOOK CELL 21 (zero-based)
train_nc = np.zeros_like(train_raw)
train_nc[neg_mask] = train_raw[neg_mask]         
train_nc[pos_mask] = 1.0 - train_raw[pos_mask]   

# Collapsed 1D version
train_nc_1d = train_nc.mean(axis=1, keepdims=True)

print(f"Training data shape: {train_nc.shape}")
print(f"Overall mean : {train_nc.mean():.4f} ")
print(f"Scores range : [{train_nc.min():.4f}, {train_nc.max():.4f}]")
print(f"\n1D collapsed mean: {train_nc_1d.mean():.4f}")

# ORIGINAL NOTEBOOK CELL 23 (zero-based)
def split_data(scores, split_ratio=0.5):
    n   = len(scores)
    idx = np.random.permutation(n)
    cut = int(n * split_ratio)
    return scores[idx[:cut]], scores[idx[cut:]]

# ORIGINAL NOTEBOOK CELL 25 (zero-based)
def sample_positive_sphere(M, K):
    V = np.abs(np.random.randn(M, K))
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

def radial_build_envelope(aligned_scores, alpha, M=200, delta_deg=30):
    S1, S2 = split_data(aligned_scores)
    K = S1.shape[1]
    U = sample_positive_sphere(M, K)
    mags = np.linalg.norm(S1, axis=1)
    dirs = S1 / (mags[:, None] + 1e-12)
    cos_thresh = np.cos(np.radians(delta_deg))
    q_tilde = np.zeros(M)
    for m in range(M):
        mask = (dirs @ U[m]) >= cos_thresh
        q_tilde[m] = np.quantile(mags[mask], 1 - alpha) if mask.sum() > 5 else np.quantile(mags, 1 - alpha)
    boundary = U * q_tilde[:, None]
    tau_scores = (S2[:, None, :] / (boundary[None, :, :] + 1e-12)).max(axis=2).min(axis=1)
    n2 = len(tau_scores)
    t_hat = np.sort(tau_scores)[np.clip(int(np.ceil((n2 + 1) * (1 - alpha))) - 1, 0, n2 - 1)]
    return {"U": U, "q_tilde": q_tilde, "t_hat": t_hat, "method": "radial"}

def radial_is_in_region(scores, envelope):
    U, q_tilde, t_hat = envelope["U"], envelope["q_tilde"], envelope["t_hat"]
    boundary = U * (q_tilde * t_hat)[:, None]
    return np.any(np.all(scores[:, None, :] <= boundary[None, :, :], axis=2), axis=1)

# ORIGINAL NOTEBOOK CELL 27 (zero-based)
def strip_shape_discovery(S1, alpha, M):
    """For each (j, bin m), compute (1-alpha) quantile of all other dims i."""
    N1, K     = S1.shape
    bin_edges = np.zeros((K, M + 1))
    for j in range(K):
        bin_edges[j] = np.linspace(S1[:, j].min(), S1[:, j].max(), M + 1)
    limits = np.zeros((K, K, M))
    for j in range(K):
        for m in range(M):
            lo, hi   = bin_edges[j, m], bin_edges[j, m + 1]
            in_strip = (S1[:, j] >= lo) & (S1[:, j] < hi)
            for i in range(K):
                if i == j: continue
                if in_strip.sum() < 3:
                    limits[j, i, m] = np.quantile(S1[:, i], 1 - alpha)
                else:
                    limits[j, i, m] = np.quantile(S1[in_strip, i], 1 - alpha)
    for j in range(K):
        for i in range(K):
            if i == j: continue
            for m in range(M - 2, -1, -1):
                limits[j, i, m] = max(limits[j, i, m], limits[j, i, m + 1])
    return bin_edges, limits

def get_bin_indices(scores, bin_edges):
    """For each point and dimension, find which bin it falls in."""
    N, K    = scores.shape
    M       = bin_edges.shape[1] - 1
    indices = np.zeros((N, K), dtype=int)
    for j in range(K):
        raw           = np.digitize(scores[:, j], bin_edges[j]) - 1
        indices[:, j] = np.clip(raw, 0, M - 1)
    return indices

def strip_size_scaling(S2, bin_edges, limits, alpha):
    """Find t_hat for the strip envelope."""
    N2, K   = S2.shape
    bin_idx = get_bin_indices(S2, bin_edges)
    ratios  = np.zeros((N2, K, K))
    for j in range(K):
        for i in range(K):
            if i == j: continue
            lims            = limits[j, i, bin_idx[:, j]]
            ratios[:, j, i] = S2[:, i] / (lims + 1e-12)
    tau_scores = ratios.max(axis=(1, 2))
    n2  = N2
    idx = int(np.ceil((n2 + 1) * (1 - alpha))) - 1
    return np.sort(tau_scores)[np.clip(idx, 0, n2 - 1)]

def strip_is_in_region(scores, envelope):
    """True if score vector is inside the scaled strip envelope."""
    bin_edges, limits, t_hat = envelope["bin_edges"], envelope["limits"], envelope["t_hat"]
    N, K          = scores.shape
    bin_idx       = get_bin_indices(scores, bin_edges)
    scaled_limits = limits * t_hat
    inside        = np.ones(N, dtype=bool)
    for j in range(K):
        for i in range(K):
            if i == j: continue
            inside &= (scores[:, i] <= scaled_limits[j, i, bin_idx[:, j]])
    return inside

def strip_build_envelope(aligned_scores, alpha, M=20):
    """Full two-stage calibration for the strip method."""
    S1, S2            = split_data(aligned_scores)
    bin_edges, limits = strip_shape_discovery(S1, alpha, M)
    t_hat             = strip_size_scaling(S2, bin_edges, limits, alpha)
    return {"bin_edges": bin_edges, "limits": limits, "t_hat": t_hat, "method": "strip"}

# ORIGINAL NOTEBOOK CELL 29 (zero-based)
def collapsed_build_envelope(aligned_scores_1d, alpha):
    """
    1D conformal envelope: interval [0, t_hat * q_tilde].
    aligned_scores_1d: (N, 1) array of row-mean scalars from the aligned K-dim matrix.
    """
    S1, S2     = split_data(aligned_scores_1d)
    q_tilde    = np.quantile(S1, 1 - alpha)
    tau_scores = (S2 / (q_tilde + 1e-12)).ravel()
    n2         = len(tau_scores)
    idx        = int(np.ceil((n2 + 1) * (1 - alpha))) - 1
    t_hat      = np.sort(tau_scores)[np.clip(idx, 0, n2 - 1)]
    return {"q_tilde": q_tilde, "t_hat": t_hat, "method": "collapsed"}

def collapsed_is_in_region(scores_1d, envelope):
    """True if scalar score is inside the 1D envelope [0, t_hat * q_tilde]."""
    threshold = envelope["q_tilde"] * envelope["t_hat"]
    return (scores_1d.ravel() <= threshold)

# ORIGINAL NOTEBOOK CELL 31 (zero-based)
def is_in_region(scores, envelope):
    if envelope["method"] == "radial":
        return radial_is_in_region(scores, envelope)
    elif envelope["method"] == "strip":
        return strip_is_in_region(scores, envelope)
    else:
        return collapsed_is_in_region(scores, envelope)

# ORIGINAL NOTEBOOK CELL 33 (zero-based)
np.random.seed(42)
alpha = 0.1

print("Building radial envelope on NC scores...")
t0 = time.time()
radial_env = radial_build_envelope(train_nc, alpha, M=200, delta_deg=30)
print(f"  Done in {time.time()-t0:.1f}s  |  t_hat = {radial_env['t_hat']:.4f}")

print("\nBuilding strip envelope on NC scores...")
t0  = time.time()
strip_env = strip_build_envelope(train_nc, alpha, M=20)
print(f"  Done in {time.time()-t0:.1f}s  |  t_hat = {strip_env['t_hat']:.4f}")

print("\nBuilding collapsed 1D envelope on NC scores...")
t0  = time.time()
collapsed_env = collapsed_build_envelope(train_nc_1d, alpha)
print(f"  Done in {time.time()-t0:.1f}s  |  threshold = {collapsed_env['q_tilde'] * collapsed_env['t_hat']:.4f}")

# ORIGINAL NOTEBOOK CELL 35 (zero-based)
def predict_gram_type_nc(test_raw, envelope):
    """
    For each test phage, compute NC under each hypothesis and check against the envelope.

    NC under gram-neg hypothesis = raw scores        (low if truly gram-neg)
    NC under gram-pos hypothesis = 1 - raw scores    (low if truly gram-pos)

    A hypothesis is included in the prediction set if its NC vector
    falls inside the envelope (i.e. is sufficiently unsurprising).
    """
    nc_gramneg = test_raw           # (N_test, K) — gram-neg NC
    nc_grampos = 1.0 - test_raw     # (N_test, K) — gram-pos NC

    if envelope["method"] == "collapsed":
        nc_gramneg = nc_gramneg.mean(axis=1, keepdims=True)
        nc_grampos = nc_grampos.mean(axis=1, keepdims=True)

    in_gramneg = is_in_region(nc_gramneg, envelope)   # (N_test,) bool
    in_grampos = is_in_region(nc_grampos, envelope)   # (N_test,) bool

    results = []
    for i in range(len(test_raw)):
        pred_set = []
        if in_gramneg[i]: pred_set.append("gram-neg")
        if in_grampos[i]: pred_set.append("gram-pos")
        results.append({
            "in_gramneg"     : bool(in_gramneg[i]),
            "in_grampos"     : bool(in_grampos[i]),
            "prediction_set" : pred_set
        })
    return results

# ORIGINAL NOTEBOOK CELL 36 (zero-based)
radial_results    = predict_gram_type_nc(test_raw, radial_env)
strip_results     = predict_gram_type_nc(test_raw, strip_env)
collapsed_results = predict_gram_type_nc(test_raw, collapsed_env)

# ORIGINAL NOTEBOOK CELL 38 (zero-based)
def evaluate(results, true_labels, method_name, alpha):
    n = len(results)
    set_sizes = [len(r["prediction_set"]) for r in results]
    covered = sum(true_labels[i] in r["prediction_set"] for i, r in enumerate(results))
    singletons = [(r, true_labels[i]) for i, r in enumerate(results) if len(r["prediction_set"]) == 1]
    correct_singletons = sum(r["prediction_set"][0] == lbl for r, lbl in singletons)
    empty = sum(s == 0 for s in set_sizes)
    size2 = sum(s == 2 for s in set_sizes)

    print(f"\n{'='*58}")
    print(f" {method_name.upper()} — {n} test phages | alpha={alpha}")
    print(f"{'='*58}")
    print(f" Coverage : {covered/n:.3f}  (target >= {1-alpha:.2f})")
    print(f" Avg prediction set size: {np.mean(set_sizes):.3f}")
    print(f" Empty sets  (size 0) : {empty}  ({empty/n*100:.1f}%)")
    print(f" Singleton   (size 1) : {len(singletons)}  ({len(singletons)/n*100:.1f}%)")
    print(f" Uncertain   (size 2) : {size2}  ({size2/n*100:.1f}%)")
    if singletons:
        print(f" Singleton accuracy : {correct_singletons/len(singletons):.3f}")
    else:
        print(" Singleton accuracy : N/A")
    print(f"\n Per-gram-type coverage:")
    for gt in ["gram-neg", "gram-pos"]:
        idx_gt = [i for i, lbl in enumerate(true_labels) if lbl == gt]
        cov_gt = sum(true_labels[i] in results[i]["prediction_set"] for i in idx_gt)
        print(f"   {gt:<12}: {cov_gt/len(idx_gt):.3f}  (n={len(idx_gt)})")

    return {
        "coverage" : covered / n,
        "avg_set_size" : np.mean(set_sizes),
        "empty_rate" : empty / n,
        "singleton_rate" : len(singletons) / n,
        "size2_rate" : size2 / n,
        "singleton_accuracy" : correct_singletons / len(singletons) if singletons else None,
        "set_sizes" : set_sizes,
        "per_type_coverage"  : {
            gt: sum(true_labels[i] in results[i]["prediction_set"]
                    for i in [j for j, lbl in enumerate(true_labels) if lbl == gt])
               / max(1, sum(lbl == gt for lbl in true_labels))
            for gt in ["gram-neg", "gram-pos"]
        },
    }

# ORIGINAL NOTEBOOK CELL 39 (zero-based)
radial_metrics    = evaluate(radial_results,    test_labels, "Radial (NC)",       alpha)
strip_metrics     = evaluate(strip_results,     test_labels, "Strip (NC)",        alpha)
collapsed_metrics = evaluate(collapsed_results, test_labels, "Collapsed 1D (NC)", alpha)

# ORIGINAL NOTEBOOK CELL 40 (zero-based)
methods_names = ["Radial", "Strip", "Collapsed 1D"]
metrics       = [radial_metrics, strip_metrics, collapsed_metrics]

summary = pd.DataFrame({
    "Method" : methods_names,
    "Coverage" : [f"{m['coverage']:.3f}" for m in metrics],
    "Target" : [f">= {1-alpha:.2f}"] * 3,
    "Avg set size" : [f"{m['avg_set_size']:.3f}" for m in metrics],
    "Empty %" : [f"{m['empty_rate']*100:.1f}" for m in metrics],
    "Size-2 %" : [f"{m['size2_rate']*100:.1f}" for m in metrics],
    "Singleton acc." : [f"{m['singleton_accuracy']:.3f}" if m['singleton_accuracy'] else "N/A"
                        for m in metrics],
})
print(summary.to_string(index=False))

# ORIGINAL NOTEBOOK CELL 42 (zero-based)
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
colors    = ["steelblue", "darkorange", "mediumseagreen"]

# Plot 1: Set size distribution
ax      = axes[0]
x       = np.array([0, 1, 2])
offsets = [-0.25, 0.0, 0.25]
for name, m, c, offset in zip(methods_names, metrics, colors, offsets):
    sizes  = m["set_sizes"]
    counts = [sizes.count(s) / len(sizes) for s in [0, 1, 2]]
    ax.bar(x + offset, counts, width=0.23, label=name, color=c, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(["Empty (0)", "Singleton (1)", "Uncertain (2)"])
ax.set_title("Set Size Distribution")
ax.set_ylabel("Fraction of test phages")
ax.legend(fontsize=8)

# Plot 2: Overall coverage
ax   = axes[1]
covs = [m["coverage"] for m in metrics]
bars = ax.bar(methods_names, covs, color=colors, alpha=0.8)
ax.axhline(1 - alpha, color="red", linestyle="--", label=f"Target ({1-alpha:.2f})")
ax.set_ylim(0, 1.05)
ax.set_title("Overall Coverage")
ax.set_ylabel("Coverage")
ax.legend(fontsize=8)
for bar, val in zip(bars, covs):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}",
            ha="center", va="bottom", fontsize=9)

# Plot 3: Per-gram-type coverage
ax         = axes[2]
gram_types = ["gram-neg", "gram-pos"]
x          = np.arange(len(gram_types))
offsets    = [-0.25, 0.0, 0.25]
for name, m, c, offset in zip(methods_names, metrics, colors, offsets):
    vals = [m["per_type_coverage"][gt] for gt in gram_types]
    ax.bar(x + offset, vals, width=0.23, label=name, color=c, alpha=0.8)
ax.axhline(1 - alpha, color="red", linestyle="--", label=f"Target ({1-alpha:.2f})")
ax.set_xticks(x)
ax.set_xticklabels(gram_types)
ax.set_ylim(0, 1.05)
ax.set_title("Coverage by Gram Type")
ax.set_ylabel("Coverage")
ax.legend(fontsize=8)

# Plot 4: Singleton accuracy
ax      = axes[3]
accs    = [m["singleton_accuracy"] or 0 for m in metrics]
sing_ns = [int(m["singleton_rate"] * len(m["set_sizes"])) for m in metrics]
bars    = ax.bar(methods_names, accs, color=colors, alpha=0.8)
ax.set_ylim(0, 1.05)
ax.set_title("Singleton Accuracy")
ax.set_ylabel("Accuracy (when set size = 1)")
for bar, val, n_s in zip(bars, accs, sing_ns):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
            f"{val:.3f}\n(n={n_s})", ha="center", va="bottom", fontsize=9)

plt.suptitle("Gram-Type Classification",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# ORIGINAL NOTEBOOK CELL 43 (zero-based)
print("Distribution of mean raw scores per phage:")
mean_scores = test_df[valid_cols].mean(axis=1)
print(f"  Overall: mean={mean_scores.mean():.3f}, std={mean_scores.std():.3f}")
print(f"  Min={mean_scores.min():.3f}, Max={mean_scores.max():.3f}")

# How many phages have mean score near 0.5 (genuinely ambiguous)?
near_half = ((mean_scores > 0.3) & (mean_scores < 0.7)).sum()
print(f"\n  Phages with mean score in (0.3, 0.7): {near_half} / {len(mean_scores)}")

# Break down by gram type
for gt in ["gram-neg", "gram-pos"]:
    mask   = test_df["host_type"] == gt
    scores = mean_scores[mask]
    print(f"\n  {gt}: mean={scores.mean():.3f}, std={scores.std():.3f}")
    print(f"    Phages with score in (0.3, 0.7): {((scores > 0.3) & (scores < 0.7)).sum()} / {len(scores)}")

plt.figure(figsize=(10, 4))
for gt, col in [("gram-neg", "steelblue"), ("gram-pos", "darkorange")]:
    mask = test_df["host_type"] == gt
    plt.hist(mean_scores[mask], bins=50, alpha=0.6, color=col, label=gt, density=True)
plt.axvline(0.3, color="red", linestyle="--", alpha=0.7, label="Ambiguous zone")
plt.axvline(0.7, color="red", linestyle="--", alpha=0.7)
plt.xlabel("Mean raw score per phage")
plt.ylabel("Density")
plt.title("Raw score distribution")
plt.legend()
plt.tight_layout()
plt.show()

# ORIGINAL NOTEBOOK CELL 44 (zero-based)
mean_scores  = pd.Series(test_raw.mean(axis=1), index=test_df.index)
pos_mask     = test_labels == "gram-pos"
pos_indices  = np.where(pos_mask)[0]

typical_pos  = [i for i in pos_indices if mean_scores.iloc[i] > 0.7]
ambiguous_pos = [i for i in pos_indices if mean_scores.iloc[i] <= 0.7]

for name, results in [("Radial", radial_results),
                       ("Strip",  strip_results),
                       ("Collapsed", collapsed_results)]:
    cov_typical   = sum("gram-pos" in results[i]["prediction_set"] for i in typical_pos)
    cov_ambiguous = sum("gram-pos" in results[i]["prediction_set"] for i in ambiguous_pos)
    print(f"{name}:")
    print(f"  Typical gram-pos   (n={len(typical_pos)}): coverage = {cov_typical/len(typical_pos):.3f}")
    if ambiguous_pos:
        print(f"  Ambiguous gram-pos (n={len(ambiguous_pos)}): coverage = {cov_ambiguous/len(ambiguous_pos):.3f}")

# ORIGINAL NOTEBOOK CELL 46 (zero-based)
def plot_method_vs_method_scatter(sizes_x, sizes_y, name_x, name_y):
    gram_colors = {"gram-neg": "steelblue", "gram-pos": "darkorange"}
    fig = plt.figure(figsize=(9, 9))
    gs  = fig.add_gridspec(2, 2, width_ratios=[1, 0.2], height_ratios=[0.2, 1],
                           hspace=0.05, wspace=0.05)
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    jitter = 0.35
    x_jit  = sizes_x + np.random.uniform(-jitter, jitter, len(sizes_x))
    y_jit  = sizes_y + np.random.uniform(-jitter, jitter, len(sizes_y))

    for lbl, col in gram_colors.items():
        mask = np.array(test_labels) == lbl
        ax_main.scatter(x_jit[mask], y_jit[mask], c=col, s=10, alpha=0.4,
                        linewidths=0, label=lbl)

    for v in [0.5, 1.5]:
        ax_main.axvline(v, color="black", linewidth=1.0, alpha=0.4)
        ax_main.axhline(v, color="black", linewidth=1.0, alpha=0.4)

    for xi in [0, 1, 2]:
        for yi in [0, 1, 2]:
            n_box = ((sizes_x == xi) & (sizes_y == yi)).sum()
            ax_main.text(xi, yi, str(n_box), ha="center", va="center",
                         fontsize=12, fontweight="bold",
                         bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))

    ax_main.set_xticks([0, 1, 2])
    ax_main.set_yticks([0, 1, 2])
    ax_main.set_xlim(-0.5, 2.5)
    ax_main.set_ylim(-0.5, 2.5)
    ax_main.set_xlabel(f"{name_x} prediction set size", fontsize=11)
    ax_main.set_ylabel(f"{name_y} prediction set size", fontsize=11)
    ax_main.legend(title="True gram type", fontsize=9, loc="upper left")

    for lbl, col in gram_colors.items():
        mask   = np.array(test_labels) == lbl
        counts = [(sizes_x[mask] == s).sum() for s in [0, 1, 2]]
        ax_top.bar([0, 1, 2], counts, color=col, alpha=0.7, width=0.4, label=lbl)
    ax_top.set_xlim(-0.5, 2.5)
    ax_top.set_xticks([])
    ax_top.set_ylabel("Count", fontsize=9)
    ax_top.set_title(f"Comparison: {name_x} vs {name_y}", fontsize=11)

    for lbl, col in gram_colors.items():
        mask   = np.array(test_labels) == lbl
        counts = [(sizes_y[mask] == s).sum() for s in [0, 1, 2]]
        ax_right.barh([0, 1, 2], counts, color=col, alpha=0.7, height=0.4, label=lbl)
    ax_right.set_ylim(-0.5, 2.5)
    ax_right.set_yticks([])
    ax_right.set_xlabel("Count", fontsize=9)

    plt.show()


sizes_radial    = np.array([len(r["prediction_set"]) for r in radial_results])
sizes_strip     = np.array([len(r["prediction_set"]) for r in strip_results])
sizes_collapsed = np.array([len(r["prediction_set"]) for r in collapsed_results])

plot_method_vs_method_scatter(sizes_collapsed, sizes_radial,  "Collapsed 1D", "Radial")
plot_method_vs_method_scatter(sizes_collapsed, sizes_strip,   "Collapsed 1D", "Strip")
plot_method_vs_method_scatter(sizes_radial,    sizes_strip,   "Radial",       "Strip")

# ORIGINAL NOTEBOOK CELL 47 (zero-based)


# ORIGINAL NOTEBOOK CELL 48 (zero-based)
