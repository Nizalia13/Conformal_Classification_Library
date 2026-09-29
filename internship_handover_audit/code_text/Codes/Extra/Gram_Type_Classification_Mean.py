
# ORIGINAL NOTEBOOK CELL 1 (zero-based)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# np.random.seed(13)

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
combined   = scores_df.join(meta_df, how="inner")
score_cols = [c for c in scores_df.columns if c.startswith("host_type_")]

print(f"Combined shape: {combined.shape}")
print(f"Score columns : {len(score_cols)}")
print("\nGram-type distribution (protein level):")
print(combined["host_type"].value_counts())
print("\nSplit distribution (0=test, 1–4=train):")
print(combined["split"].value_counts().sort_index())

# ORIGINAL NOTEBOOK CELL 7 (zero-based)
print("Mean-pooling proteins → phage level...")
t0 = time.time()

phage_scores = combined.groupby("accession")[score_cols].mean()   # NaN-aware
phage_meta   = combined.groupby("accession")[["host_type", "split"]].first()
phage_df     = phage_scores.join(phage_meta)

print(f"  Phage-level shape: {phage_df.shape}  ({time.time()-t0:.1f}s)")
print("\nGram-type distribution (phage level):")
print(phage_df["host_type"].value_counts())
print("\nSplit distribution (phage level):")
print(phage_df["split"].value_counts().sort_index())


# ORIGINAL NOTEBOOK CELL 9 (zero-based)
nan_rates = phage_df[score_cols].isna().mean().sort_values(ascending=False)

print(f"NaN rate per column (top 10):")
print(nan_rates.head(10).to_string())

NAN_THRESHOLD = 0.50
valid_cols = nan_rates[nan_rates <= NAN_THRESHOLD].index.tolist()
dropped    = nan_rates[nan_rates  > NAN_THRESHOLD].index.tolist()

print(f"\nColumns kept   : {len(valid_cols)} / {len(score_cols)}")
print(f"Columns dropped: {len(dropped)}")
if dropped:
    print(f"  Dropped: {dropped}")

K = len(valid_cols)
print(f"\nFinal K (dimensions) = {K}")

# ORIGINAL NOTEBOOK CELL 11 (zero-based)
# --- 1. Train / Test Split ---
# split=0  → test  |  split=1,2,3,4 → train
train_df = phage_df[phage_df["split"] != 0].copy()
test_df  = phage_df[phage_df["split"] == 0].copy()

# Filter gram types (keeping consistency with your original logic)
train_df = train_df[train_df["host_type"].isin(["gram-neg", "gram-pos"])]
test_df  = test_df[test_df["host_type"].isin(["gram-neg", "gram-pos"])]

print(f"Train phages : {len(train_df)}")
print(f"Test  phages : {len(test_df)}")

# --- 2. Feature Engineering (Missingness Signal) ---
# We calculate this BEFORE imputation, otherwise the NaNs disappear!
train_df['nan_count'] = train_df[valid_cols].isna().sum(axis=1)
test_df['nan_count']  = test_df[valid_cols].isna().sum(axis=1)

# --- 3. Median Imputation with Safety Cap ---
train_medians = train_df[valid_cols].median()

# If a median is > 0.6, we cap it at 0.5 to keep it neutral
safe_impute_values = train_medians.copy()
safe_impute_values[safe_impute_values > 0.6] = 0.5 

# Apply the safe imputation
train_df[valid_cols] = train_df[valid_cols].fillna(safe_impute_values)
test_df [valid_cols] = test_df [valid_cols].fillna(safe_impute_values)

# Residual NaNs (columns entirely NaN in training) → fill with 0.5
train_df[valid_cols] = train_df[valid_cols].fillna(0.5)
test_df [valid_cols] = test_df [valid_cols].fillna(0.5)

# --- 4. Reporting & Extraction ---
print("\nMedians summary after safety cap:")
print(safe_impute_values.describe())

print(f"\nResidual NaNs after imputation — train: {train_df[valid_cols].isna().sum().sum()}")
print(f"Residual NaNs after imputation — test : {test_df[valid_cols].isna().sum().sum()}")

# Extract score matrices (Note: you might want to include 'nan_count' in your final features)
train_gramneg = train_df[train_df["host_type"] == "gram-neg"][valid_cols].values
train_grampos = train_df[train_df["host_type"] == "gram-pos"][valid_cols].values

test_scores = test_df[valid_cols].values
test_labels = test_df["host_type"].values

print(f"\nTraining infectors:")
print(f"  gram-neg : {train_gramneg.shape}")
print(f"  gram-pos : {train_grampos.shape}")

# ORIGINAL NOTEBOOK CELL 12 (zero-based)
# # --- Train / test split ---
# # split=0  → test  |  split=1,2,3,4 → train
# train_df = phage_df[phage_df["split"] != 0].copy()
# test_df  = phage_df[phage_df["split"] == 0].copy()

# # Drop 'unknown' gram type (only 31 phages at phage level)
# train_df = train_df[train_df["host_type"].isin(["gram-neg", "gram-pos"])]
# test_df  = test_df [test_df ["host_type"].isin(["gram-neg", "gram-pos"])]

# print(f"Train phages : {len(train_df)}")
# print(f"Test  phages : {len(test_df)}")

# # --- Median imputation (training medians only) ---
# train_medians = train_df[valid_cols].median()

# train_df[valid_cols] = train_df[valid_cols].fillna(train_medians)
# test_df [valid_cols] = test_df [valid_cols].fillna(train_medians)

# # Residual NaNs (column entirely NaN in training) → fill with 0.5 as a safe neutral
# train_df[valid_cols] = train_df[valid_cols].fillna(0.5)
# test_df [valid_cols] = test_df [valid_cols].fillna(0.5)

# print(f"Residual NaNs after imputation — train: {train_df[valid_cols].isna().sum().sum()}")
# print(f"Residual NaNs after imputation — test : {test_df [valid_cols].isna().sum().sum()}")

# # --- Extract score matrices per gram type ---
# train_gramneg = train_df[train_df["host_type"] == "gram-neg"][valid_cols].values
# train_grampos = train_df[train_df["host_type"] == "gram-pos"][valid_cols].values

# test_scores = test_df[valid_cols].values
# test_labels = test_df["host_type"].values

# print(f"\nTraining infectors:")
# print(f"  gram-neg : {train_gramneg.shape}")
# print(f"  gram-pos : {train_grampos.shape}")

# ORIGINAL NOTEBOOK CELL 14 (zero-based)
def split_data(scores, split_ratio=0.5):
    """Randomly split scores into S1 (shape discovery) and S2 (size scaling)."""
    n = len(scores)
    idx = np.random.permutation(n)
    cut = int(n * split_ratio)
    return scores[idx[:cut]], scores[idx[cut:]]

# ORIGINAL NOTEBOOK CELL 16 (zero-based)
def sample_positive_sphere(M, K):
    """
    Sample M directions uniformly on the positive orthant of the unit sphere S^{K-1}.
    Draw from N(0,I), take absolute values, normalise.
    """
    V = np.abs(np.random.randn(M, K))
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    return V / (norms + 1e-12)

# ORIGINAL NOTEBOOK CELL 17 (zero-based)
def radial_shape_discovery(S1, alpha, M, delta_deg=15):
    """
    For each of M directions on the positive unit sphere:
      1. Find S1 points whose direction is within delta_deg of it
      2. Compute the (1-alpha)
    """
    K = S1.shape[1]
    U = sample_positive_sphere(M, K)
    mags = np.linalg.norm(S1, axis=1)
    dirs = S1 / (mags[:, None] + 1e-12)
    cos_thresh = np.cos(np.radians(delta_deg))
    q_tilde = np.zeros(M)

    for m in range(M):
        sims = dirs @ U[m]
        mask = sims >= cos_thresh
        if mask.sum() < 5:
            q_tilde[m] = np.quantile(mags, 1 - alpha)   # fallback: global quantile
        else:
            q_tilde[m] = np.quantile(mags[mask], 1 - alpha)

    return U, q_tilde

# ORIGINAL NOTEBOOK CELL 18 (zero-based)
def radial_size_scaling(S2, U, q_tilde, alpha):
    """
    Find t_hat: the global scale factor giving the envelope exactly (1-alpha)
    coverage on S2.

    tau(s) = min_m { max_k s[k] / boundary[m,k] }
    t_hat  = (1-alpha) quantile of tau scores over S2.
    """
    boundary = U * q_tilde[:, None]                            # (M, K)
    ratios = S2[:, None, :] / (boundary[None, :, :] + 1e-12)# (N2, M, K)
    t_per_sector = ratios.max(axis=2)                              # (N2, M)
    tau_scores = t_per_sector.min(axis=1)                        # (N2,)

    n2 = len(tau_scores)
    idx = int(np.ceil((n2 + 1) * (1 - alpha))) - 1
    t_hat = np.sort(tau_scores)[np.clip(idx, 0, n2 - 1)]
    return t_hat

# ORIGINAL NOTEBOOK CELL 19 (zero-based)
def radial_is_in_region(scores, envelope):
    """
    True for each score vector inside the scaled radial envelope.
    A point is inside if ANY sector contains it (all K dims ≤ scaled boundary).
    """
    U, q_tilde, t_hat = envelope["U"], envelope["q_tilde"], envelope["t_hat"]
    q_final = q_tilde * t_hat
    boundary = U * q_final[:, None]                                # (M, K)
    inside = np.any(
        np.all(scores[:, None, :] <= boundary[None, :, :], axis=2), axis=1
    )
    return inside

# ORIGINAL NOTEBOOK CELL 20 (zero-based)
def radial_build_envelope(infection_scores, alpha, M=200, delta_deg=15):
    """Full two-stage calibration for the radial method."""
    S1, S2  = split_data(infection_scores)
    U, q_tilde = radial_shape_discovery(S1, alpha, M, delta_deg)
    t_hat = radial_size_scaling(S2, U, q_tilde, alpha)
    return {"U": U, "q_tilde": q_tilde, "t_hat": t_hat, "method": "radial"}

# ORIGINAL NOTEBOOK CELL 22 (zero-based)
def strip_shape_discovery(S1, alpha, M):
    """
    For each conditioning dimension j and each of M equal-width bins along j:
      - Find S1 points in that bin
      - Compute (1-alpha) quantile of every other dimension i
      → limits[j, i, m] = threshold for dim i when dim j is in bin m

    Monotonicity propagation ensures the envelope has no holes.

    Returns
    -------
    bin_edges : (K, M+1) bin boundaries
    limits : (K, K, M) quantile thresholds
    """
    N1, K = S1.shape

    bin_edges = np.zeros((K, M + 1))
    for j in range(K):
        bin_edges[j] = np.linspace(S1[:, j].min(), S1[:, j].max(), M + 1)

    limits = np.zeros((K, K, M))
    for j in range(K):
        for m in range(M):
            lo, hi   = bin_edges[j, m], bin_edges[j, m + 1]
            in_strip = (S1[:, j] >= lo) & (S1[:, j] < hi)
            for i in range(K):
                if i == j:
                    continue
                if in_strip.sum() < 3:
                    limits[j, i, m] = np.quantile(S1[:, i], 1 - alpha)
                else:
                    limits[j, i, m] = np.quantile(S1[in_strip, i], 1 - alpha)

    # Monotonicity: limits are non-decreasing as bin index decreases
    for j in range(K):
        for i in range(K):
            if i == j:
                continue
            for m in range(M - 2, -1, -1):
                limits[j, i, m] = max(limits[j, i, m], limits[j, i, m + 1])

    return bin_edges, limits

# ORIGINAL NOTEBOOK CELL 23 (zero-based)
def get_bin_indices(scores, bin_edges):
    """For each point and dimension, find which bin it falls in."""
    N, K = scores.shape
    M = bin_edges.shape[1] - 1
    indices = np.zeros((N, K), dtype=int)
    for j in range(K):
        raw = np.digitize(scores[:, j], bin_edges[j]) - 1
        indices[:, j] = np.clip(raw, 0, M - 1)
    return indices

# ORIGINAL NOTEBOOK CELL 24 (zero-based)
def strip_size_scaling(S2, bin_edges, limits, alpha):
    """
    Find t_hat for the strip envelope.

    tau(s) = max over (j, i≠j) of: s[i] / limits[j, i, bin_j(s)]
    t_hat = (1-alpha) quantile of tau scores over S2.
    """
    N2, K = S2.shape
    bin_idx = get_bin_indices(S2, bin_edges)

    ratios = np.zeros((N2, K, K))
    for j in range(K):
        for i in range(K):
            if i == j:
                continue
            lims = limits[j, i, bin_idx[:, j]]
            ratios[:, j, i] = S2[:, i] / (lims + 1e-12)

    tau_scores = ratios.max(axis=(1, 2))
    n2 = N2
    idx = int(np.ceil((n2 + 1) * (1 - alpha))) - 1
    t_hat = np.sort(tau_scores)[np.clip(idx, 0, n2 - 1)]
    return t_hat

# ORIGINAL NOTEBOOK CELL 25 (zero-based)
def strip_is_in_region(scores, envelope):
    """
    True for each score vector inside the scaled strip envelope.
    A point is inside if for ALL pairs (j, i≠j): s[i] ≤ t_hat * limits[j,i,bin_j(s)]
    """
    bin_edges, limits, t_hat = (
        envelope["bin_edges"], envelope["limits"], envelope["t_hat"]
    )
    N, K = scores.shape
    bin_idx = get_bin_indices(scores, bin_edges)
    scaled_limits = limits * t_hat

    inside = np.ones(N, dtype=bool)
    for j in range(K):
        for i in range(K):
            if i == j:
                continue
            lims = scaled_limits[j, i, bin_idx[:, j]]
            inside &= (scores[:, i] <= lims)
    return inside

# ORIGINAL NOTEBOOK CELL 26 (zero-based)
def strip_build_envelope(infection_scores, alpha, M=20):
    """Full two-stage calibration for the strip method."""
    S1, S2 = split_data(infection_scores)
    bin_edges, limits = strip_shape_discovery(S1, alpha, M)
    t_hat  = strip_size_scaling(S2, bin_edges, limits, alpha)
    return {"bin_edges": bin_edges, "limits": limits, "t_hat": t_hat, "method": "strip"}

# ORIGINAL NOTEBOOK CELL 27 (zero-based)
def is_in_region(scores, envelope):
    if envelope["method"] == "radial":
        return radial_is_in_region(scores, envelope)
    else:
        return strip_is_in_region(scores, envelope)


def predict_gram_type(test_scores, envelopes):
    """
    For each test phage, check its score vector against both gram-type envelopes
    and return the prediction set.
    """
    in_gramneg = is_in_region(test_scores, envelopes["gram-neg"])
    in_grampos = is_in_region(test_scores, envelopes["gram-pos"])

    results = []
    for i in range(len(test_scores)):
        pred_set = []
        if in_gramneg[i]: pred_set.append("gram-neg")
        if in_grampos[i]: pred_set.append("gram-pos")
        results.append({
            "in_gramneg"    : bool(in_gramneg[i]),
            "in_grampos"    : bool(in_grampos[i]),
            "prediction_set": pred_set,
        })
    return results

# ORIGINAL NOTEBOOK CELL 29 (zero-based)
alpha = 0.1   # target coverage = 1 - alpha = 0.90

print("Building radial envelopes...")
t0 = time.time()
radial_envelopes = {"gram-neg": radial_build_envelope(train_gramneg, alpha, M=200, delta_deg=30),
                    "gram-pos": radial_build_envelope(train_grampos, alpha, M=200, delta_deg=30),}
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  t_hat (gram-neg) = {radial_envelopes['gram-neg']['t_hat']:.4f}")
print(f"  t_hat (gram-pos) = {radial_envelopes['gram-pos']['t_hat']:.4f}")

print("\nBuilding strip envelopes...")
t0 = time.time()
strip_envelopes = {"gram-neg": strip_build_envelope(train_gramneg, alpha, M=20),
                   "gram-pos": strip_build_envelope(train_grampos, alpha, M=20),}
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  t_hat (gram-neg) = {strip_envelopes['gram-neg']['t_hat']:.4f}")
print(f"  t_hat (gram-pos) = {strip_envelopes['gram-pos']['t_hat']:.4f}")

# ORIGINAL NOTEBOOK CELL 31 (zero-based)
def evaluate(results, true_labels, method_name, alpha):
    n          = len(results)
    set_sizes  = [len(r["prediction_set"]) for r in results]

    # Overall coverage
    covered = sum(true_labels[i] in r["prediction_set"] for i, r in enumerate(results))

    # Per-gram-type coverage
    for gt in ["gram-neg", "gram-pos"]:
        idx_gt  = [i for i, lbl in enumerate(true_labels) if lbl == gt]
        cov_gt  = sum(true_labels[i] in results[i]["prediction_set"] for i in idx_gt)
        n_gt    = len(idx_gt)

    # Singleton accuracy
    singletons = [(r, true_labels[i]) for i, r in enumerate(results) if len(r["prediction_set"]) == 1]
    correct_singletons = sum(r["prediction_set"][0] == lbl for r, lbl in singletons)

    empty = sum(s == 0 for s in set_sizes)

    print(f"\n{'='*58}")
    print(f" {method_name.upper()} METHOD — {n} test phages | alpha={alpha}")
    print(f"{'='*58}")
    print(f" Coverage (true type in set) : {covered/n:.3f}  (target ≥ {1-alpha:.2f})")
    print(f" Average prediction set size : {np.mean(set_sizes):.3f}")
    print(f" Empty prediction sets: {empty} ({empty/n*100:.1f}%)")
    if singletons:
        print(f" Singleton accuracy : {correct_singletons/len(singletons):.3f}  ({len(singletons)} singletons)")
    else:
        print(" Singleton accuracy : N/A (no singletons)")

    print(f"\n Per-gram-type coverage:")
    for gt in ["gram-neg", "gram-pos"]:
        idx_gt = [i for i, lbl in enumerate(true_labels) if lbl == gt]
        cov_gt = sum(true_labels[i] in results[i]["prediction_set"] for i in idx_gt)
        n_gt   = len(idx_gt)
        print(f"   {gt:<12}: {cov_gt/n_gt:.3f}  (n={n_gt})")

    return {
        "coverage" : covered / n,
        "avg_set_size" : np.mean(set_sizes),
        "empty_rate" : empty / n,
        "singleton_rate" : len(singletons) / n,
        "singleton_accuracy": correct_singletons / len(singletons) if singletons else None,
        "set_sizes": set_sizes,
        "per_type_coverage" : {
            gt: sum(true_labels[i] in results[i]["prediction_set"]
                    for i in [j for j, lbl in enumerate(true_labels) if lbl == gt])
               / max(1, sum(lbl == gt for lbl in true_labels))
            for gt in ["gram-neg", "gram-pos"]
        },
    }

# ORIGINAL NOTEBOOK CELL 32 (zero-based)
radial_results = predict_gram_type(test_scores, radial_envelopes)
strip_results  = predict_gram_type(test_scores, strip_envelopes)

radial_metrics = evaluate(radial_results, test_labels, "Radial", alpha)
strip_metrics  = evaluate(strip_results,  test_labels, "Strip",  alpha)

# ORIGINAL NOTEBOOK CELL 33 (zero-based)
summary = pd.DataFrame({
    "Method" : ["Radial", "Strip"],
    "Coverage": [radial_metrics["coverage"], strip_metrics["coverage"]],
    "Target coverage": [1 - alpha, 1 - alpha],
    "Avg set size": [radial_metrics["avg_set_size"], strip_metrics["avg_set_size"]],
    "Empty rate" : [radial_metrics["empty_rate"], strip_metrics["empty_rate"]],
    "Singleton accuracy": [radial_metrics["singleton_accuracy"], strip_metrics["singleton_accuracy"]],
})
print(summary.to_string(index=False))


methods = ["Radial", "Strip"]
metrics = [radial_metrics, strip_metrics]

set_size_data = {}

for name, m in zip(methods, metrics):
    sizes = m["set_sizes"]
    n = len(sizes)
    
    # Calculating the fractions for Empty (0), Size 1, and Size 2 predictions
    distribution = {
        "Empty":  sizes.count(0) / n,
        "Size 1": sizes.count(1) / n,
        "Size 2": sizes.count(2) / n
    }
    set_size_data[name] = distribution


print(f"\n\n--- Set Size Distribution (alpha={alpha}) ---")
for method, counts in set_size_data.items():
    print(f"\n{method} Method:")
    for label, fraction in counts.items():
        print(f" {label:<8}: {fraction*100:.2f} %")

# ORIGINAL NOTEBOOK CELL 35 (zero-based)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
methods   = ["Radial", "Strip"]
metrics   = [radial_metrics, strip_metrics]
colors    = ["steelblue", "darkorange"]


# Plot 1: Set size distribution
ax = axes[0]
for name, m, c in zip(methods, metrics, colors):
   sizes  = m["set_sizes"]
   counts = [sizes.count(s) / len(sizes) for s in [0, 1, 2]]
   x      = np.array([0, 1, 2])
   offset = -0.15 if name == "Radial" else 0.15
   ax.bar(x + offset, counts, width=0.3, label=name, color=c, alpha=0.8)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Empty", "Size 1", "Size 2"])
ax.set_title("Set Size Distribution")
ax.set_ylabel("Fraction of test phages")
ax.legend()


# Plot 2: Overall coverage
ax = axes[1]
covs = [radial_metrics["coverage"], strip_metrics["coverage"]]
bars = ax.bar(methods, covs, color=colors, alpha=0.8)
ax.axhline(1 - alpha, color="red", linestyle="--", label=f"Target ({1-alpha:.2f})")
ax.set_ylim(0, 1.05)
ax.set_title("Overall Coverage")
ax.set_ylabel("Coverage")
ax.legend()
for bar, val in zip(bars, covs):
   ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}",
           ha="center", va="bottom", fontsize=10)


# Plot 3: Per-gram-type coverage
ax = axes[2]
gram_types = ["gram-neg", "gram-pos"]
x = np.arange(len(gram_types))
for j, (name, m, c) in enumerate(zip(methods, metrics, colors)):
   vals   = [m["per_type_coverage"][gt] for gt in gram_types]
   offset = -0.2 if name == "Radial" else 0.2
   ax.bar(x + offset, vals, width=0.35, label=name, color=c, alpha=0.8)
ax.axhline(1 - alpha, color="red", linestyle="--", label=f"Target ({1-alpha:.2f})")
ax.set_xticks(x)
ax.set_xticklabels(gram_types)
ax.set_ylim(0, 1.05)
ax.set_title("Coverage by Gram Type")
ax.set_ylabel("Coverage")
ax.legend()


# Plot 4: Singleton accuracy
ax = axes[3]
accs = [
   radial_metrics["singleton_accuracy"] or 0,
   strip_metrics ["singleton_accuracy"] or 0,
]
bars = ax.bar(methods, accs, color=colors, alpha=0.8)
ax.set_ylim(0, 1.05)
ax.set_title("Singleton Accuracy")
ax.set_ylabel("Accuracy")
ax.legend()
for bar, val in zip(bars, accs):
   ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}",
           ha="center", va="bottom", fontsize=10)


plt.suptitle(f"Gram-Type Classification (Mean)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# ORIGINAL NOTEBOOK CELL 36 (zero-based)
