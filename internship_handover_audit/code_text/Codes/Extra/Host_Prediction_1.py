
# ORIGINAL NOTEBOOK CELL 1 (zero-based)
import os
import sys
import pickle
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# numpy compatibility shim (needed on some cluster environments)
import numpy
sys.modules['numpy._core'] = numpy.core
warnings.filterwarnings("ignore")

# ORIGINAL NOTEBOOK CELL 3 (zero-based)
METADATA_PATH = Path("../Data/metadata.csv")

SPLIT_DIRS = {i: Path(f"../Data/random_pc_r_5_da_2_bal_s_{i}") for i in range(5)}

for i, d in SPLIT_DIRS.items():
    status = "OK" if d.exists() else "MISSING"
    print(f"  Split {i}: {d.name}  [{status}]")

# Loading metadata 
print("\nLoading metadata...")
meta_df = pd.read_csv(METADATA_PATH, index_col="proteinID")
print(f"  metadata shape : {meta_df.shape}")

alpha = 0.1

# ORIGINAL NOTEBOOK CELL 5 (zero-based)
gram_types = ["gramneg", "grampos"]

# Discovering function names from split-0 grampos files
sample_files = sorted(SPLIT_DIRS[0].glob("all_preds_grampos_*.pkl"))
functions = sorted([f.stem.replace("all_preds_grampos_", "") for f in sample_files])
print(f"Found {len(functions)} protein functions:")
print("  " + ", ".join(functions))

# Discovering host labels from one reference file per gram type
ref_gramneg = pd.read_pickle(SPLIT_DIRS[0] / f"all_preds_gramneg_{functions[0]}.pkl")
ref_grampos = pd.read_pickle(SPLIT_DIRS[0] / f"all_preds_grampos_{functions[0]}.pkl")
hosts_gramneg = [c.replace(f"_{functions[0]}", "") for c in ref_gramneg.columns]
hosts_grampos = [c.replace(f"_{functions[0]}", "") for c in ref_grampos.columns]
all_hosts = sorted(set(hosts_gramneg + hosts_grampos))

print(f"\\nGram-negative hosts ({len(hosts_gramneg)}): {hosts_gramneg}")
print(f"\\nGram-positive hosts ({len(hosts_grampos)}): {hosts_grampos}")
print(f"\\nAll hosts combined ({len(all_hosts)}): {all_hosts}")

# ORIGINAL NOTEBOOK CELL 6 (zero-based)
# Comparing host sets between Gram types
set_neg = set(hosts_gramneg)
set_pos = set(hosts_grampos)

shared_hosts = sorted(set_neg.intersection(set_pos))
only_neg = sorted(set_neg - set_pos)
only_pos = sorted(set_pos - set_neg)


print(f"Hosts present in BOTH Gram types ({len(shared_hosts)}):")
print(f"  {shared_hosts}")
print(f"\nHosts unique to Gram-negative ONLY ({len(only_neg)}):")
print(f"  {only_neg}")
print(f"\nHosts unique to Gram-positive ONLY ({len(only_pos)}):")
print(f"  {only_pos}")

# ORIGINAL NOTEBOOK CELL 9 (zero-based)
import gc 

print("Assembling PHAGE-level score matrix cleanly...")
t0 = time.time()

# 1. Pre-align your metadata index for lightning-fast matching inside the loops
# We only pull what we need: accession maps proteins to phages, split tells us the fold
meta_lookup = meta_df[["accession", "split"]]

# Containers to hold localized phage-level frames
train_phage_frames = []
test_phage_frames = []

for gram in gram_types:
    for func in functions:
        # Loop through splits 0 to 4 cleanly
        for s in range(5):
            p = SPLIT_DIRS[s] / f"all_preds_{gram}_{func}.pkl"
            if not p.exists():
                continue
                
            # Load the individual file (downcasted to save intermediate RAM)
            prot_df = pd.read_pickle(p).astype(np.float32)
            
            # Instantly join with metadata to separate train/test fractions
            # This drops rows that don't belong to this cross-validation block
            combined = prot_df.join(meta_lookup, how="inner")
            
            if s == 0:
                # --- TEST SET DATA ---
                # Keep only what genuinely belongs to fold 0
                test_subset = combined[combined["split"] == 0]
                if not test_subset.empty:
                    # Compress from 1.3M proteins down to a few thousand phages immediately!
                    phage_test = test_subset.groupby("accession")[prot_df.columns].mean()
                    test_phage_frames.append(phage_test)
            else:
                # --- TRAINING SET DATA (Splits 1, 2, 3, 4) ---
                # Keep only proteins matching their active training assignment
                train_subset = combined[combined["split"] == s]
                if not train_subset.empty:
                    # Compress from proteins to phages immediately!
                    phage_train = train_subset.groupby("accession")[prot_df.columns].mean()
                    train_phage_frames.append(phage_train)
                    
            # Aggressively clear intermediate memory fragments
            del prot_df, combined
            
        gc.collect()

print("Merging columns at the PHAGE level...")
# 2. Combine your files column-wise. Because they are grouped by phage, 
# this deals with thousands of rows instead of millions.
# Outer join automatically inserts NaNs for missing combinations, matching your exact logic!
train_phage_scores = pd.concat(train_phage_frames, axis=1)
test_phage_scores  = pd.concat(test_phage_frames, axis=1)

# Average out duplicate columns if a host/function matches across different frames
# Transpose, group by columns (now rows), take mean, and transpose back
train_phage_scores = train_phage_scores.T.groupby(level=0).mean().T
test_phage_scores  = test_phage_scores.T.groupby(level=0).mean().T

# 3. Re-attach the base phage labels (host, split) from the original metadata
phage_meta = meta_df.groupby("accession")[["host", "split"]].first()

train_phage_df = train_phage_scores.join(phage_meta, how="inner")
test_phage_df  = test_phage_scores.join(phage_meta, how="inner")
train_phage_df["split"] = train_phage_df["split"].astype(int)
test_phage_df["split"]  = test_phage_df["split"].astype(int)

# Extract final score column definitions for your downstream cells
score_cols = [c for c in train_phage_scores.columns]

print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Train phage matrix shape: {train_phage_df.shape} (Phages x Features)")
print(f"  Test  phage matrix shape: {test_phage_df.shape} (Phages x Features)")

# ORIGINAL NOTEBOOK CELL 10 (zero-based)
### Loading Metadata"))
print("Loading metadata ...")
t0 = time.time()
meta_df = pd.read_csv(METADATA_PATH, index_col="proteinID")
meta_df = meta_df[["accession", "host", "host_type", "split"]]
print(f"  metadata shape: {meta_df.shape}  ({time.time()-t0:.1f}s)")
print(meta_df.head(5))
print("\nUnique hosts in metadata:", meta_df["host"].nunique())
print("\nHost value counts (top 10):")
print(meta_df["host"].value_counts().head(10))

# ORIGINAL NOTEBOOK CELL 11 (zero-based)
phage_level_meta = meta_df.groupby("accession")[["host", "host_type"]].first()

print("Phage counts per host (top 10):")
print(phage_level_meta["host"].value_counts().head(10))
print(phage_level_meta["host"].value_counts())

# ORIGINAL NOTEBOOK CELL 13 (zero-based)
# Extract the set of host present as score columns
# Score columns are {Host}_{function} — extract unique host names
scored_hosts = set()
for col in score_cols:
    for func in functions:
        if col.endswith("_" + func):
            host_name = col[: -(len(func) + 1)]
            scored_hosts.add(host_name)
            break
print(f"Hosts covered by score columns ({len(scored_hosts)}): {sorted(scored_hosts)}")

# Keep only phages whose true host is in scored_hosts
train_phage_df = train_phage_df[train_phage_df["host"].isin(scored_hosts)].copy()
test_phage_df = test_phage_df[test_phage_df["host"].isin(scored_hosts)].copy()

print(f"\\nAfter filtering:")
print(f"  Train phages: {len(train_phage_df)}")
print(f"  Test  phages: {len(test_phage_df)}")
print("\\nTrain host distribution:")
print(train_phage_df["host"].value_counts())

# ORIGINAL NOTEBOOK CELL 15 (zero-based)
sorted_hosts = sorted(scored_hosts)
H = len(sorted_hosts)

def build_host_score_matrix(phage_df, host_list, score_cols, functions):
    """
    For each phage, compute one aggregated score per host by averaging
    across all function columns for that host.
    Returns array of shape (n_phages, n_hosts).
    """
    n = len(phage_df)
    mat = np.full((n, len(host_list)), np.nan)
    for j, host in enumerate(host_list):
        host_cols = [c for c in score_cols if c.startswith(host + "_")]
        if not host_cols:
            continue
        mat[:, j] = phage_df[host_cols].mean(axis=1).values
    return mat

print("Building host score matrices ...")
t0 = time.time()
train_raw_full = build_host_score_matrix(train_phage_df, sorted_hosts, score_cols, functions)
test_raw_full  = build_host_score_matrix(test_phage_df,  sorted_hosts, score_cols, functions)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Train host score matrix: {train_raw_full.shape}  (phages x hosts)")
print(f"  Test  host score matrix: {test_raw_full.shape}")

train_labels = train_phage_df["host"].to_numpy(dtype=str)
test_labels  = test_phage_df["host"].to_numpy(dtype=str)

# ORIGINAL NOTEBOOK CELL 17 (zero-based)
# Compute global column medians from training set only
train_medians = np.nanmedian(train_raw_full, axis=0)   # shape (H,)

# Cap at 0.6 → 0.5 to avoid imputing artificially high host affinity
safe_medians = np.where(train_medians > 0.6, 0.5, train_medians)

# Apply to both train and test
def impute(mat, medians):
    out = mat.copy()
    nan_mask = np.isnan(out)
    out[nan_mask] = np.take(medians, np.where(nan_mask)[1])
    return out

train_raw = impute(train_raw_full, safe_medians)
test_raw = impute(test_raw_full,  safe_medians)

# Final fallback for any host column with no training data at all
train_raw = np.nan_to_num(train_raw, nan=0.5)
test_raw = np.nan_to_num(test_raw,  nan=0.5)

nan_frac_train = np.isnan(train_raw_full).mean()
nan_frac_test  = np.isnan(test_raw_full).mean()
print(f"NaN fraction before imputation — train: {nan_frac_train:.3f}, test: {nan_frac_test:.3f}")
print(f"NaN fraction after  imputation — train: {np.isnan(train_raw).mean():.3f}, test: {np.isnan(test_raw).mean():.3f}")

print("\nImputed medians (first 5 hosts):")
for h, m, sm in zip(sorted_hosts[:5], train_medians[:5], safe_medians[:5]):
    print(f"  {h}: raw_median={m:.4f}, safe_impute={sm:.4f}")

# ORIGINAL NOTEBOOK CELL 19 (zero-based)
print("Mean host score per phage (averaged across all host columns):")
host_to_idx = {h: i for i, h in enumerate(sorted_hosts)}
for host in sorted_hosts:
    mask = train_labels == host
    if mask.sum() == 0:
        continue
    j = host_to_idx[host]
    mean_self = train_raw[mask, j].mean()   # score for own (true) host
    mean_other = np.delete(train_raw[mask], j, axis=1).mean()  # score for other hosts
    print(f"  {host:25s}: self_score={mean_self:.3f}  other_mean={mean_other:.3f}  n={mask.sum()}")

# ORIGINAL NOTEBOOK CELL 21 (zero-based)
def get_true_label_nc(raw_scores, labels, host_list):
    """
    For each phage, extract the score value for its true host.
    transformed score = 1 - score_{true_host}
    Returns 1D array of shape (n_phages,).
    """
    host_to_idx = {h: i for i, h in enumerate(host_list)}
    nc = np.array([
        1.0 - raw_scores[i, host_to_idx[labels[i]]]
        for i in range(len(labels))
        if labels[i] in host_to_idx
    ])
    return nc

nc_train_true = get_true_label_nc(train_raw, train_labels, sorted_hosts)
print(f"True-label scores (train): mean={nc_train_true.mean():.3f}, std={nc_train_true.std():.3f}")
print(f"  Low scores (well predicted): {(nc_train_true < 0.3).mean()*100:.1f}% of phages")
print(f"  High scores (poorly predicted): {(nc_train_true > 0.7).mean()*100:.1f}% of phages")

# ORIGINAL NOTEBOOK CELL 23 (zero-based)
train_shuffled_idx = np.random.default_rng(42).permutation(len(train_raw))
half = len(train_shuffled_idx) // 2
s1_idx = train_shuffled_idx[:half]
s2_idx = train_shuffled_idx[half:]

s1_raw = train_raw[s1_idx]
s2_raw = train_raw[s2_idx]
labels_S1 = train_labels[s1_idx]
labels_S2 = train_labels[s2_idx]

print(f"S1 (shape discovery): {s1_raw.shape}")
print(f"S2 (size scaling):    {s2_raw.shape}")
print("\\nHost distribution in S2 (top 10):")
s2_counts = pd.Series(labels_S2).value_counts()
print(s2_counts.head(10))
min_count = s2_counts.min()
print(f"\\nSmallest class in S2: {s2_counts.idxmin()} — {min_count} phages")
if min_count < 5:
    print("  WARNING: very few S2 phages for this host — quantile will be unreliable")

# ORIGINAL NOTEBOOK CELL 25 (zero-based)
def quantile_for_class(tau_scores, labels, cls, alpha):
    """
    Compute the conformal quantile for a single class.
    Returns the ceiling((n+1)(1-alpha))-th smallest tau score.
    """
    vals = np.sort(tau_scores[labels == cls])
    n = len(vals)
    if n == 0:
        return 0.0
    idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return vals[np.clip(idx, 0, n - 1)]

# def t_hat_over_all_classes(tau_scores, labels, alpha, host_list):
#     """
#     Compute per-class conformal quantile and take the maximum.
#     This guarantees coverage >= (1-alpha) for every host class.
#     Uses t_hat_per_class max strategy — guarantees the worst-covered
#     class still meets the target.
#     """
#     per_class = {
#         h: quantile_for_class(tau_scores, labels, h, alpha)
#         for h in host_list
#     }
#     t_hat = max(per_class.values())
#     worst = max(per_class, key=per_class.get)
#     print(f"    t_hat = {t_hat:.4f}  (worst class: {worst})")
#     return t_hat, per_class

def t_hat_over_all_classes(tau_scores, labels, alpha, host_list):
    # Treat all tau scores as one global pool and find the global (1 - alpha) quantile
    t_hat = np.quantile(tau_scores, 1 - alpha)
    
    # seeing how individual classes would have fared
    per_class = {
        h: quantile_for_class(tau_scores, labels, h, alpha)
        for h in host_list
    }
    
    print(f"  GLOBAL POOL t_hat = {t_hat:.4f}")
    return t_hat, per_class

# ORIGINAL NOTEBOOK CELL 27 (zero-based)
def collapsed_build_envelope(s1_raw, s2_raw, labels_S1, labels_S2, alpha, host_list):
    """
    Build collapsed 1D conformal envelope.
    S1: learn q_tilde as the (1-alpha) quantile of true-host NC scores.
    S2: compute tau = NC_{true} / q_tilde for each phage; calibrate t_hat.
    """
    host_to_idx = {h: i for i, h in enumerate(host_list)}
    # S1: true-label NC scores
    nc_s1 = np.array([1.0 - s1_raw[i, host_to_idx[labels_S1[i]]]
                      for i in range(len(labels_S1))
                      if labels_S1[i] in host_to_idx])
    q_tilde = np.quantile(nc_s1, 1 - alpha)
    # S2: tau scores
    nc_s2 = np.array([1.0 - s2_raw[i, host_to_idx[labels_S2[i]]]
                      for i in range(len(labels_S2))
                      if labels_S2[i] in host_to_idx])
    tau_scores = nc_s2 / (q_tilde + 1e-12)
    # Use per-class t_hat to guarantee all classes covered
    labels_S2_valid = np.array([labels_S2[i] for i in range(len(labels_S2))
                                if labels_S2[i] in host_to_idx])
    t_hat, per_class = t_hat_over_all_classes(tau_scores, labels_S2_valid, alpha, host_list)
    return {"method": "collapsed", "q_tilde": q_tilde, "t_hat": t_hat,
            "t_hat_per_class": per_class}

def collapsed_predict(test_raw, envelope, host_list):
    """
    For each test phage, return the prediction set of hosts whose
    NC score is within the calibrated envelope.
    """
    threshold = envelope["q_tilde"] * envelope["t_hat"]
    results = []
    for i in range(len(test_raw)):
        pred_set = []
        for j, h in enumerate(host_list):
            nc = 1.0 - test_raw[i, j]
            if nc <= threshold:
                pred_set.append(h)
        results.append({"prediction_set": pred_set})
    return results

print("Building Collapsed 1D envelope ...")
t0 = time.time()
c_env = collapsed_build_envelope(s1_raw, s2_raw, labels_S1, labels_S2, alpha, sorted_hosts)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  q_tilde = {c_env['q_tilde']:.4f},  t_hat = {c_env['t_hat']:.4f}")

# ORIGINAL NOTEBOOK CELL 29 (zero-based)
def sample_positive_sphere(M, K, seed=0):
    rng = np.random.default_rng(seed)
    V = np.abs(rng.standard_normal((M, K)))
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

def radial_build_envelope(s1_raw, s2_raw, labels_S1, labels_S2, alpha, host_list, 
                           M=200, delta_deg=30):
    K = len(host_list)
    U = sample_positive_sphere(M, K)
    # S1: true-label NC vectors — for host classification NC_j = 1 - score_j
    # Shape discovery uses the FULL NC vector (all hosts) of each S1 phage
    S1_nc = 1.0 - s1_raw   # (n_s1, K)
    mags  = np.linalg.norm(S1_nc, axis=1)
    dirs  = S1_nc / (mags[:, None] + 1e-12)
    cos_thresh = np.cos(np.radians(delta_deg))
    q_tilde = np.array([
        np.quantile(mags[(dirs @ U[m]) >= cos_thresh], 1 - alpha)
        if ((dirs @ U[m]) >= cos_thresh).sum() > 5
        else np.quantile(mags, 1 - alpha)
        for m in range(M)
    ])

    # S2: tau scores
    S2_nc = 1.0 - s2_raw   # (n_s2, K)
    # tau_i = min over m of max_k( S2_nc[i,k] / (U[m,k] * q_tilde[m]) )
    tau_scores = (
        (S2_nc[:, None, :] / (U * q_tilde[:, None])[None, :, :])
        .max(axis=2).min(axis=1)
    )
    labels_S2_valid = labels_S2   # all should be valid after earlier filter
    t_hat, per_class = t_hat_over_all_classes(tau_scores, labels_S2_valid, alpha, host_list)
    return {"method": "radial", "U": U, "q_tilde": q_tilde,
            "t_hat": t_hat, "t_hat_per_class": per_class}

def radial_predict(test_raw, envelope, host_list):
    U, q_tilde, t_hat = envelope["U"], envelope["q_tilde"], envelope["t_hat"]
    results = []
    for i in range(len(test_raw)):
        pred_set = []
        for j, h in enumerate(host_list):
            # Evaluate the scalar radial check for host h
            nc_h = 1.0 - test_raw[i, j]
            if nc_h <= q_tilde.mean() * t_hat:
                pred_set.append(h)
        results.append({"prediction_set": pred_set})
    return results

print("\nBuilding Radial envelope ...")
t0 = time.time()
r_env = radial_build_envelope(s1_raw, s2_raw, labels_S1, labels_S2, alpha, sorted_hosts, M=200)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  t_hat = {r_env['t_hat']:.4f}")

# ORIGINAL NOTEBOOK CELL 31 (zero-based)
def strip_build_envelope(s1_raw, s2_raw, labels_S1, labels_S2, alpha, host_list, M=10):
    S1_nc = 1.0 - s1_raw
    S2_nc = 1.0 - s2_raw
    K = S1_nc.shape[1]
    bin_edges = np.array([np.linspace(S1_nc[:, j].min(), S1_nc[:, j].max(), M + 1) for j in range(K)])
    limits = np.zeros((K, K, M))
    for j in range(K):
        for m in range(M):
            lo, hi = bin_edges[j, m], bin_edges[j, m + 1]
            in_strip = (S1_nc[:, j] >= lo) & (S1_nc[:, j] < hi)
            for i in range(K):
                if i == j:
                    continue
                src = S1_nc[in_strip, i] if in_strip.sum() >= 3 else S1_nc[:, i]
                limits[j, i, m] = np.quantile(src, 1 - alpha)

    # Monotonicity propagation
    for j in range(K):
        for i in range(K):
            if i == j:
                continue
            for m in range(M - 2, -1, -1):
                limits[j, i, m] = max(limits[j, i, m], limits[j, i, m + 1])

    # S2 tau scores
    def get_tau(nc_mat):
        N, K_ = nc_mat.shape
        M_ = bin_edges.shape[1] - 1
        bin_idx = np.array([
            np.clip(np.digitize(nc_mat[:, j], bin_edges[j]) - 1, 0, M_ - 1) for j in range(K_)]).T
        ratios = np.zeros((N, K_, K_))
        for j in range(K_):
            for i in range(K_):
                if i == j:
                    continue
                lims = limits[j, i, bin_idx[:, j]]
                ratios[:, j, i] = nc_mat[:, i] / (lims + 1e-12)
        return ratios.max(axis=(1, 2))

    tau_scores = get_tau(S2_nc)
    t_hat, per_class = t_hat_over_all_classes(tau_scores, labels_S2, alpha, host_list)
    return {"method": "strip", "bin_edges": bin_edges, "limits": limits,
            "t_hat": t_hat, "t_hat_per_class": per_class, "_get_tau": get_tau}

def strip_predict(test_raw, envelope, host_list):
    bin_edges = envelope["bin_edges"]
    limits = envelope["limits"]
    t_hat = envelope["t_hat"]
    K = len(host_list)
    M = bin_edges.shape[1] - 1
    test_nc = 1.0 - test_raw
    N = len(test_raw)
    bin_idx = np.array([
        np.clip(np.digitize(test_nc[:, j], bin_edges[j]) - 1, 0, M - 1)
        for j in range(K)
    ]).T
    results = []
    for i in range(N):
        pred_set = []
        for j_h, h in enumerate(host_list):
            nc_h = test_nc[i, j_h]
            ok = True
            for j2 in range(K):
                if j2 == j_h:
                    continue
                lim = limits[j2, j_h, bin_idx[i, j2]]
                if nc_h > lim * t_hat:
                    ok = False
                    break
            if ok:
                pred_set.append(h)
        results.append({"prediction_set": pred_set})
    return results

print("\nBuilding Strip envelope (this may take a few minutes for large H) ...")
t0 = time.time()
s_env = strip_build_envelope(s1_raw, s2_raw, labels_S1, labels_S2, alpha, sorted_hosts, M=10)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  t_hat = {s_env['t_hat']:.4f}")

# ORIGINAL NOTEBOOK CELL 33 (zero-based)
def predict(test_raw, envelope, host_list):
    if envelope["method"] == "collapsed":
        return collapsed_predict(test_raw, envelope, host_list)
    elif envelope["method"] == "radial":
        return radial_predict(test_raw, envelope, host_list)
    elif envelope["method"] == "strip":
        return strip_predict(test_raw, envelope, host_list)
    raise ValueError(f"Unknown method: {envelope['method']}")


# ORIGINAL NOTEBOOK CELL 35 (zero-based)
def evaluate(results, true_labels, host_list, alpha):
    n = len(results)
    set_sizes = [len(r["prediction_set"]) for r in results]
    covered   = sum(true_labels[i] in r["prediction_set"] for i, r in enumerate(results))
    singletons = [(r, true_labels[i]) for i, r in enumerate(results) if len(r["prediction_set"]) == 1]
    correct_singletons = sum(r["prediction_set"][0] == lbl for r, lbl in singletons)
    empty  = sum(s == 0 for s in set_sizes)
    per_host_cov = {}
    for h in host_list:
        idx = [i for i in range(n) if true_labels[i] == h]
        if not idx:
            continue
        per_host_cov[h] = sum(true_labels[i] in results[i]["prediction_set"] for i in idx) / len(idx)
    return {
        "coverage":          covered / n,
        "avg_set_size":      np.mean(set_sizes),
        "empty_rate":        empty / n,
        "singleton_rate":    len(singletons) / n,
        "singleton_accuracy": correct_singletons / len(singletons) if singletons else None,
        "set_sizes":         set_sizes,
        "per_host_coverage": per_host_cov,
    }

# ORIGINAL NOTEBOOK CELL 36 (zero-based)
print("\n" + "="*60)
print(" EVALUATING ON HELD-OUT TEST SET (FOLD 0) ")
print("="*60)

print("\nPredicting with Collapsed 1D ...")
t0 = time.time()
c_res = predict(test_raw, c_env, sorted_hosts)
print(f"  Done in {time.time()-t0:.1f}s")

print("Predicting with Radial ...")
t0 = time.time()
r_res = predict(test_raw, r_env, sorted_hosts)
print(f"  Done in {time.time()-t0:.1f}s")

print("Predicting with Strip ...")
t0 = time.time()
s_res = predict(test_raw, s_env, sorted_hosts)
print(f"  Done in {time.time()-t0:.1f}s")

test_metrics = {}
for key, res in [("collapsed", c_res), ("radial", r_res), ("strip", s_res)]:
    test_metrics[key] = evaluate(res, test_labels, sorted_hosts, alpha)

# Summary table
methods_names = ["Collapsed 1D", "Radial", "Strip"]
method_keys   = ["collapsed",    "radial", "strip"]
rows = []
for name, key in zip(methods_names, method_keys):
    m = test_metrics[key]
    rows.append({
        "Method":         name,
        "Coverage":       f"{m['coverage']:.3f}",
        "Avg set size":   f"{m['avg_set_size']:.2f}",
        "Empty %":        f"{m['empty_rate']*100:.1f}%",
        "Singleton %":    f"{m['singleton_rate']*100:.1f}%",
        "Singleton acc.": f"{m['singleton_accuracy']:.3f}" if m['singleton_accuracy'] else "N/A",
    })
summary_df = pd.DataFrame(rows)
print("\n=== DEFINITIVE HELD-OUT TEST RESULTS (FOLD 0) ===")
print(summary_df.to_string(index=False))

# ORIGINAL NOTEBOOK CELL 38 (zero-based)
rows_host = []
for h in sorted_hosts:
    row = {"Host": h, "N_test": sum(test_labels == h)}
    for name, key in zip(methods_names, method_keys):
        cov = test_metrics[key]["per_host_coverage"].get(h, float("nan"))
        row[f"{name}"] = f"{cov:.3f}"
    rows_host.append(row)
host_df = pd.DataFrame(rows_host)
print("\n=== PER-HOST COVERAGE ===")
print(host_df.to_string(index=False))

# Flag any host below target
print("\nHosts below target coverage (< 0.90):")
for name, key in zip(methods_names, method_keys):
    below = [(h, v) for h, v in test_metrics[key]["per_host_coverage"].items() if v < 1 - alpha]
    if below:
        print(f"  {name}: " + ", ".join(f"{h}={v:.3f}" for h, v in below))
    else:
        print(f"  {name}: all hosts meet target")

# ORIGINAL NOTEBOOK CELL 40 (zero-based)
fig, axes = plt.subplots(1, 4, figsize=(24, 5))
colors  = ["steelblue", "darkorange", "mediumseagreen"]
offsets = [-0.25, 0.0, 0.25]

# Plot 1: Set size distribution (0, 1, 2, 3+)
ax = axes[0]
size_labels = ["Empty (0)", "Singleton (1)", "Size 2", "Size 3+"]
x = np.arange(len(size_labels))
for name, key, c, off in zip(methods_names, method_keys, colors, offsets):
    m = test_metrics[key]
    n_total = len(m["set_sizes"])
    fracs = [
        sum(1 for s in m["set_sizes"] if s == 0) / n_total,
        sum(1 for s in m["set_sizes"] if s == 1) / n_total,
        sum(1 for s in m["set_sizes"] if s == 2) / n_total,
        sum(1 for s in m["set_sizes"] if s >= 3) / n_total,
    ]
    bars = ax.bar(x + off, fracs, width=0.23, label=name, color=c, alpha=0.8)
    for rect, val in zip(bars, fracs):
        if val > 0.01:
            ax.text(rect.get_x() + rect.get_width()/2, val + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(size_labels, fontsize=8)
ax.set_title("Set Size Distribution")
ax.set_ylabel("Fraction of test phages")
ax.legend(fontsize=8)

# Plot 2: Overall coverage
ax = axes[1]
means = [test_metrics[k]["coverage"] for k in method_keys]
bars  = ax.bar(methods_names, means, color=colors, alpha=0.8)
ax.axhline(1 - alpha, color="red", linestyle="--", label=f"Target ({1-alpha:.2f})")
ax.set_ylim(0, 1.05)
ax.set_title("Overall Coverage")
ax.set_ylabel("Coverage")
ax.legend(fontsize=8)
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)

# Plot 3: Per-host coverage heatmap
ax = axes[2]
cov_matrix = np.array([
    [test_metrics[key]["per_host_coverage"].get(h, 0) for h in sorted_hosts]
    for key in method_keys
])
im = ax.imshow(cov_matrix, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
ax.set_xticks(range(len(sorted_hosts)))
ax.set_xticklabels(sorted_hosts, rotation=90, fontsize=6)
ax.set_yticks(range(len(method_keys)))
ax.set_yticklabels(methods_names, fontsize=8)
ax.set_title("Per-Host Coverage")
plt.colorbar(im, ax=ax, shrink=0.8)

# Plot 4: Singleton accuracy
ax = axes[3]
accuracies = [test_metrics[k]["singleton_accuracy"] or 0 for k in method_keys]
ns = [int(test_metrics[k]["singleton_rate"] * len(test_metrics[k]["set_sizes"])) for k in method_keys]
bars = ax.bar(methods_names, accuracies, color=colors, alpha=0.8)
ax.set_ylim(0, 1.05)
ax.set_title("Singleton Accuracy")
ax.set_ylabel("Accuracy (when set size = 1)")
for bar, val, n_s in zip(bars, accuracies, ns):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
            f"{val:.3f}\n(n={n_s})", ha="center", va="bottom", fontsize=8)

plt.suptitle("Host-Level Classification Performance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("host_classification_results.png", dpi=150, bbox_inches="tight")

# ORIGINAL NOTEBOOK CELL 42 (zero-based)
print("\nTrue-host score distribution (test set):")
true_host_scores = np.array([
    test_raw[i, sorted_hosts.index(test_labels[i])]
    for i in range(len(test_labels))
    if test_labels[i] in sorted_hosts
])
print(f"  Mean: {true_host_scores.mean():.3f}, Std: {true_host_scores.std():.3f}")
print(f"  Phages with true-host score > 0.5: {(true_host_scores > 0.5).mean()*100:.1f}%")
print(f"  Phages with true-host score < 0.2: {(true_host_scores < 0.2).mean()*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# True-host score histogram
ax = axes[0]
ax.hist(true_host_scores, bins=50, color="steelblue", alpha=0.8, density=True)
ax.axvline(0.5, color="red", linestyle="--", label="Score = 0.5")
ax.set_xlabel("Score for true host")
ax.set_ylabel("Density")
ax.set_title("True-Host Score Distribution (Test Set)")
ax.legend()

# Per-host mean true-host score
ax = axes[1]
host_means = []
host_ns = []
for h in sorted_hosts:
    mask = test_labels == h
    if mask.sum() == 0:
        host_means.append(0)
        host_ns.append(0)
        continue
    j = sorted_hosts.index(h)
    host_means.append(test_raw[mask, j].mean())
    host_ns.append(mask.sum())
bars = ax.bar(sorted_hosts, host_means, color="darkorange", alpha=0.8)
ax.axhline(0.5, color="red", linestyle="--", alpha=0.7)
ax.set_xticks(range(len(sorted_hosts)))
ax.set_xticklabels(sorted_hosts, rotation=90, fontsize=7)
ax.set_ylabel("Mean true-host score")
ax.set_title("Mean True-Host Score per Genus (Test)")
for bar, n in zip(bars, host_ns):
    ax.text(bar.get_x() + bar.get_width()/2, 0.01, f"n={n}", ha="center", va="bottom", fontsize=6, rotation=90, color="white" if n>0 else "black")

plt.tight_layout()
plt.savefig("host_score_distribution.png", dpi=150, bbox_inches="tight")
plt.show()