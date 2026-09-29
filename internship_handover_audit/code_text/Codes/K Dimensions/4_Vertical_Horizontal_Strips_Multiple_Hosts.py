
# ORIGINAL NOTEBOOK CELL 1 (zero-based)
import numpy as np

# ORIGINAL NOTEBOOK CELL 3 (zero-based)
def simulate_scores(n_phages, K, mean=None, correlation=0.0, variance=0.02):
    """
    Simulates K-dimensional conformal scores for phages that infect a host.
    """
    if mean is None:
        mean = np.full(K, 0.3)

    cov = np.full((K, K), correlation * variance)
    np.fill_diagonal(cov, variance)

    scores = np.abs(np.random.multivariate_normal(mean, cov, n_phages))
    return np.clip(scores, 0, 1)

# ORIGINAL NOTEBOOK CELL 5 (zero-based)
def split_data(scores, split_ratio=0.5):
    """
    Randomly splits scores into S1 (shape discovery) and S2 (size scaling).
    """

    n = len(scores)
    idx = np.random.permutation(n)   
    cut = int(n * split_ratio)   # index to split the data
    return scores[idx[:cut]], scores[idx[cut:]]   

# ORIGINAL NOTEBOOK CELL 7 (zero-based)
def shape_discovery(S1, alpha, M):
    """
    For each conditioning dimension j and each of M equal-width bins along j:
      - Find all S1 points whose j-th value falls in that bin
      - Compute the (1-alpha) quantile of every other dimension i
      → limits[j, i, m] = threshold for dim i when dim j is in bin m
    """
    N1, K = S1.shape       # number of samples and dimensions

    bin_edges = np.zeros((K, M + 1))        # bin edges for each dimension 
    for j in range(K):
        bin_edges[j] = np.linspace(0, S1[:, j].max(), M + 1)

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

    # to ensure that there are no holes in the envelope
    for j in range(K):
        for i in range(K):
            if i == j:
                continue
            for m in range(M - 2, -1, -1):
                limits[j, i, m] = max(limits[j, i, m], limits[j, i, m + 1])

    return bin_edges, limits

# ORIGINAL NOTEBOOK CELL 9 (zero-based)
def get_bin_indices(scores, bin_edges):
    """
    For each point and dimension, finds which bin it falls in.
    Returns (N, K) integer array of 0-based bin indices.
    """
    N, K = scores.shape
    M = bin_edges.shape[1] - 1
    indices = np.zeros((N, K), dtype=int)
    for j in range(K):
        raw = np.digitize(scores[:, j], bin_edges[j]) - 1
        indices[:, j] = np.clip(raw, 0, M - 1)
    return indices

# ORIGINAL NOTEBOOK CELL 10 (zero-based)
def size_scaling(S2, bin_edges, limits, alpha):
    """
    Finds t_hat: the global scale factor for the strip envelope.

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

# ORIGINAL NOTEBOOK CELL 12 (zero-based)
def is_in_region(scores, bin_edges, limits, t_hat):
    """
    Returns True for each score vector inside the scaled strip envelope.

    A point is inside if for every pair (j, i≠j):
        s[i] <= t_hat * limits[j, i, bin_j(s)]
    """
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

# ORIGINAL NOTEBOOK CELL 13 (zero-based)
def build_envelope(infection_scores, alpha, M=20):
    """
    Builds an envelope from the score vectors of phages known
    to infect a given host.
    """

    S1, S2 = split_data(infection_scores)
    bin_edges, limits = shape_discovery(S1, alpha, M)
    t_hat = size_scaling(S2, bin_edges, limits, alpha)
    return {"bin_edges": bin_edges, "limits": limits, "t_hat": t_hat}

# ORIGINAL NOTEBOOK CELL 15 (zero-based)
INFECTS     = "infects"
NOT_INFECTS = "does not infect"

# ORIGINAL NOTEBOOK CELL 16 (zero-based)
def predict_one_host(score_1, envelope):
    """
    Predicts whether a phage infects a given host.
    """

    s1_in = bool(is_in_region(score_1[None, :], envelope["bin_edges"],
                                envelope["limits"], envelope["t_hat"])[0])
    outcome = INFECTS if s1_in else NOT_INFECTS
    return outcome, s1_in

# ORIGINAL NOTEBOOK CELL 17 (zero-based)
def predict_phage_hosts(score_1_per_host, envelopes, hosts):
    """
    Produces the prediction set for a single phage across all hosts.
    """
    prediction_set = []
    results        = {}

    for host in hosts:
        outcome, s1_in = predict_one_host(score_1_per_host[host], envelopes[host])
        results[host]  = {"outcome": outcome, "s1_in": s1_in}
        if s1_in:
            prediction_set.append(host)

    return prediction_set, results

# ORIGINAL NOTEBOOK CELL 19 (zero-based)
np.random.seed(42)

hosts = [f"Host_{chr(65+i)}" for i in range(6)]
true_host = "Host_B"
alpha = 0.1
K = 5
M = 20
N_cal = 500    # calibration phages per host

# ORIGINAL NOTEBOOK CELL 20 (zero-based)
# Building one envelope per host using simulated infector scores
envelopes = {}
for host in hosts:
    host_mean = np.random.uniform(0.1, 0.5, K)
    infectors = simulate_scores(N_cal, K, mean = host_mean, correlation=0.3)
    envelopes[host] = build_envelope(infectors, alpha, M=M)

# ORIGINAL NOTEBOOK CELL 21 (zero-based)
# Simulate score vectors for one test phage
test_scores = {}
for host in hosts:
    if host == true_host:
        test_scores[host] = np.random.uniform(0.1, 0.4, K)
    else:
        test_scores[host] = np.random.uniform(0.6, 0.9, K)

# ORIGINAL NOTEBOOK CELL 22 (zero-based)
pred_set, results = predict_phage_hosts(test_scores, envelopes, hosts)

print("=" * 60)
print(f"STRIP METHOD | alpha={alpha} | true host: {true_host}")
print("=" * 60)
print(f"{'Host':<15} | {'s1_in':<8} | {'Outcome'}")
print("-" * 45)

for host in hosts:
    r      = results[host]
    s1_txt = "YES" if r['s1_in'] else "NO"
    marker = "  <-- TRUE HOST" if host == true_host else ""
    print(f"{host:<15} | {s1_txt:<8} | {r['outcome']}{marker}")

print("-" * 45)
print(f"PREDICTION SET : {pred_set}")
print(f"SET SIZE       : {len(pred_set)} / {len(hosts)} hosts")