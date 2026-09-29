
# ORIGINAL NOTEBOOK CELL 1 (zero-based)
import numpy as np

# ORIGINAL NOTEBOOK CELL 3 (zero-based)
def simulate_scores(n_phages, K, mean=None, correlation=0.0, variance=0.02):
    """
    Simulates K-dimensional conformal scores.
    mean = array of length K (specifies the mean for all directions)
    """
    if mean is None:
        mean = np.full(K, 0.3)   # default mean for all dimensions
    cov = np.full((K, K), correlation * variance)
    np.fill_diagonal(cov, variance)
    scores = np.abs(np.random.multivariate_normal(mean, cov, n_phages))

    return np.clip(scores, 0 , 1)

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
    Builds a strip-based envelope shape from calibration data S1.

    For each conditioning dimension j and each of M equal-width bins along j we:
      - Find all S1 points whose j-th value falls in that bin
      - Compute the (1-alpha) quantile of every other dimension i in that bin
      → limits[j, i, m] = threshold for dim i when dim j is in bin m

    Monotonicity enforcement: limits are made non-decreasing as bin index
    decreases to ensure that the envelope has no holes.

    Returns
      - bin_edges : (K, M+1) bin boundaries for each dimension
      - limits    : (K, K, M) quantile thresholds
    """
    N1, K = S1.shape

    # for each dimension, we ware creating M equal-width bins over [0, 1]
    bin_edges = np.zeros((K, M + 1))
    for j in range(K):
        bin_edges[j] = np.linspace(S1[:, j].min(), S1[:, j].max(), M + 1)

    # limits[j, i, m] = (1-alpha) quantile of S1[:, i] for points in strip m of dim j
    limits = np.zeros((K, K, M))

    for j in range(K):                    # conditioning dimension
        for m in range(M):                # index of the strip aalong dimension j
            lo, hi   = bin_edges[j, m], bin_edges[j, m + 1]     # strip boundaries along dimension j
            in_strip = (S1[:, j] >= lo) & (S1[:, j] < hi)       # mask for points in the strip (boolean)

            for i in range(K):                  # dimensions to compute quantiles for
                if i == j:
                    continue
                if in_strip.sum() < 3:
                     # If there are very few points in the strip we use global quantile
                    limits[j, i, m] = np.quantile(S1[:, i], 1 - alpha)
                else:
                    limits[j, i, m] = np.quantile(S1[in_strip, i], 1 - alpha)

    # We need to ensure that the limits are non-increasing as the strip index increases 
    # (higher values of j shouldn't allow more of i).
    # Without this, the envelope can have holes.
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
    For each point and each dimension, finds which bin it falls in.
    """


    N, K = scores.shape
    M    = bin_edges.shape[1] - 1      # number of bins is one less than the number of edges
    indices = np.zeros((N, K), dtype=int)
    for j in range(K):
        raw = np.digitize(scores[:, j], bin_edges[j]) - 1   
        indices[:, j] = np.clip(raw, 0, M - 1)
    return indices

# ORIGINAL NOTEBOOK CELL 10 (zero-based)
def size_scaling(S2, bin_edges, limits, alpha):
    """
    Finds t_hat by computing a tau score for each S2 point.

    tau for point s = max over all (j, i≠j) pairs of: s[i] / limits[j, i, bin_j(s)]
    that is how much would we need to scale all the limits to include s - we take the max ratio along all pairs of dimensions and comditions
    t_hat is then the (1-alpha) quantile of these tau scores.
    """

    N2, K   = S2.shape
    bin_idx = get_bin_indices(S2, bin_edges)     # finding the bin

    # ratios[n, j, i] = S2[n, i] / limits[j, i, bin_idx[n, j]]
    ratios = np.zeros((N2, K, K))
    for j in range(K):
        for i in range(K):
            if i == j:
                continue
            lims = limits[j, i, bin_idx[:, j]]        # (N2,)
            ratios[:, j, i] = S2[:, i] / (lims + 1e-12)


    tau_scores = ratios.max(axis=(1, 2))               # (N2,)

    n2    = N2
    idx   = int(np.ceil((n2 + 1) * (1 - alpha))) - 1
    t_hat = np.sort(tau_scores)[np.clip(idx, 0, n2 - 1)]

    return t_hat, tau_scores

# ORIGINAL NOTEBOOK CELL 12 (zero-based)
def is_in_region(scores, bin_edges, limits, t_hat):
    """
    Returns a boolean array: True if the score vector lies inside the
    scaled strip envelope.

    A point s is inside if for every pair (j, i≠j):
        s[i] <= t_hat * limits[j, i, bin_j(s)]
    """

    N, K          = scores.shape
    bin_idx       = get_bin_indices(scores, bin_edges)
    scaled_limits = limits * t_hat  # scaling all the limits by t_hat

    inside = np.ones(N, dtype=bool)  # starting with all points inside, then marking those that violate any condition as false
    for j in range(K):
        for i in range(K):
            if i == j:
                continue
            lims    = scaled_limits[j, i, bin_idx[:, j]]
            inside &= (scores[:, i] <= lims)

    return inside

# ORIGINAL NOTEBOOK CELL 14 (zero-based)
def evaluate_coverage(S2, bin_edges, limits, t_hat):
    """
    Empirical coverage on the size-scaling set S2.
    Should be >= (1 - alpha)
    """
    return is_in_region(S2, bin_edges, limits, t_hat).mean()

# ORIGINAL NOTEBOOK CELL 16 (zero-based)
np.random.seed(42)
alpha = 0.1
M     = 20    # bins per dimension data per bin shrinks fast with K

print(f"{'K':>4} | {'t_hat':>8} | {'coverage':>10}")
print("-" * 30)

for K in [2, 5, 10, 20]:
    scores            = simulate_scores(n_phages=2000, K=K, correlation=0.3)
    S1, S2            = split_data(scores, split_ratio=0.5)
    bin_edges, limits = shape_discovery(S1, alpha, M)
    t_hat, _          = size_scaling(S2, bin_edges, limits, alpha)
    coverage          = evaluate_coverage(S2, bin_edges, limits, t_hat)

    print(f"{K:>4} | {t_hat:>8.3f} | {coverage:>10.3f}")

# ORIGINAL NOTEBOOK CELL 17 (zero-based)
