
# ORIGINAL NOTEBOOK CELL 1 (zero-based)
import numpy as np

# ORIGINAL NOTEBOOK CELL 3 (zero-based)
def simulate_scores(n_phages, K, mean=None, correlation=0.0, variance=0.02):
    """
    Simulates K-dimensional scores for phages that infect a host.
    
    Paramaters:
    - n_phages: number of phages to simulate
    - K: number of weak learners (functions)
    - mean: mean vector of length K 
    
    Returns:
    - scores: (n_phages, K) array clipped to [0, 1]
    """

    # If the mean is not provided we use a default value.
    if mean is None:
        mean = np.full(K, 0.3)

    # Creating a covariance matrix 
    cov = np.full((K, K), correlation * variance)
    np.fill_diagonal(cov, variance)

    # Simulating scores from a multivariate normal distribution and clipping them to [0, 1]
    scores = np.abs(np.random.multivariate_normal(mean, cov, n_phages))
    return np.clip(scores, 0, 1)

# ORIGINAL NOTEBOOK CELL 5 (zero-based)
def split_data(scores, split_ratio=0.5):
    """
    Randomly splits scores into S1 (shape discovery) and S2 (size scaling).
    """
    n   = len(scores)
    idx = np.random.permutation(n)
    cut = int(n * split_ratio)    # index to split the data
    return scores[idx[:cut]], scores[idx[cut:]]

# ORIGINAL NOTEBOOK CELL 7 (zero-based)
def sample_positive_sphere(M, K):
    """
    Samples M directions uniformly on the positive orthant of S^(K-1).
    Draws from N(0, I), take absolute values, normalise.

    Returns: U - (M, K) array of unit vectors, all components >= 0
    """
    V     = np.abs(np.random.randn(M, K))     # (M, K) — raw positive vectors
    norms = np.linalg.norm(V, axis=1, keepdims=True)     # norms of each vector
    return V / (norms + 1e-12)

# ORIGINAL NOTEBOOK CELL 8 (zero-based)
def shape_discovery(S1, alpha, M, delta_deg=10):
    """
    For each of M directions u_m on the positive unit sphere:
      1. Find calibration points in S1 whose direction is within delta_deg of u_m
      2. Compute the (1-alpha) quantile of their magnitudes → q_tilde[m]

    - S1: (N1, K) calibration scores for shape discovery

    Returns
    - U: (M, K) direction vectors
    - q_tilde: (M,)  local radius quantile per direction
    """

    K    = S1.shape[1]
    U    = sample_positive_sphere(M, K)      # (M, K)

    # Normalizing S1 rows to get their directions and magnitudes
    mags = np.linalg.norm(S1, axis=1)          # magnitudes of S1 points
    dirs = S1 / (mags[:, None] + 1e-12)        # directions of the S1 points

    cos_thresh = np.cos(np.radians(delta_deg))     # threshold for being close to u_m

    q_tilde = np.zeros(M)
    for m in range(M):
        # Cosine similarity between each S1 point's direction and u_m
        sims = dirs @ U[m]          # cosine similarity of each S1 point with direction u_m
        mask = sims >= cos_thresh   # points that are within delta_deg of u_m

        if mask.sum() < 5:
            # If there are very few points near this direction - fall back to global quantile
            q_tilde[m] = np.quantile(mags, 1 - alpha)
        else:
            q_tilde[m] = np.quantile(mags[mask], 1 - alpha)

    return U, q_tilde

# ORIGINAL NOTEBOOK CELL 10 (zero-based)
def size_scaling(S2, U, q_tilde, alpha):
    """
    Finds t_hat: the global scale factor that gives the envelope
    exactly (1-alpha) coverage on the held-out set S2.

    For each S2 point s, we compute:
        tau(s) = min over m { max over k ( s[k] / boundary[m,k] ) }
    This is the smallest t such that s lies inside at least one sector
    of the envelope scaled by t.

    t_hat = (1-alpha) quantile of tau scores over S2.
    """

    # Boundary point for each direction: b_m = u_m * q_tilde[m]
    boundary     = U * q_tilde[:, None]             # calculating the boundary points for each region
    
    # For each S2 point and each sector m, we need to find out how much would we need to scale the boundary to just reach s?
    ratios       = S2[:, None, :] / (boundary[None, :, :] + 1e-12)         # ratio[i, m, k] = S2[i,k] / boundary[m,k]
    t_per_sector = ratios.max(axis=2)                         # max over k to get t for each sector m on point i
    tau_scores   = t_per_sector.min(axis=1)                   # min over m to get t*(si)      # min over m to get t*(si)


    n2    = len(tau_scores)
    idx   = int(np.ceil((n2 + 1) * (1 - alpha))) - 1        # index ffor 1-alpha quantile
    t_hat = np.sort(tau_scores)[np.clip(idx, 0, n2 - 1)]

    return t_hat, tau_scores

# ORIGINAL NOTEBOOK CELL 12 (zero-based)
def is_in_region(scores, U, q_tilde, t_hat):
    """
    Returns a boolean array: True if the score vector lies inside the final scaled envelope.

    We consider that a point is inside if at least one sector m covers it, i.e.
    ALL components of the score are <= the scaled boundary of that sector.
    """
    q_final  = q_tilde * t_hat      # scaling the quantiles to get the final boundary points
    boundary = U * q_final[:, None]  
    # A point is inside if any sector m covers it (all K dims ≤ boundary)                                
    inside   = np.any(np.all(scores[:, None, :] <= boundary[None, :, :], axis=2), axis=1)
    
    return inside

# ORIGINAL NOTEBOOK CELL 14 (zero-based)
def evaluate_coverage(S2, U, q_tilde, t_hat):
    """
    Empirical coverage on the size-scaling set S2.
    Should be >= (1 - alpha)
    """
    return is_in_region(S2, U, q_tilde, t_hat).mean()

# ORIGINAL NOTEBOOK CELL 16 (zero-based)
np.random.seed(42)
alpha = 0.1
M     = 200

print(f"{'K':>4} | {'t_hat':>8} | {'coverage':>10}")
print("-" * 30)

for K in [2, 5, 10, 20]:
    scores = simulate_scores(n_phages=1000, K=K, correlation=0.3)
    S1, S2 = split_data(scores, split_ratio=0.5)
    U, q_tilde = shape_discovery(S1, alpha, M, delta_deg=15)
    t_hat, _ = size_scaling(S2, U, q_tilde, alpha)
    coverage = evaluate_coverage(S2, U, q_tilde, t_hat)

    print(f"{K:>4} | {t_hat:>8.3f} | {coverage:>10.3f}")

# ORIGINAL NOTEBOOK CELL 17 (zero-based)
