
# ORIGINAL NOTEBOOK CELL 1 (zero-based)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

# ORIGINAL NOTEBOOK CELL 3 (zero-based)
def simulate_scores(n_phages, K, mean=None, correlation=0.0, variance=0.02):
    """
    Simulates K-dimensional scores for phages that infect a host.

    Parameters
    ----------
    n_phages    : number of phages to simulate
    K           : number of weak learners (protein functions)
    mean        : mean vector of length K (default: 0.3 for all dims)

    Returns
    -------
    scores : (n_phages, K) array clipped to [0, 1]
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
    cut = int(n * split_ratio)
    return scores[idx[:cut]], scores[idx[cut:]]

# ORIGINAL NOTEBOOK CELL 7 (zero-based)
def sample_positive_sphere(M, K):
    """
    Samples M directions uniformly on the positive orthant of S^{K-1}.

    Method :
      Draw V ~ N(0, I_{K×K}), take absolute values, normalise.

    Parameters
    ----------
    M : number of directions
    K : score dimensionality

    Returns
    -------
    U : (M, K) array of unit vectors, all components >= 0
    """
    V = np.abs(np.random.randn(M, K))          # (M, K) positive vectors
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    return V / (norms + 1e-12)

# ORIGINAL NOTEBOOK CELL 9 (zero-based)
def shape_discovery(S1, alpha, M, eps=1e-3, max_iter=50):
    """
    Implements Stage 1 of Algorithm 1 from the CSA paper.

    For M projection directions {u_m} on the positive unit sphere:
      1. Project all S1 points: proj[i,m] = u_m^T s_i
      2. Binary search over β to find the minimum β* such that the
         intersection of halfplanes H(u_m, q̃_m(β)) covers (1-α) of S1
      3. Return the directions U and per-direction quantiles q̃

    Parameters
    ----------
    S1       : (N1, K) calibration scores for shape discovery
    alpha    : miscoverage level (target coverage = 1 - alpha)
    M        : number of projection directions
    eps      : tolerance for binary search convergence
    max_iter : maximum binary search iterations

    Returns
    -------
    U       : (M, K) projection directions
    q_tilde : (M,)  per-direction quantile thresholds at β*
    beta_star : the β* found by binary search
    """
    K = S1.shape[1]
    U = sample_positive_sphere(M, K)     # (M, K)

    # Computing all projections at once: proj[i, m] = u_m^T s_i
    # Shape: (N1, M)
    proj = S1 @ U.T

    # Binary search bounds for β (paper: β ∈ [α/M, α])
    beta_lo = alpha / M
    beta_hi = alpha

    q_tilde = None
    beta_star = None

    for _ in range(max_iter):
        beta = (beta_lo + beta_hi) / 2

        # Per-direction (1-β) quantile
        # q_tilde[m] = quantile of {u_m^T s_i for s_i in S1} at level (1-β)
        q_tilde = np.quantile(proj, 1 - beta, axis=0)   # (M,)

        # A point is inside the envelope if ALL projections are within threshold
        # inside[i] = True iff u_m^T s_i <= q_tilde[m] for all m
        inside = np.all(proj <= q_tilde[None, :], axis=1)  # (N1,)
        coverage = inside.mean()

        # Binary search: shrink β until coverage ≈ 1 - alpha
        if coverage > 1 - alpha:
            beta_lo = beta      # envelope too large → increase β to shrink it
        else:
            beta_hi = beta      # envelope too small → decrease β to grow it

        if (beta_hi - beta_lo) < eps:
            beta_star = beta
            break

    if beta_star is None:
        beta_star = (beta_lo + beta_hi) / 2

    # Recomputing q_tilde at final β*
    q_tilde = np.quantile(proj, 1 - beta_star, axis=0)   # (M,)

    return U, q_tilde, beta_star

# ORIGINAL NOTEBOOK CELL 11 (zero-based)
def size_scaling(S2, U, q_tilde, alpha):
    """
    Implements Stage 2 of Algorithm 1 from the CSA paper.

    For each S2 point s:
      τ(s) = max_m ( u_m^T s / q̃_m )
    This is the smallest t such that s is inside the envelope scaled by t.

    t̂ = ⌈(|S2|+1)(1-α)⌉-th quantile of τ scores over S2.

    Parameters
    ----------
    S2      : (N2, K) calibration scores for size scaling
    U       : (M, K) projection directions from shape_discovery
    q_tilde : (M,)   per-direction thresholds from shape_discovery
    alpha   : miscoverage level

    Returns
    -------
    t_hat      : scalar scale factor
    tau_scores : (N2,) τ score for each S2 point
    """
    # proj_S2[i, m] = u_m^T s_i  for s_i in S2
    proj_S2 = S2 @ U.T                         # (N2, M)

    # tau[i] = max_m ( u_m^T s_i / q̃_m )
    tau_scores = (proj_S2 / (q_tilde[None, :] + 1e-12)).max(axis=1)   # (N2,)

    # Conformal quantile: ⌈(N2+1)(1-α)⌉ / N2
    n2 = len(tau_scores)
    idx = int(np.ceil((n2 + 1) * (1 - alpha))) - 1
    t_hat = np.sort(tau_scores)[np.clip(idx, 0, n2 - 1)]

    return t_hat, tau_scores

# ORIGINAL NOTEBOOK CELL 13 (zero-based)
def is_in_envelope(scores, U, q_tilde, t_hat):
    """
    Returns True for each score vector inside the final scaled envelope.

    A point s is inside iff  u_m^T s <= t̂ · q̃_m  for ALL m.
    Equivalently: τ(s) = max_m(u_m^T s / q̃_m) <= t̂

    This is the intersection of M halfplanes:
        Q̂ = ∩_m  H(u_m, t̂ · q̃_m)
    """
    q_final = q_tilde * t_hat                   # (M,) scaled thresholds
    proj = scores @ U.T                      # (N, M)
    return np.all(proj <= q_final[None, :], axis=1)   # (N,)

# ORIGINAL NOTEBOOK CELL 14 (zero-based)
def evaluate_coverage(S2, U, q_tilde, t_hat):
    """
    Empirical coverage on S2. Should be >= (1 - alpha) by construction.
    """
    return is_in_envelope(S2, U, q_tilde, t_hat).mean()

# ORIGINAL NOTEBOOK CELL 16 (zero-based)
np.random.seed(42)
alpha = 0.1
M = 200

print(f"{'K':>4} | {'β*':>8} | {'t̂':>8} | {'coverage':>10}")
print("-" * 40)

for K in [2, 5, 10, 20]:
    scores = simulate_scores(n_phages=1000, K=K, correlation=0.3)
    S1, S2 = split_data(scores, split_ratio=0.5)
    U, q_tilde, b_star = shape_discovery(S1, alpha, M)
    t_hat, _ = size_scaling(S2, U, q_tilde, alpha)
    coverage = evaluate_coverage(S2, U, q_tilde, t_hat)

    print(f"{K:>4} | {b_star:>8.5f} | {t_hat:>8.3f} | {coverage:>10.3f}")

# ORIGINAL NOTEBOOK CELL 18 (zero-based)
def build_envelope(infection_scores, alpha, M=200):
    """
    Full CSA calibration pipeline for one host.

    Parameters
    ----------
    infection_scores : (N, K) score vectors of phages known to infect this host
    alpha            : miscoverage level
    M                : number of projection directions

    Returns
    -------
    dict with keys: U, q_tilde, t_hat, beta_star
    """
    S1, S2 = split_data(infection_scores)
    U, q_tilde, b_star = shape_discovery(S1, alpha, M)
    t_hat, _ = size_scaling(S2, U, q_tilde, alpha)
    return {"U": U, "q_tilde": q_tilde, "t_hat": t_hat, "beta_star": b_star}

# ORIGINAL NOTEBOOK CELL 19 (zero-based)
def predict_one_host(score_vector, envelope):
    """
    Predicts whether a phage infects a given host using the CSA envelope.

    Returns
    -------
    outcome : str  ("infects" or "does not infect")
    s_in    : bool
    """
    s_in = bool(is_in_envelope(score_vector[None, :], envelope["U"], 
                               envelope["q_tilde"], envelope["t_hat"])[0])
    outcome = "infects" if s_in else "does not infect"
    return outcome, s_in

# ORIGINAL NOTEBOOK CELL 20 (zero-based)
def predict_phage_hosts(score_per_host, envelopes, hosts):
    """
    Produces the prediction set for a single test phage across all hosts.

    A host is included in the prediction set if the phage's score vector
    falls inside that host's calibrated CSA envelope.

    Returns
    -------
    prediction_set : list of host names
    results        : dict of per-host outcomes
    """
    prediction_set = []
    results = {}
    for host in hosts:
        outcome, s_in = predict_one_host(score_per_host[host], envelopes[host])
        results[host] = {"outcome": outcome, "s_in": s_in}
        if s_in:
            prediction_set.append(host)
    return prediction_set, results

# ORIGINAL NOTEBOOK CELL 21 (zero-based)
np.random.seed(42)

hosts = [f"Host_{chr(65+i)}" for i in range(6)]
true_host = "Host_B"
alpha = 0.1
K = 5
M = 200
N_cal = 500

# Building one envelope per host from simulated infector scores
envelopes = {}
for host in hosts:
    host_mean = np.random.uniform(0.1, 0.5, K)
    infectors = simulate_scores(N_cal, K, mean=host_mean, correlation=0.3)
    envelopes[host] = build_envelope(infectors, alpha, M=M)

# Simulating score vectors for one test phage
test_scores = {}
for host in hosts:
    if host == true_host:
        test_scores[host] = np.random.uniform(0.1, 0.4, K)   # looks like a true infector
    else:
        test_scores[host] = np.random.uniform(0.6, 0.9, K)   # unlikely to be inside envelope

pred_set, results = predict_phage_hosts(test_scores, envelopes, hosts)

print("=" * 60)
print(f"CSA (PAPER) METHOD | alpha={alpha} | true host: {true_host}")
print("=" * 60)
print(f"{'Host':<15} | {'in env':>6} | {'β*':>8} | {'Outcome'}")
print("-" * 55)
for host in hosts:
    r      = results[host]
    s_txt  = "YES" if r['s_in'] else "NO"
    b_star = envelopes[host]['beta_star']
    marker = "  <-- TRUE HOST" if host == true_host else ""
    print(f"{host:<15} | {s_txt:>6} | {b_star:>8.5f} | {r['outcome']}{marker}")
print("-" * 55)
print(f"PREDICTION SET : {pred_set}")
print(f"SET SIZE       : {len(pred_set)} / {len(hosts)} hosts")
print(f"TRUE HOST COVERED: {'YES' if true_host in pred_set else 'NO'}")