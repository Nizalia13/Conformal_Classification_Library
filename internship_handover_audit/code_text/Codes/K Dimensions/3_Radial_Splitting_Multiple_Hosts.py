
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
def sample_positive_sphere(M, K):
    """
    Samples M directions uniformly on the positive orthant of unit sphere in K dimensions
    """
    V     = np.abs(np.random.randn(M, K))
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    return V / (norms + 1e-12)

# ORIGINAL NOTEBOOK CELL 8 (zero-based)
def shape_discovery(S1, alpha, M, delta_deg=10):
    """
    Finds M projection direction and their corresponding quantiles to build the envelope
    For each direction, we only consider the scores which are within delta_deg of that direction
    """

    K = S1.shape[1]      # number of score dimensions
    U    = sample_positive_sphere(M, K)        # M random directions on K dimensions (positive)
    mags = np.linalg.norm(S1, axis=1)          # magnitudes of the score vectors
    dirs = S1 / (mags[:, None] + 1e-12)        # threshold for selecting scores

    cos_thresh = np.cos(np.radians(delta_deg))
    q_tilde    = np.zeros(M)

    for m in range(M):
        sims = dirs @ U[m]       # closeness of each score direction to the m-th random direction
        mask = sims >= cos_thresh       # selecting scores that are close enough
        if mask.sum() < 5:
            q_tilde[m] = np.quantile(mags, 1 - alpha)
        else:
            q_tilde[m] = np.quantile(mags[mask], 1 - alpha)

    return U, q_tilde

# ORIGINAL NOTEBOOK CELL 10 (zero-based)
def size_scaling(S2, U, q_tilde, alpha):
    """
    Finds t_hat so the envelope defined by U and q_tilde has 1-alpha coverage on S2
    tau_scores = min over m of [ max over k of s[k] / boundary[m,k] ]
    t_hat  = (1-alpha) quantile of tau scores over S2]
    """
    boundary = U * q_tilde[:, None]      # the envelope boundary in each direction
    ratios = S2[:, None, :] / (boundary[None, :, :] + 1e-12)      # how much each score exceeds the boundary in each direction
    t_per_sector = ratios.max(axis=2)    # for each score and each sector  we take the max ratio
    tau_scores = t_per_sector.min(axis=1)   # for each score we take the min ratio across sectors
    
    n2 = len(tau_scores)     
    idx = int(np.ceil((n2 + 1) * (1 - alpha))) - 1     # index for 1-alpha quantile
    
    t_hat = np.sort(tau_scores)[np.clip(idx, 0, n2 - 1)]    # 1-alpha quantile of tau scores
    return t_hat, tau_scores

# ORIGINAL NOTEBOOK CELL 12 (zero-based)
def is_in_region(score_vector, U, q_tilde, t_hat):
    """
    Checking if a score_vector is inside the scaled envelope
    a point is inside if any of the sectors contains it (all components of scores <= scaled boundary for that sector)
    points closer to the origin are alwaya inside
    """
    q_final  = q_tilde * t_hat      # final quantiles after scaling
    boundary = U * q_final[:, None]       # envelope boundary in each direction
    inside = np.any(np.all(score_vector[:, None, :] <= boundary[None, :, :], axis=2),axis=1)
    return inside

# ORIGINAL NOTEBOOK CELL 13 (zero-based)
def build_envelope(infection_scores, alpha, M=200, delta_deg=15):
    """ 
    Builds an envelope for  the given host
    """
    S1, S2 = split_data(infection_scores)     
    U, q_tilde = shape_discovery(S1, alpha, M, delta_deg)   
    t_hat, _ = size_scaling(S2, U, q_tilde, alpha)
    return {"U": U, "q_tilde": q_tilde, "t_hat": t_hat}  

# ORIGINAL NOTEBOOK CELL 15 (zero-based)
INFECTS     = "infects"
NOT_INFECTS = "does not infect"

# ORIGINAL NOTEBOOK CELL 16 (zero-based)
def predict_one_host(score_1, envelope):
    """
    Predicts whether a phage infects a given host.
    """
    s1_in = bool(is_in_region(score_1[None, :], envelope["U"], envelope["q_tilde"], envelope["t_hat"])[0])
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

hosts     = [f"Host_{chr(65+i)}" for i in range(6)]
true_host = "Host_B"
alpha     = 0.1
K         = 5
M         = 200
N_cal     = 500   

# ORIGINAL NOTEBOOK CELL 20 (zero-based)
# Building one envelope per host using the simulated scores
# Each host has a different characteristic mean in score space
envelopes = {}
for host in hosts:
    host_mean = np.random.uniform(0.1, 0.5, K)
    infectors = simulate_scores(N_cal, K, mean=host_mean, correlation=0.3)
    envelopes[host] = build_envelope(infectors, alpha, M=M, delta_deg=15)

# ORIGINAL NOTEBOOK CELL 21 (zero-based)
# Simulating the score vectors for one test phage
# This phage actually infects true_host, so its score for that host should
# look like a true infector; scores for other hosts should look different
test_scores = {}
for host in hosts:
    env = envelopes[host]
    if host == true_host:
        # Drawing a score that looks like a true infector for this host
        test_scores[host] = np.random.uniform(0.1, 0.4, K)
    else:
        # Drawing a score that is unlikely to be inside this host's envelope
        test_scores[host] = np.random.uniform(0.6, 0.9, K)

# ORIGINAL NOTEBOOK CELL 22 (zero-based)
pred_set, results = predict_phage_hosts(test_scores, envelopes, hosts)

print("=" * 60)
print(f"RADIAL METHOD | alpha={alpha} | true host: {true_host}")
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