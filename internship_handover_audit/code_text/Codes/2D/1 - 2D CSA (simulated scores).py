
# ORIGINAL NOTEBOOK CELL 1 (zero-based)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect

# ORIGINAL NOTEBOOK CELL 2 (zero-based)
np.random.seed(0)

# ORIGINAL NOTEBOOK CELL 5 (zero-based)
def simulate_scores(n_phages, mean1, mean2, cov1, cov2, correlation):
    """
    Simulates the error scores (1-p) for two models with specified strengths and correlation.
    """
    mean = [mean1, mean2]        # Mean error scores for both models
    cov = [[cov1, correlation], [correlation, cov2]]  # Covariance matrix to introduce specified correlation
    # scores = np.abs(np.random.multivariate_normal(mean, cov, n_phages))  # Simulate scores and take absolute to ensure they are positive
    scores = np.abs(np.random.multivariate_normal(mean, cov, n_phages))
    scores = np.clip(scores, 0, 1)
    return scores

# ORIGINAL NOTEBOOK CELL 7 (zero-based)
def split_calibration_data(scores, split_ratio):
    """
    Splits the calibration data into two sets based on the split ratio.
    """
    n = len(scores)
    indices = np.arange(n)
    np.random.shuffle(indices)  

    limit = int(n * split_ratio)     # Number of samples for the shape set
    shape_set = scores[indices[:limit]]
    size_set = scores[indices[limit:]]
    return shape_set, size_set

# ORIGINAL NOTEBOOK CELL 10 (zero-based)
def shape_discovery(shape_set_scores, alpha, M):
 
    # Directions (Angles 0 to 90 degrees)
    angles = np.linspace(0, np.pi/2, M)
    u_directions = np.vstack([np.cos(angles), np.sin(angles)]).T

    # Projecting the scores from the 'shape' sets onto the directions
    projections = shape_set_scores @ u_directions.T
    
    # We are searcging for the largest beta such that the intersection of half-planes defined by the quantiles has coverage >= 1-alpha
    # We will search from small beta (wide region) to large beta (tight region)
    beta_candidates = np.linspace(alpha/M, alpha, 500)
    
    beta_star = alpha/M     # Default to conservative Bonferroni
    q_tilde = np.quantile(projections, 1 - beta_star, axis=0)

    for beta in beta_candidates:
        q_beta = np.quantile(projections, 1 - beta, axis=0)
        
        # Intersection of Half-Planes
        inside = np.all(projections <= q_beta, axis=1)
        coverage = np.mean(inside)      # Proportion of samples inside the intersection region defined by q_beta
        
        if coverage >= 1 - alpha:
            beta_star = beta
            q_tilde = q_beta
        else:
            break 

    return u_directions, q_tilde, beta_star

# ORIGINAL NOTEBOOK CELL 12 (zero-based)
def size_scaling(size_set_scores, u_directions, q_tilde, alpha):
    """
    Calculates t_hat using the Finite-Sample Rank Lemma.
    """

    # Projecting the scores from the 'size' sets
    projections = size_set_scores @ u_directions.T
    
    # Calculating the non-conformity scores (Maximum Ratio)
    # tau tells us how much we must scale the shape to include the point
    ratios = projections / (q_tilde + 1e-12)
    tau_scores = np.max(ratios, axis=1)

    # Finite-Sample Quantile Calculation
    # According to the Quantile Lemma: t = sorted_tau[ceil((n+1)(1-alpha))]
    n2 = len(tau_scores)
    sorted_tau = np.sort(tau_scores)
    
    # Calculating the exact index for the quantile
    idx = int(np.ceil((n2 + 1) * (1 - alpha))) - 1
    
    # Safety clip for small datasets
    idx = np.clip(idx, 0, n2 - 1)
    t_hat = sorted_tau[idx]

    return t_hat, tau_scores

# ORIGINAL NOTEBOOK CELL 14 (zero-based)
def plot_csa_steps(scores, shape_set_scores, size_set_scores, u_directions, q_tilde, t_hat, alpha):
    fig,axs = plt.subplots(1,3, figsize=(18,5))

    # Plot 1 - the directions
    axs[0].scatter(shape_set_scores[:, 0], shape_set_scores[:, 1], color='blue', alpha=0.3, label='Shape Set Scores')


    for u in u_directions[:: 10]:    # Plotting every 10th direction for clarity
        axs[0].quiver(0, 0, u[0], u[1], scale = 3, color='red', alpha=0.5)

    axs[0].set_title('Directional Projections (u)')
    axs[0].set_xlabel('Model 1 Error Score')
    axs[0].set_ylabel('Model 2 Error Score')
    axs[0].legend()
    # axs[0].set_xlim(0, 0.5)
    # axs[0].set_ylim(0, 0.5) 


    # Plot 2 - half space intersection (the shape)
    x_range = np.linspace(0, 1, 100)
    y_range = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x_range, y_range)
    grid = np.vstack([X.ravel(), Y.ravel()]).T

    # Checking which points in the grid are inside the intersection of half spaces defined by q_tilde
    inside_shape = np.all((u_directions @ grid.T).T <= q_tilde, axis=1).reshape(X.shape)

    axs[1].scatter(shape_set_scores[:, 0], shape_set_scores[:, 1], color='blue', alpha=0.3, label='Shape Set Scores')
    axs[1].contourf(X, Y, inside_shape, levels=[0.5,1], colors='blue', alpha=0.3)
    axs[1].contour(X, Y, inside_shape, levels=[0.5], colors='blue', linewidths=2)
    axs[1].set_title('Intersection of Half Spaces (Quantile Envelope)')
    axs[1].set_xlabel('Model 1 Error Score')
    axs[1].set_ylabel('Model 2 Error Score')
    axs[1].legend()


    # Plot 3 - size scaling (zooming in or out) to get the final region
    q_final = q_tilde * t_hat  # Final quantiles after scaling
    inside_final_shape = np.all((u_directions @ grid.T).T <= q_final, axis=1).reshape(X.shape)

    axs[2].scatter(size_set_scores[:, 0], size_set_scores[:, 1], color='green', alpha=0.3, label='Size Set Scores')
    axs[2].scatter(shape_set_scores[:, 0], shape_set_scores[:, 1], color='blue', alpha=0.3, label='Shape Set Scores')

    # plotting the original shape dashed and final shape with a solid line
    axs[2].contour(X, Y, inside_shape, levels=[0.5], colors='blue', linestyles='--')
    axs[2].contourf(X, Y, inside_final_shape, levels=[0.5,1], colors = 'green', alpha=0.3)
    axs[2].contour(X, Y, inside_final_shape, levels=[0.5], colors='green', linewidths=2)
    axs[2].set_title(f"Final Calibrated Region (Scaled Quantile Envelope)(t = {t_hat:.2f})")
    axs[2].set_xlabel('Model 1 Error Score')
    axs[2].set_ylabel('Model 2 Error Score')
    axs[2].legend()
    plt.tight_layout()
    plt.show()

# ORIGINAL NOTEBOOK CELL 16 (zero-based)
alpha = 0.1  # 90% coverage guarantee
M = 500  # Number of directions for half spaces

# simulating the data 
raw_scores = simulate_scores(n_phages = 400, mean1 = 0.15, mean2 = 0.15, cov1 = 0.01, cov2 = 0.01, correlation = -0.0085)

# splitting the data into shape and size sets
S1, S2 = split_calibration_data(raw_scores, split_ratio=0.5)

# finding the shape (geometry) of the acceptance region using half spaces
u_directions, q_tilde, beta_star = shape_discovery(S1, alpha, M)

# finding the scaling factor 't' to guarantee the 1-alpha coverage probability
t_hat, tau_scores = size_scaling(S2, u_directions, q_tilde, alpha)

# plotting the steps of the CSA method
plot_csa_steps(raw_scores, S1, S2, u_directions, q_tilde, t_hat, alpha)