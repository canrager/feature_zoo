# %% [markdown]
# # Characterization of manifolds in activation space.
#
# This notebook measures the quality of learned representations in LLM embedding space
# by dimensionality, smoothness and topology preservation.

# %%
# Notebook setup
%cd /home/can/feature_zoo/
%load_ext autoreload
%autoreload 2

# %%
# Initialize Experiment config (defines dataset, model, sae)

# Choose one

override = ["llm=llama3.1-8b-base-layer15", "data=integers100"]
# override = ["llm=llama3.1-8b-base-layer15", "data=years100"]
# override = ["llm=llama3.1-8b-base-layer15", "data=emotions200"]
# override = ["llm=gpt2-layer6", "data=colors101"]
# override = ["llm=gpt2-layer6", "data=integers100"]
# override = ["llm=gpt2-layer6", "data=integers1000"]
# override = ["llm=gpt2-layer6", "data=integers1000step5"]
# override = ["llm=llama3.1-8b-base-layer15", "data=integers1000step5"]
# override = ["llm=llama3.1-8b-base-layer15", "data=years100"]
# override = ["llm=llama3.1-8b-base-layer15", "data=integers500"]
# override = ["llm=llama3.1-8b-base-layer15", "data=integers1000"]
# override = ["llm=llama3.1-8b-base-layer15", "data=days7"]
# override = ["llm=llama3.1-8b-base-layer15", "data=months12"]
# override = ["llm=llama3.1-8b-base-layer15", "data=colors101"]
# override = ["llm=llama3.1-8b-base-layer15", "data=colors101_lch"]
# override = ["llm=llama3.1-8b-base-layer15", "data=colors195_lch"]
# override = ["llm=llama3.1-8b-base-embedding", "data=integers100", "exp=sum"]
# override = ["llm=olmo3-32b-base-layer32", "data=integers100"]
# override = ["llm=gpt2-layer6", "data=years100"]
# override = ["llm=gpt2-layer6", "data=colors6"]
# override = ["llm=gpt2-embedding", "data=integers100"]

# %% 
# Imports

from src.config import load_config
from src.cache_llm import load_short_trajectory_acts
from src.random_baseline import ensure_random_baseline_exists
import torch as th
import numpy as np
import matplotlib.pyplot as plt

def to_numpy(t: th.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()

def load_concept_labels(cfg):
    with open(f"data/texts/{cfg.filter.data}.txt") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

# %%
# Cache data for both original and random baselines

def load_experiment_data(override):
    th.cuda.empty_cache()

    cfg = load_config(overrides=override)
    model_name = cfg.llm.name
    data_name = cfg.data.name

    # print(f"Loading {model_name} - {data_name} (original)...")
    original_dict = load_short_trajectory_acts(cfg, force_recompute=False)

    # print(f"Loading {model_name} - random baseline...")
    random_cfg = ensure_random_baseline_exists(cfg)
    random_dict = load_short_trajectory_acts(random_cfg, force_recompute=False)

    def process_return_dict(return_dict):
        if isinstance(return_dict["llm_BCD"], th.Tensor):
            llm_BCD = to_numpy(return_dict["llm_BCD"])
            input_ids_BT = return_dict["input_ids_BT"].cpu() if isinstance(return_dict["input_ids_BT"], th.Tensor) else return_dict["input_ids_BT"]
        else:
            llm_BCD = return_dict["llm_BCD"]
            input_ids_BT = return_dict["input_ids_BT"]
        return {
            "elements_C": return_dict["elements_C"],
            "labels_b": return_dict["labels_b"],
            "texts_b": return_dict["texts_b"],
            "input_ids_BT": input_ids_BT,
            "llm_BCD": llm_BCD,
        }

    experiment_data = {
        "original": process_return_dict(original_dict),
        "random": process_return_dict(random_dict),
        "data_name": data_name,
    }

    del original_dict, random_dict
    th.cuda.empty_cache()

    # print(f"{model_name} original llm_BCD shape: {experiment_data['original']['llm_BCD'].shape}")
    # print(f"{model_name} random llm_BCD shape: {experiment_data['random']['llm_BCD'].shape}")
    return experiment_data, model_name, data_name

experiment_data, model_name, data_name = load_experiment_data(override)

#%% [markdown]
# # Effective dimensionality

#%% [markdown]
# ## PCA
# SVD on the full dataset can take long. Don't worry we will only do this once!
import einops

def compute_svd(experiment_data):
    llm_og_BCD = experiment_data["original"]["llm_BCD"]
    llm_rd_BCD = experiment_data["random"]["llm_BCD"]

    # Center
    llm_og_BCD = llm_og_BCD - np.mean(llm_og_BCD, axis=-1, keepdims=True)
    llm_rd_BCD = llm_rd_BCD - np.mean(llm_rd_BCD, axis=-1, keepdims=True)

    # Reshape b = B x C
    llm_og_bD = einops.rearrange(llm_og_BCD, "B C D -> (B C) D")
    llm_rd_bD = einops.rearrange(llm_rd_BCD, "B C D -> (B C) D")

    # SVD on centered data is equal to PCA
    U_og_bD, S_og_D, Vh_og_DD = np.linalg.svd(llm_og_bD, full_matrices=False)
    U_rd_bD, S_rd_D, Vh_rd_DD = np.linalg.svd(llm_rd_bD, full_matrices=False)

    return U_og_bD, S_og_D, Vh_og_DD, U_rd_bD, S_rd_D, Vh_rd_DD

U_og_bD, S_og_D, Vh_og_DD, U_rd_bD, S_rd_D, Vh_rd_DD = compute_svd(experiment_data)
B, C, D = experiment_data["original"]["llm_BCD"].shape
# U_og_BCD = einops.rearrange(U_og_bD, "(B C) R -> B C R", B=B, C=C)
# U_rd_BCD = einops.rearrange(U_rd_bD, "(B C) R -> B C R", B=B, C=C)

#%%
def compute_relative_variance(S_og_D, S_rd_D):
    rel_var_og = S_og_D**2 / np.sum(S_og_D**2)
    rel_var_rd = S_rd_D**2 / np.sum(S_rd_D**2)
    cum_var_og = np.cumsum(rel_var_og)
    cum_var_rd = np.cumsum(rel_var_rd)
    return rel_var_og, rel_var_rd, cum_var_og, cum_var_rd

rel_var_og, rel_var_rd, cum_var_og, cum_var_rd = compute_relative_variance(S_og_D, S_rd_D)

#%%
def plot_spectral_analysis(rel_var_og, rel_var_rd, cum_var_og, cum_var_rd, model_name, data_name, comp_max=60):
    x = np.arange(comp_max)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{model_name} - {data_name}: Spectral Analysis", fontsize=14)

    # Relative variance
    axes[0, 0].semilogy(x, rel_var_og[:comp_max], 'o-', c='tab:blue', markersize=4, label=f'Original ({data_name})')
    axes[0, 0].semilogy(x, rel_var_rd[:comp_max], 's-', c='tab:orange', markersize=4, label='Random')
    axes[0, 0].set_xlabel("PCA component")
    axes[0, 0].set_ylabel("Relative variance")
    axes[0, 0].set_title("Eigenspectrum")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Cumulative relative variance
    axes[0, 1].plot(x, cum_var_og[:comp_max], 'o-', c='tab:blue', markersize=4, label=f'Original ({data_name})')
    axes[0, 1].plot(x, cum_var_rd[:comp_max], 's-', c='tab:orange', markersize=4, label='Random')
    axes[0, 1].set_xlabel("PCA component")
    axes[0, 1].set_ylabel("Cumulative relative variance")
    axes[0, 1].set_title("Cumulative Variance")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Decay rate (consecutive difference in relative variance)
    decay_og = rel_var_og[:-1] - rel_var_og[1:]
    decay_rd = rel_var_rd[:-1] - rel_var_rd[1:]

    # Fit power law to random decay: f(x) = a * x^(-b)
    from scipy.optimize import curve_fit
    def power_law(x, a, b):
        return a * x ** (-b)
    x_fit = np.arange(1, comp_max)  # start at 1 to avoid 0^(-b)
    popt, _ = curve_fit(power_law, x_fit, decay_rd[1:comp_max], p0=[decay_rd[1], 1.0], maxfev=10000)
    a_fit, b_fit = popt

    axes[1, 0].semilogy(x[:-1], decay_og[:comp_max-1], 'o-', c='tab:blue', markersize=4, label=f'Original ({data_name})')
    axes[1, 0].semilogy(x[:-1], decay_rd[:comp_max-1], 's-', c='tab:orange', markersize=4, label='Random')
    axes[1, 0].semilogy(x_fit, power_law(x_fit, *popt), '--', c='tab:red', linewidth=2,
                         label=f'Fit Random: {a_fit:.2e}·x^(-{b_fit:.3f})')
    axes[1, 0].set_xlabel("PCA component")
    axes[1, 0].set_ylabel("Variance decay rate")
    axes[1, 0].set_title("Decay Rate")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Difference (original - random)
    diff = rel_var_og[:comp_max] - rel_var_rd[:comp_max]
    axes[1, 1].plot(x, diff, 'o-', c='tab:green', markersize=4)
    axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel("PCA component")
    axes[1, 1].set_ylabel("Δ Relative variance")
    axes[1, 1].set_title("Difference (Orig - Rand)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_spectral_analysis(rel_var_og, rel_var_rd, cum_var_og, cum_var_rd, model_name, data_name)

#%% [markdown]
# ## Tortuosity (DM — Distance Metric)

#%%
# Tortuosity

def _path_length(points_CD):
    """Total Euclidean path length along consecutive points."""
    diffs_CD = np.diff(points_CD, axis=0)
    return np.sum(np.linalg.norm(diffs_CD, axis=1))

def tortuosity_dm(points_CD):
    """Distance Metric: ratio of path length to chord length."""
    chord = np.linalg.norm(points_CD[-1] - points_CD[0])
    if chord < 1e-10:
        return float('inf')
    return _path_length(points_CD) / chord

def _tortuosity_per_template_BR(llm_BCD):
    """Compute DM tortuosity per template at each PCA truncation rank.

    Returns shape (B, R) where R is the number of truncation ranks.
    """
    llm_BCD = llm_BCD - np.mean(llm_BCD, axis=-1, keepdims=True)
    B, C, D = llm_BCD.shape
    n_components = min(C, D)
    dm_BR = np.empty((B, n_components))
    for b in range(B):
        llm_CD = llm_BCD[b]
        U_CD, S_D, _ = np.linalg.svd(llm_CD, full_matrices=False)
        for k in range(1, n_components + 1):
            truncated_Ck = U_CD[:, :k] * S_D[:k]
            dm_BR[b, k - 1] = tortuosity_dm(truncated_Ck)
    return dm_BR

def compute_tortuosity_over_rank(experiment_data):
    """Compute per-template DM tortuosity for original and random data."""
    orig_dm_BR = _tortuosity_per_template_BR(experiment_data["original"]["llm_BCD"])
    rand_dm_BR = _tortuosity_per_template_BR(experiment_data["random"]["llm_BCD"])
    n_components = orig_dm_BR.shape[1]
    ranks_R = np.arange(1, n_components + 1)
    return ranks_R, orig_dm_BR, rand_dm_BR

def find_convergence_start(tau, N=10, eps=0.1):
    """Find first rank r where tau remains stable for N consecutive steps.

    Criterion: max absolute change over window [r, r+N] < eps.
    """
    for r in range(len(tau) - N):
        window_changes = [abs(tau[i + 1] - tau[i]) for i in range(r, r + N)]
        if max(window_changes) < eps:
            return r
    return None

ranks_R, orig_dm_BR, rand_dm_BR = compute_tortuosity_over_rank(experiment_data)

orig_mean_R = np.mean(orig_dm_BR, axis=0)
conv_rank = find_convergence_start(orig_mean_R, N=50, eps=0.5)
# print(f"Convergence rank: {ranks_R[conv_rank] if conv_rank is not None else 'not found'}")

#%%
# Plot tortuosity

def plot_tortuosity(ranks_R, orig_dm_BR, rand_dm_BR, conv_rank, model_name, data_name, max_traces=1000):
    fig, ax = plt.subplots(figsize=(7, 4))

    rng = np.random.default_rng(42)

    # Individual template trajectories (faint, random subset))
    orig_idx = rng.choice(orig_dm_BR.shape[0], size=min(max_traces, orig_dm_BR.shape[0]), replace=False)
    rand_idx = rng.choice(rand_dm_BR.shape[0], size=min(max_traces, rand_dm_BR.shape[0]), replace=False)
    for b in orig_idx:
        ax.plot(ranks_R, orig_dm_BR[b], color='tab:blue', alpha=0.15, linewidth=0.8)
    for b in rand_idx:
        ax.plot(ranks_R, rand_dm_BR[b], color='tab:orange', alpha=0.15, linewidth=0.8)

    # Mean trajectories (bold)
    orig_mean_R = np.mean(orig_dm_BR, axis=0)
    rand_mean_R = np.mean(rand_dm_BR, axis=0)
    ax.plot(ranks_R, orig_mean_R, 'o-', markersize=3, color='tab:blue', linewidth=2, label=f"Original ({data_name})")
    ax.plot(ranks_R, rand_mean_R, 's-', markersize=3, color='tab:orange', linewidth=2, label="Random")

    if conv_rank is not None:
        ax.axvline(ranks_R[conv_rank], color='tab:red', linestyle='--', alpha=0.7,
                   label=f"Convergence (rank {ranks_R[conv_rank]})")

    ax.set_xlabel("PCA truncation rank")
    ax.set_ylabel("DM tortuosity")
    ax.set_title(f"{model_name} — {data_name}: Tortuosity (DM)")
    ax.set_ylim((0, 250))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_tortuosity(ranks_R, orig_dm_BR, rand_dm_BR, conv_rank, model_name, data_name)

#%%
# Truncate SVD to pre-convergence ranks for downstream analysis

assert conv_rank is not None, "Convergence rank not found — adjust conv_N/conv_eps or inspect tortuosity plot"

### optional manual override of conv rank
# conv_rank = 10


R = ranks_R[conv_rank]  # number of components to keep
# print(f"Truncating to R={R} components")

U_og_bR = U_og_bD[:, :R]
S_og_R = S_og_D[:R]
Vh_og_RD = Vh_og_DD[:R, :]

U_rd_bR = U_rd_bD[:, :R]
S_rd_R = S_rd_D[:R]
Vh_rd_RD = Vh_rd_DD[:R, :]

U_og_BCR = einops.rearrange(U_og_bR, "(B C) R -> B C R", B=B, C=C)
U_rd_BCR = einops.rearrange(U_rd_bR, "(B C) R -> B C R", B=B, C=C)

llm_og_BCR = U_og_BCR * S_og_R[None, None, :] 
llm_rd_BCR = U_rd_BCR * S_rd_R[None, None, :] 

#%% [markdown]
# ## Gram matrix condition number (max distance / min distance)
# The condition number gives the ratio between the max and min entry of the gram matrix.
# Random should be evenly distributed, nonrandom structure should have some points closer than others thus higher condition number.
# To be robust against ultra low values, we'll take lam min, the rank that explains 99 percent variance.
#%%
# Gram matrix condition number: Intersection in embedding space

def effective_condition_number_B(llm_BCR, energy_threshold=0.99):
    B, C, R = llm_BCR.shape
    cond_B = np.empty(B)
    for b in range(B):
        G = llm_BCR[b].T @ llm_BCR[b]
        eigvals = np.linalg.eigvalsh(G)[::-1]  # descending
        eigvals = np.maximum(eigvals, 0)        # numerical safety
        cumulative = np.cumsum(eigvals) / np.sum(eigvals)
        r_eff = np.searchsorted(cumulative, energy_threshold) + 1
        cond_B[b] = eigvals[0] / eigvals[r_eff - 1]
    return cond_B

cond_og_B = effective_condition_number_B(U_og_BCR)
cond_rd_B = effective_condition_number_B(U_rd_BCR)

fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, max(np.max(cond_og_B), np.max(cond_rd_B)), 50)
ax.hist(cond_og_B, bins=bins, alpha=0.6, color='tab:blue', label=f'Original ({data_name})')
ax.hist(cond_rd_B, bins=bins, alpha=0.6, color='tab:orange', label='Random')
ax.set_xlabel("Gram matrix condition number")
ax.set_ylabel("Count")
ax.set_title(f"{model_name} — {data_name}: Gram Matrix Condition Numbers (R={R})")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


high_cond_thresh = 40
high_cond_mask = cond_og_B > high_cond_thresh
if np.any(high_cond_mask):
    # print(f"\nOriginal templates with condition number > 1e11 ({np.sum(high_cond_mask)}/{B}):")
    texts_b = experiment_data["original"]["texts_b"]  # flat (B*C) despite key name
    labels_b = experiment_data["original"]["labels_b"]
    for idx in np.where(high_cond_mask)[0]:
        template = texts_b[idx * C].replace(labels_b[idx * C], "{}")
        # print(f"  [{idx}] cond={cond_og_B[idx]:.2e}: {template}")
else:
    print(f"\nNo original templates with condition number > {high_cond_thresh}")


#%%

#%% [markdown]
# ## Global Lipschitz constant (max consecutive-point distance) per template

#%%
def global_lipschitz_B(llm_BCR):
    """Max Euclidean distance between consecutive points per template."""
    diffs_BCR = np.diff(llm_BCR, axis=1)  # (B, C-1, R)
    norms_BC = np.linalg.norm(diffs_BCR, axis=2)  # (B, C-1)
    return np.max(norms_BC, axis=1)  # (B,)

lip_og_B = global_lipschitz_B(llm_og_BCR)
lip_rd_B = global_lipschitz_B(llm_rd_BCR)

fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, max(np.max(lip_og_B), np.max(lip_rd_B)), 50)
ax.hist(lip_og_B, bins=bins, alpha=0.6, color='tab:blue', label=f'Original ({data_name})')
ax.hist(lip_rd_B, bins=bins, alpha=0.6, color='tab:orange', label='Random')
ax.set_xlabel("Global Lipschitz constant (max consecutive Δ)")
ax.set_ylabel("Count")
ax.set_title(f"{model_name} — {data_name}: Global Lipschitz Constant per Template (R={R})")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%% [markdown]
# # Local Smoothness

#%% [markdown]
# ## Discrete Jacobian norm along trajectory
# Plot the Euclidean norm of consecutive-point differences (discrete Jacobian)
# per template. Faint individual traces + bold mean, for original vs random.

#%%
def discrete_jacobian_norms_BC(llm_BCR):
    """Euclidean norm of consecutive-point differences per template.

    Returns shape (B, C-1).
    """
    diffs_BCR = np.diff(llm_BCR, axis=1)  # (B, C-1, R)
    return np.linalg.norm(diffs_BCR, axis=2)  # (B, C-1)

def plot_discrete_jacobian(og_norms_BC, rd_norms_BC, model_name, data_name, max_traces=1000):
    C_minus_1 = og_norms_BC.shape[1]
    x = np.arange(C_minus_1)

    fig, ax = plt.subplots(figsize=(7, 4))
    rng = np.random.default_rng(42)

    # Individual traces (faint)
    orig_idx = rng.choice(og_norms_BC.shape[0], size=min(max_traces, og_norms_BC.shape[0]), replace=False)
    rand_idx = rng.choice(rd_norms_BC.shape[0], size=min(max_traces, rd_norms_BC.shape[0]), replace=False)
    for b in orig_idx:
        ax.plot(x, og_norms_BC[b], color='tab:blue', alpha=0.15, linewidth=0.8)
    for b in rand_idx:
        ax.plot(x, rd_norms_BC[b], color='tab:orange', alpha=0.15, linewidth=0.8)

    # Mean trajectories (bold)
    og_mean_C = np.mean(og_norms_BC, axis=0)
    rd_mean_C = np.mean(rd_norms_BC, axis=0)
    ax.plot(x, og_mean_C, 'o-', markersize=3, color='tab:blue', linewidth=2, label=f"Original ({data_name})")
    ax.plot(x, rd_mean_C, 's-', markersize=3, color='tab:orange', linewidth=2, label="Random")

    ax.set_xlabel("Step along trajectory")
    ax.set_ylabel("‖Δ embedding‖")
    ax.set_title(f"{model_name} — {data_name}: Discrete Jacobian Norm (R={R})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

og_norms_BC = discrete_jacobian_norms_BC(llm_og_BCR)
rd_norms_BC = discrete_jacobian_norms_BC(llm_rd_BCR)
plot_discrete_jacobian(og_norms_BC, rd_norms_BC, model_name, data_name)

#%% [markdown]
# ## Local PCA along trajectory
# Characterize local smoothness by fitting PCA in a sliding window along each
# trajectory. Two signals:
# 1. **Tangent plane rotation**: max principal angle between consecutive local
#    tangent planes (large → abrupt direction change).
# 2. **Local explained variance**: fraction of local variance captured by the
#    top `m` components (high → locally low-dimensional / smooth).

#%%
def local_pca_along_trajectory(llm_CR, k, m):
    """Fit local PCA in a symmetric window of size k around each trajectory point.

    Args:
        llm_CR: trajectory points, shape (C, R)
        k: window size (will be clamped to C)
        m: number of top components to keep

    Returns:
        eigvals_Cm: top m eigenvalues at each point, shape (C, m)
        tangent_CmR: top m eigenvectors at each point, shape (C, m, R)
    """
    C, R = llm_CR.shape
    k = min(k, C)
    half = k // 2
    eigvals_Cm = np.empty((C, m))
    tangent_CmR = np.empty((C, m, R))

    for i in range(C):
        lo = max(0, i - half)
        hi = min(C, i + half + 1)
        # Ensure we always take k points when possible
        if hi - lo < k:
            if lo == 0:
                hi = min(C, lo + k)
            else:
                lo = max(0, hi - k)
        window_wR = llm_CR[lo:hi]  # (w, R) where w <= k
        window_wR = window_wR - window_wR.mean(axis=0, keepdims=True)
        _, S_R, Vh_RR = np.linalg.svd(window_wR, full_matrices=False)
        # Eigenvalues of covariance ∝ S^2
        eigvals_Cm[i] = S_R[:m] ** 2
        tangent_CmR[i] = Vh_RR[:m]

    return eigvals_Cm, tangent_CmR


def principal_angles_between_planes(V1_mR, V2_mR):
    """Compute principal angles between two subspaces.

    Args:
        V1_mR, V2_mR: orthonormal bases, shape (m, R)

    Returns:
        angles_m: principal angles in radians, shape (m,)
    """
    M_mm = V1_mR @ V2_mR.T  # (m, m)
    _, sigmas_m, _ = np.linalg.svd(M_mm)
    # Clamp for numerical safety before arccos
    sigmas_m = np.clip(sigmas_m, -1.0, 1.0)
    return np.arccos(sigmas_m)


def local_tangent_rotation_BC(llm_BCR, k, m):
    """Max principal angle between consecutive local tangent planes per template.

    Returns shape (B, C-1).
    """
    B, C, R = llm_BCR.shape
    rotation_BC = np.empty((B, C - 1))
    for b in range(B):
        _, tangent_CmR = local_pca_along_trajectory(llm_BCR[b], k, m)
        for i in range(C - 1):
            angles_m = principal_angles_between_planes(tangent_CmR[i], tangent_CmR[i + 1])
            rotation_BC[b, i] = np.max(angles_m)
    return rotation_BC


def local_explained_variance_BC(llm_BCR, k, m):
    """Fraction of local variance explained by top m components at each point.

    Returns shape (B, C).
    """
    B, C, R = llm_BCR.shape
    m_all = min(min(k, C), R)  # max eigenvalues SVD can return
    expvar_BC = np.empty((B, C))
    for b in range(B):
        eigvals_Call, _ = local_pca_along_trajectory(llm_BCR[b], k, m_all)
        top_m_C = eigvals_Call[:, :m].sum(axis=1)
        total_C = eigvals_Call.sum(axis=1)
        expvar_BC[b] = top_m_C / np.maximum(total_C, 1e-30)
    return expvar_BC


#%%
# Compute local PCA metrics

k_window = min(11, C)
m_tangent = 3

og_rotation_BC = local_tangent_rotation_BC(llm_og_BCR, k_window, m_tangent)
rd_rotation_BC = local_tangent_rotation_BC(llm_rd_BCR, k_window, m_tangent)

og_expvar_BC = local_explained_variance_BC(llm_og_BCR, k_window, m_tangent)
rd_expvar_BC = local_explained_variance_BC(llm_rd_BCR, k_window, m_tangent)

#%%
# Plot 1 — Tangent plane rotation

def plot_tangent_rotation(og_rotation_BC, rd_rotation_BC, model_name, data_name, k, m, max_traces=1000):
    C_minus_1 = og_rotation_BC.shape[1]
    x = np.arange(C_minus_1)

    fig, ax = plt.subplots(figsize=(7, 4))
    rng = np.random.default_rng(42)

    # Individual traces (faint)
    orig_idx = rng.choice(og_rotation_BC.shape[0], size=min(max_traces, og_rotation_BC.shape[0]), replace=False)
    rand_idx = rng.choice(rd_rotation_BC.shape[0], size=min(max_traces, rd_rotation_BC.shape[0]), replace=False)
    for b in orig_idx:
        ax.plot(x, og_rotation_BC[b], color='tab:blue', alpha=0.15, linewidth=0.8)
    for b in rand_idx:
        ax.plot(x, rd_rotation_BC[b], color='tab:orange', alpha=0.15, linewidth=0.8)

    # Mean trajectories (bold)
    og_mean = np.mean(og_rotation_BC, axis=0)
    rd_mean = np.mean(rd_rotation_BC, axis=0)
    ax.plot(x, og_mean, 'o-', markersize=3, color='tab:blue', linewidth=2, label=f"Original ({data_name})")
    ax.plot(x, rd_mean, 's-', markersize=3, color='tab:orange', linewidth=2, label="Random")

    ax.set_xlabel("Step along trajectory")
    ax.set_ylabel("Max principal angle (rad)")
    ax.set_title(f"{model_name} — {data_name}: Tangent Plane Rotation (k={k}, m={m}, R={R})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_tangent_rotation(og_rotation_BC, rd_rotation_BC, model_name, data_name, k_window, m_tangent, max_traces=20)

#%%
# Plot 2 — Local explained variance ratio

def plot_local_explained_variance(og_expvar_BC, rd_expvar_BC, model_name, data_name, k, m, max_traces=1000):
    C_pts = og_expvar_BC.shape[1]
    x = np.arange(C_pts)

    fig, ax = plt.subplots(figsize=(7, 4))
    rng = np.random.default_rng(42)

    # Individual traces (faint)
    orig_idx = rng.choice(og_expvar_BC.shape[0], size=min(max_traces, og_expvar_BC.shape[0]), replace=False)
    rand_idx = rng.choice(rd_expvar_BC.shape[0], size=min(max_traces, rd_expvar_BC.shape[0]), replace=False)
    for b in orig_idx:
        ax.plot(x, og_expvar_BC[b], color='tab:blue', alpha=0.15, linewidth=0.8)
    for b in rand_idx:
        ax.plot(x, rd_expvar_BC[b], color='tab:orange', alpha=0.15, linewidth=0.8)

    # Mean trajectories (bold)
    og_mean = np.mean(og_expvar_BC, axis=0)
    rd_mean = np.mean(rd_expvar_BC, axis=0)
    ax.plot(x, og_mean, 'o-', markersize=3, color='tab:blue', linewidth=2, label=f"Original ({data_name})")
    ax.plot(x, rd_mean, 's-', markersize=3, color='tab:orange', linewidth=2, label="Random")

    ax.set_xlabel("Step along trajectory")
    ax.set_ylabel("Explained variance ratio (top m)")
    ax.set_title(f"{model_name} — {data_name}: Local Explained Variance (k={k}, m={m}, R={R})")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_local_explained_variance(og_expvar_BC, rd_expvar_BC, model_name, data_name, k_window, m_tangent)

# %% [markdown]
# # Basis Fit Complexity
# For each PCA component of the centroid trajectory, find the minimum complexity
# (across multiple basis families) needed to achieve R² ≥ 0.95.

# %%
# Fitting functions for basis fit complexity analysis

import pywt
from scipy.interpolate import make_lsq_spline, BSpline


def r2_score(y_true_C, y_pred_C):
    """Standard R² (coefficient of determination)."""
    ss_res = np.sum((y_true_C - y_pred_C) ** 2)
    ss_tot = np.sum((y_true_C - np.mean(y_true_C)) ** 2)
    if ss_tot < 1e-30:
        return 1.0
    return 1.0 - ss_res / ss_tot


def fit_chebyshev(llm_C, y_C, max_degree=10, r2_thresh=0.95):
    """Fit Chebyshev polynomials, keeping top-N coefficients by magnitude.

    Returns (min_N, r2, sparse_coeffs) achieving the threshold.
    """
    llm_cheb_C = 2.0 * llm_C - 1.0
    all_coeffs = np.polynomial.chebyshev.chebfit(llm_cheb_C, y_C, max_degree)
    n_total = len(all_coeffs)
    sorted_idx = np.argsort(np.abs(all_coeffs))[::-1]
    for N in range(1, n_total + 1):
        sparse = np.zeros_like(all_coeffs)
        sparse[sorted_idx[:N]] = all_coeffs[sorted_idx[:N]]
        y_pred_C = np.polynomial.chebyshev.chebval(llm_cheb_C, sparse)
        r2 = r2_score(y_C, y_pred_C)
        if r2 >= r2_thresh:
            return N, r2, sparse
    return n_total, r2, sparse


def fit_fourier(llm_C, y_C, max_harmonics=10, r2_thresh=0.95):
    """Fit Fourier basis, keeping top-N coefficients by magnitude.

    Maps data to [0, 0.5] so the period-1 basis doesn't force f(start)=f(end).
    """
    C = len(llm_C)
    u_C = llm_C * 0.5

    max_cols = 2 * max_harmonics + 1
    design_CM = np.ones((C, max_cols))
    for h in range(1, max_harmonics + 1):
        design_CM[:, 2 * h - 1] = np.cos(2 * np.pi * h * u_C)
        design_CM[:, 2 * h] = np.sin(2 * np.pi * h * u_C)

    all_coeffs, _, _, _ = np.linalg.lstsq(design_CM, y_C, rcond=None)
    n_total = len(all_coeffs)
    sorted_idx = np.argsort(np.abs(all_coeffs))[::-1]
    for N in range(1, n_total + 1):
        sparse = np.zeros_like(all_coeffs)
        sparse[sorted_idx[:N]] = all_coeffs[sorted_idx[:N]]
        y_pred_C = design_CM @ sparse
        r2 = r2_score(y_C, y_pred_C)
        if r2 >= r2_thresh:
            return N, r2, sparse
    return n_total, r2, sparse


def fit_bspline(llm_C, y_C, n_interior_knots, r2_thresh=0.95):
    """Fit cubic B-spline with a fixed number of interior knots.

    Returns (n_interior_knots, r2).
    """
    k = 3  # cubic
    interior = np.linspace(0, 1, n_interior_knots + 2)[1:-1]
    knots = np.concatenate([
        np.zeros(k + 1),
        interior,
        np.ones(k + 1),
    ])
    # make_lsq_spline requires sorted llm_C
    sort_idx = np.argsort(llm_C)
    llm_sorted_C = llm_C[sort_idx]
    y_sorted_C = y_C[sort_idx]
    # Clamp endpoints slightly inside to avoid knot boundary issues
    llm_sorted_C = np.clip(llm_sorted_C, 1e-10, 1.0 - 1e-10)
    spline = make_lsq_spline(llm_sorted_C, y_sorted_C, knots, k=k)
    y_pred_C = np.empty_like(y_C)
    y_pred_C[sort_idx] = spline(llm_sorted_C)
    r2 = r2_score(y_C, y_pred_C)
    return n_interior_knots, r2, spline


def fit_bspline_sweep(llm_C, y_C, max_knots=10, r2_thresh=0.95):
    """Fit cubic B-spline with max interior knots, keeping top-N coefficients by magnitude.

    Returns (min_N, r2, sparse_spline) achieving the threshold.
    """
    k = 3  # cubic
    interior = np.linspace(0, 1, max_knots + 2)[1:-1]
    knots = np.concatenate([
        np.zeros(k + 1),
        interior,
        np.ones(k + 1),
    ])
    sort_idx = np.argsort(llm_C)
    llm_sorted_C = np.clip(llm_C[sort_idx], 1e-10, 1.0 - 1e-10)
    y_sorted_C = y_C[sort_idx]
    full_spline = make_lsq_spline(llm_sorted_C, y_sorted_C, knots, k=k)

    all_coeffs = full_spline.c
    n_total = len(all_coeffs)
    sorted_coeff_idx = np.argsort(np.abs(all_coeffs))[::-1]
    for N in range(1, n_total + 1):
        sparse_coeffs = np.zeros_like(all_coeffs)
        sparse_coeffs[sorted_coeff_idx[:N]] = all_coeffs[sorted_coeff_idx[:N]]
        sparse_spline = BSpline(full_spline.t, sparse_coeffs, full_spline.k)
        y_pred_sorted = sparse_spline(llm_sorted_C)
        y_pred_C = np.empty_like(y_C)
        y_pred_C[sort_idx] = y_pred_sorted
        r2 = r2_score(y_C, y_pred_C)
        if r2 >= r2_thresh:
            return N, r2, sparse_spline
    return n_total, r2, sparse_spline


def fit_wavelet(llm_C, y_C, r2_thresh=0.95):
    """Fit wavelet basis by retaining top-N coefficients.

    Returns (min_N, r2) achieving the threshold, or (total_coeffs, best_r2).
    """
    coeffs_list = pywt.wavedec(y_C, "db4")
    # Flatten all coefficients
    all_coeffs = np.concatenate([c.ravel() for c in coeffs_list])
    n_total = len(all_coeffs)

    # Sort by absolute value descending
    sorted_idx = np.argsort(np.abs(all_coeffs))[::-1]

    # Reconstruct coefficient structure shapes
    shapes = [c.shape for c in coeffs_list]
    boundaries = np.cumsum([0] + [c.size for c in coeffs_list])

    for N in range(1, n_total + 1):
        # Zero out all but top-N
        sparse = np.zeros_like(all_coeffs)
        sparse[sorted_idx[:N]] = all_coeffs[sorted_idx[:N]]
        # Rebuild coefficient list
        rebuilt = []
        for i, shape in enumerate(shapes):
            rebuilt.append(sparse[boundaries[i]:boundaries[i + 1]].reshape(shape))
        y_pred = pywt.waverec(rebuilt, "db4")
        # waverec can produce array slightly longer than input
        y_pred_C = y_pred[: len(y_C)]
        r2 = r2_score(y_C, y_pred_C)
        if r2 >= r2_thresh:
            return N, r2, y_pred_C
    return n_total, r2, y_pred_C


# %%
# Derivative evaluation functions (use fit artifacts, no re-fitting)


def eval_deriv_chebyshev(llm_C, coeffs):
    """Derivative of Chebyshev fit. Chain rule factor 2 for [0,1]->[−1,1]."""
    llm_cheb_C = 2.0 * llm_C - 1.0
    dcoeffs = np.polynomial.chebyshev.chebder(coeffs)
    return 2.0 * np.polynomial.chebyshev.chebval(llm_cheb_C, dcoeffs)


def eval_deriv_fourier(llm_C, coeffs):
    """Analytic derivative of half-period Fourier fit. Chain rule: du/dx = 0.5."""
    u_C = llm_C * 0.5  # <-- must match fit_fourier
    n_harmonics = (len(coeffs) - 1) // 2
    deriv_C = np.zeros(len(u_C))
    for h in range(1, n_harmonics + 1):
        a_h, b_h = coeffs[2 * h - 1], coeffs[2 * h]
        deriv_C += -2 * np.pi * h * a_h * np.sin(2 * np.pi * h * u_C)
        deriv_C += 2 * np.pi * h * b_h * np.cos(2 * np.pi * h * u_C)
    return deriv_C * 0.5  # <-- chain rule factor for u = x * 0.5


def eval_deriv_bspline(llm_C, spline):
    """Derivative via scipy BSpline.derivative()."""
    sort_idx = np.argsort(llm_C)
    llm_sorted_C = np.clip(llm_C[sort_idx], 1e-10, 1.0 - 1e-10)
    dspline = spline.derivative()
    deriv_C = np.empty(len(llm_C))
    deriv_C[sort_idx] = dspline(llm_sorted_C)
    return deriv_C


def eval_deriv_wavelet(llm_C, y_fit_C):
    """Finite-difference derivative of wavelet reconstruction."""
    dz = llm_C[1] - llm_C[0]
    return np.gradient(y_fit_C, dz)


_DERIV_DISPATCH = {
    "Chebyshev": eval_deriv_chebyshev,
    "Fourier": eval_deriv_fourier,
    "B-spline": eval_deriv_bspline,
    "Wavelet": eval_deriv_wavelet,
}


def eval_fit_chebyshev(llm_C, coeffs):
    """Evaluate Chebyshev fit on [0, 1]."""
    llm_cheb_C = 2.0 * llm_C - 1.0
    return np.polynomial.chebyshev.chebval(llm_cheb_C, coeffs)


def eval_fit_fourier(llm_C, coeffs):
    """Evaluate Fourier fit from coefficients (half-period mapping)."""
    u_C = llm_C * 0.5  # <-- must match fit_fourier
    n_harmonics = (len(coeffs) - 1) // 2
    y_C = np.full(len(u_C), coeffs[0])
    for h in range(1, n_harmonics + 1):
        y_C += coeffs[2 * h - 1] * np.cos(2 * np.pi * h * u_C)
        y_C += coeffs[2 * h] * np.sin(2 * np.pi * h * u_C)
    return y_C


def eval_fit_bspline(llm_C, spline):
    """Evaluate B-spline fit."""
    sort_idx = np.argsort(llm_C)
    llm_sorted_C = np.clip(llm_C[sort_idx], 1e-10, 1.0 - 1e-10)
    y_C = np.empty(len(llm_C))
    y_C[sort_idx] = spline(llm_sorted_C)
    return y_C


def eval_fit_wavelet(llm_C, y_fit_C):
    """Return wavelet reconstruction (artifact is already the fitted curve)."""
    return y_fit_C


_EVAL_DISPATCH = {
    "Chebyshev": eval_fit_chebyshev,
    "Fourier": eval_fit_fourier,
    "B-spline": eval_fit_bspline,
    "Wavelet": eval_fit_wavelet,
}


# %%
# Sweep all methods per PCA component

def compute_basis_complexity_per_component(centroid_CR, max_components=15, r2_thresh=0.95):
    """Compute basis fit complexity for each PCA component of the centroid trajectory.

    Args:
        centroid_CR: shape (C, R) — mean over B of llm_BCR
        max_components: max number of PCA components to analyze
        r2_thresh: R² threshold for "good enough" fit

    Returns:
        dict with keys = method names, values = list of (complexity, r2, artifact) per component.
        artifact is method-specific: Chebyshev coeffs, Fourier coeffs, BSpline object, or wavelet reconstruction.
    """
    C, R_total = centroid_CR.shape
    n_comp = min(R_total, max_components)
    llm_C = np.linspace(0, 1, C)

    methods = {
        "Chebyshev": lambda y: fit_chebyshev(llm_C, y, max_degree=15, r2_thresh=r2_thresh),
        "Fourier": lambda y: fit_fourier(llm_C, y, max_harmonics=8, r2_thresh=r2_thresh),
        "B-spline": lambda y: fit_bspline_sweep(llm_C, y, max_knots=15, r2_thresh=r2_thresh),
        "Wavelet": lambda y: fit_wavelet(llm_C, y, r2_thresh=r2_thresh),
    }

    results = {name: [] for name in methods}
    for j in range(n_comp):
        y_C = centroid_CR[:, j]
        for name, fn in methods.items():
            complexity, r2, artifact = fn(y_C)
            results[name].append((complexity, r2, artifact))
    return results

def select_best_methods(results_og, results_rd):
    """For each PCA component, pick the method with lowest relative complexity.

    Relative complexity = og_complexity / max(rd_complexity, 1.0).

    Returns:
        list of (method_name, component_index) tuples
    """
    method_names = list(results_og.keys())
    n_comp = len(results_og[method_names[0]])
    best_methods = []
    for j in range(n_comp):
        best_name = None
        best_ratio = np.inf
        for name in method_names:
            og_complexity = results_og[name][j][0]
            rd_complexity = results_rd[name][j][0]
            ratio = og_complexity / max(rd_complexity, 1.0)
            if ratio < best_ratio:
                best_ratio = ratio
                best_name = name
        best_methods.append((best_name, j))
    return best_methods


# %%
# Normalization + plotting

def plot_basis_complexity(centroid_og_CR, centroid_rd_CR, model_name, data_name,
                          max_components=15, r2_thresh=0.95,
                          results_og=None, results_rd=None):
    """Plot basis fit complexity per PCA component.

    Two panels:
    - Left: raw complexity (number of basis elements) for original and random.
    - Right: complexity ratio (original / random) per method. Values < 1 mean
      the real manifold is simpler than chance.
    """

    if results_og is None:
        results_og = compute_basis_complexity_per_component(centroid_og_CR, max_components, r2_thresh)
    if results_rd is None:
        results_rd = compute_basis_complexity_per_component(centroid_rd_CR, max_components, r2_thresh)

    method_names = list(results_og.keys())
    n_methods = len(method_names)
    n_comp = len(results_og[method_names[0]])
    colors = plt.cm.tab10(np.arange(n_methods))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: raw complexity for original vs random (grouped bars)
    ax = axes[0]
    x = np.arange(n_comp)
    bar_width = 0.8 / (n_methods * 2)
    for i, name in enumerate(method_names):
        og_raw = [results_og[name][j][0] for j in range(n_comp)]
        rd_raw = [results_rd[name][j][0] for j in range(n_comp)]
        offset_og = (2 * i) * bar_width
        offset_rd = (2 * i + 1) * bar_width
        ax.bar(x + offset_og, og_raw, bar_width, color=colors[i], label=f"{name} (orig)")
        ax.bar(x + offset_rd, rd_raw, bar_width, color=colors[i], alpha=0.4, label=f"{name} (rand)")
    ax.set_xlabel("PCA component index")
    ax.set_ylabel("Raw complexity (# basis elements)")
    ax.set_title(f"Raw Complexity (R²≥{r2_thresh})")
    ax.set_xticks(x + bar_width * (n_methods - 0.5))
    ax.set_xticklabels([str(i) for i in range(n_comp)])
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # Right panel: complexity ratio (original / random)
    ax = axes[1]
    bar_width = 0.8 / n_methods
    for i, name in enumerate(method_names):
        og_raw = np.array([results_og[name][j][0] for j in range(n_comp)], dtype=float)
        rd_raw = np.array([results_rd[name][j][0] for j in range(n_comp)], dtype=float)
        # Ratio: original / random (avoid division by zero)
        ratio = og_raw / np.maximum(rd_raw, 1.0)
        ax.bar(x + i * bar_width, ratio, bar_width, color=colors[i], label=name)

    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label="chance level")
    ax.set_xlabel("PCA component index")
    ax.set_ylabel("Complexity ratio (original / random)")
    ax.set_title("Relative Complexity (<1 = simpler than chance)")
    ax.set_xticks(x + bar_width * (n_methods - 1) / 2)
    ax.set_xticklabels([str(i) for i in range(n_comp)])
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"{model_name} — {data_name}: Basis Fit Complexity per PCA Component (R²≥{r2_thresh})",
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()


# Compute centroids: mean over templates (B dimension)
centroid_og_CR = np.mean(llm_og_BCR, axis=0)  # (C, R)
centroid_rd_CR = np.mean(llm_rd_BCR, axis=0)  # (C, R)

# Basis complexity (shared between plots)
results_og = compute_basis_complexity_per_component(centroid_og_CR)
results_rd = compute_basis_complexity_per_component(centroid_rd_CR)

plot_basis_complexity(centroid_og_CR, centroid_rd_CR, model_name, data_name,
                      results_og=results_og, results_rd=results_rd)


#%%

def plot_per_component_projections(centroid_og_CR, centroid_rd_CR,
                                   results_og, results_rd,
                                   model_name, data_name, max_components=15):
    """Plot per-component centroid projections and their basis fits.

    Layout: one row per PCA component.
    Columns: one per fit method (original) + one for random (best method).
    The lowest-complexity original method per row gets a highlighted border.
    """
    C = centroid_og_CR.shape[0]
    llm_C = np.linspace(0, 1, C)

    method_names = list(results_og.keys())
    n_methods = len(method_names)
    n_comp = min(centroid_og_CR.shape[1], max_components, len(results_og[method_names[0]]))
    method_colors = {"Chebyshev": "C0", "Fourier": "C1", "B-spline": "C2", "Wavelet": "C3"}

    # Evaluate fits for all methods on original data
    fit_og = {}  # method_name -> (C, n_comp) array
    for name in method_names:
        eval_fn = _EVAL_DISPATCH[name]
        y_CR = np.zeros((C, n_comp))
        for j in range(n_comp):
            artifact = results_og[name][j][2]
            y_CR[:, j] = eval_fn(llm_C, artifact)
        fit_og[name] = y_CR

    # Evaluate fits for random (best method only)
    best_methods_rd = select_best_methods(results_rd, results_og)
    fit_rd_CR = np.zeros((C, n_comp))
    for method_name, j in best_methods_rd[:n_comp]:
        artifact = results_rd[method_name][j][2]
        eval_fn = _EVAL_DISPATCH[method_name]
        fit_rd_CR[:, j] = eval_fn(llm_C, artifact)

    # Find best (lowest complexity) original method per component
    best_og_method_per_comp = []
    for j in range(n_comp):
        best_name = min(method_names, key=lambda n: results_og[n][j][0])
        best_og_method_per_comp.append(best_name)

    # Layout: n_methods original columns + 1 random column
    n_cols = n_methods + 1
    fig, axes = plt.subplots(n_comp, n_cols, figsize=(2.8 * n_cols, 2.2 * n_comp),
                             sharex=True)
    if n_comp == 1:
        axes = axes[np.newaxis, :]

    for j in range(n_comp):
        # Original columns (one per method)
        for mi, name in enumerate(method_names):
            ax = axes[j, mi]
            color = method_colors.get(name, "C7")
            ax.scatter(llm_C, centroid_og_CR[:, j], color="grey", alpha=1.0, s=4, zorder=0)
            ax.plot(llm_C, fit_og[name][:, j], color=color, linewidth=1.0)
            ax.grid(True, alpha=0.3)

            # Complexity annotation
            complexity = results_og[name][j][0]
            r2 = results_og[name][j][1]
            ax.text(0.97, 0.92, f"k={complexity} R²={r2:.2f}",
                    transform=ax.transAxes, fontsize=6, ha="right", va="top")

            # Highlight best method with a thick colored border
            if name == best_og_method_per_comp[j]:
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2.5)

            if mi == 0:
                ax.set_ylabel(f"PC{j}")

        # Random column (rightmost)
        ax_rd = axes[j, n_methods]
        method_rd = best_methods_rd[j][0]
        color_rd = method_colors.get(method_rd, "C7")
        ax_rd.scatter(llm_C, centroid_rd_CR[:, j], color="grey", alpha=1.0, s=4, zorder=0)
        ax_rd.plot(llm_C, fit_rd_CR[:, j], color=color_rd, linewidth=1.0)
        ax_rd.grid(True, alpha=0.3)
        ax_rd.text(0.97, 0.92, method_rd, transform=ax_rd.transAxes,
                   fontsize=6, ha="right", va="top", color=color_rd)

    # Column headers
    for mi, name in enumerate(method_names):
        axes[0, mi].set_title(name, color=method_colors.get(name, "C7"))
    axes[0, n_methods].set_title("Random")

    # Bottom x-labels
    for col in range(n_cols):
        axes[-1, col].set_xlabel("z")

    fig.suptitle(
        f"{model_name} — {data_name}: Per-Component Projections "
        r"$\Psi_j(z)$",
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()


plot_per_component_projections(centroid_og_CR, centroid_rd_CR,
                               results_og, results_rd,
                               model_name, data_name)



#%% [markdown]
# ## Metric tensor: squishing / stretching of space
# **Metric tensor** — Top row: local stretching factor $\sqrt{g(z)}$ showing how much the embedding space expands or compresses at each position. Bottom row: per-component derivative magnitudes $|d\Psi_j/dz|$, colored by best-fit basis. Left column: original, right column: random.


# %%
# Metric tensor: select best methods, compute g(z), and plot sqrt(g(z))


def compute_metric_tensor(centroid_og_CR, results_og, results_rd, max_components=15):
    """Compute 1D metric tensor g(z) = sum_j (dPsi_j/dz)^2.

    For each PCA component, uses the basis family with lowest relative complexity.

    Args:
        centroid_og_CR: shape (C, R) — mean trajectory
        results_og: dict from compute_basis_complexity_per_component (original)
        results_rd: dict from compute_basis_complexity_per_component (random)
        max_components: number of PCA components to include

    Returns:
        (g_C, best_methods, deriv_CR) where
        g_C: shape (C,) — metric tensor values
        best_methods: list of (method_name, component_index)
        deriv_CR: shape (C, n_comp) — per-component derivatives
    """
    C, R_total = centroid_og_CR.shape
    n_comp = min(R_total, max_components)
    llm_C = np.linspace(0, 1, C)

    best_methods = select_best_methods(results_og, results_rd)

    deriv_CR = np.zeros((C, n_comp))
    for method_name, j in best_methods[:n_comp]:
        artifact = results_og[method_name][j][2]
        deriv_fn = _DERIV_DISPATCH[method_name]
        deriv_CR[:, j] = deriv_fn(llm_C, artifact)

    g_C = np.sum(deriv_CR ** 2, axis=1)
    return g_C, best_methods[:n_comp], deriv_CR


def plot_metric_tensor(centroid_og_CR, centroid_rd_CR,
                       results_og, results_rd, model_name, data_name,
                       max_components=15):
    """Plot sqrt(g(z)) and per-component derivatives for original vs random.

    2x2 grid:
    - Top row: local stretching factor sqrt(g(z))
    - Bottom row: per-component |dPsi_j/dz| colored by method
    - Left column: original, Right column: random
    """
    C = centroid_og_CR.shape[0]
    llm_C = np.linspace(0, 1, C)
    dz = llm_C[1] - llm_C[0]
    method_colors = {"Chebyshev": "C0", "Fourier": "C1", "B-spline": "C2", "Wavelet": "C3"}

    # Compute metric tensors for both
    g_og_C, methods_og, deriv_og_CR = compute_metric_tensor(
        centroid_og_CR, results_og, results_rd, max_components
    )
    g_rd_C, methods_rd, deriv_rd_CR = compute_metric_tensor(
        centroid_rd_CR, results_rd, results_og, max_components
    )
    sqrt_g_og_C = np.sqrt(g_og_C)
    sqrt_g_rd_C = np.sqrt(g_rd_C)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # --- Top row: sqrt(g(z)) ---
    for col, (sqrt_g_C, label) in enumerate([
        (sqrt_g_og_C, "Original"), (sqrt_g_rd_C, "Random")
    ]):
        ax = axes[0, col]
        arc_length = np.trapezoid(sqrt_g_C, dx=dz)
        stretch_ratio = np.max(sqrt_g_C) / np.maximum(np.min(sqrt_g_C), 1e-10)
        ax.plot(llm_C, sqrt_g_C, color="k", linewidth=1.5, label=r"$\sqrt{g(z)}$")
        ax.axhline(np.mean(sqrt_g_C), color="gray", linestyle="--", alpha=0.6,
                   label=f"mean = {np.mean(sqrt_g_C):.1f}")
        ax.set_xlabel("z (normalized position)")
        ax.set_ylabel(r"$\sqrt{g(z)}$ (local stretching)")
        ax.set_title(f"{label}: stretching factor  "
                     f"(arc={arc_length:.1f}, max/min={stretch_ratio:.1f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- Bottom row: per-component |dPsi_j/dz| ---
    for col, (deriv_CR, best_methods, label) in enumerate([
        (deriv_og_CR, methods_og, "Original"),
        (deriv_rd_CR, methods_rd, "Random"),
    ]):
        ax = axes[1, col]
        n_comp = deriv_CR.shape[1]
        for j in range(n_comp):
            method_name = best_methods[j][0]
            color = method_colors.get(method_name, "C7")
            lbl = f"PC{j} ({method_name})"
            ax.plot(llm_C, np.abs(deriv_CR[:, j]), color=color, alpha=0.6,
                    linewidth=0.8, label=lbl)
        ax.set_xlabel("z (normalized position)")
        ax.set_ylabel(r"$|d\Psi_j / dz|$")
        ax.set_title(f"{label}: per-component derivative magnitudes")
        ax.legend(fontsize=5, loc="upper right", ncol=-(-n_comp // 3))
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{model_name} — {data_name}: Metric Tensor", fontsize=12)
    plt.tight_layout()
    plt.show()


# %%
# Metric tensor: local stretching factor sqrt(g(z))
plot_metric_tensor(centroid_og_CR, centroid_rd_CR, results_og, results_rd, model_name, data_name)

#%% [markdown]
# Interesting in the cell below there seems to be a transition between PC 9 and 10, even for random. Does that tell us sth about the template?

# %%
# Per-component derivative magnitudes: original vs random side-by-side
# 

def plot_per_component_derivatives(centroid_og_CR, centroid_rd_CR,
                                   results_og, results_rd,
                                   model_name, data_name, max_components=15):
    """Plot per-component |dPsi_j/dz| in separate subplots.

    Layout: one row per PCA component.
    Columns: one per fit method (original) + one for random (best method).
    The lowest-complexity original method per row gets a highlighted border.
    """
    C = centroid_og_CR.shape[0]
    llm_C = np.linspace(0, 1, C)

    method_names = list(results_og.keys())
    n_methods = len(method_names)
    n_comp = min(centroid_og_CR.shape[1], max_components, len(results_og[method_names[0]]))
    method_colors = {"Chebyshev": "C0", "Fourier": "C1", "B-spline": "C2", "Wavelet": "C3"}

    # Compute derivatives for all methods on original data
    deriv_og = {}  # method_name -> (C, n_comp) array
    for name in method_names:
        deriv_fn = _DERIV_DISPATCH[name]
        d_CR = np.zeros((C, n_comp))
        for j in range(n_comp):
            artifact = results_og[name][j][2]
            d_CR[:, j] = deriv_fn(llm_C, artifact)
        deriv_og[name] = d_CR

    # Compute derivatives for random (best method only)
    _, best_methods_rd, deriv_rd_CR = compute_metric_tensor(
        centroid_rd_CR, results_rd, results_og, max_components
    )

    # Find best (lowest complexity) original method per component
    best_og_method_per_comp = []
    for j in range(n_comp):
        best_name = min(method_names, key=lambda n: results_og[n][j][0])
        best_og_method_per_comp.append(best_name)

    # Layout: n_methods original columns + 1 random column
    n_cols = n_methods + 1
    fig, axes = plt.subplots(n_comp, n_cols, figsize=(2.8 * n_cols, 2.2 * n_comp),
                             sharex=True)
    if n_comp == 1:
        axes = axes[np.newaxis, :]

    # Compute numerical (finite-difference) derivatives of centroids
    num_deriv_og_CR = np.zeros((C, n_comp))
    num_deriv_rd_CR = np.zeros((C, n_comp))
    for j in range(n_comp):
        num_deriv_og_CR[:, j] = np.gradient(centroid_og_CR[:, j], llm_C)
        num_deriv_rd_CR[:, j] = np.gradient(centroid_rd_CR[:, j], llm_C)

    for j in range(n_comp):
        # Original columns (one per method)
        for mi, name in enumerate(method_names):
            ax = axes[j, mi]
            color = method_colors.get(name, "C7")
            ax.scatter(llm_C, np.abs(num_deriv_og_CR[:, j]), color="grey", alpha=1.0, s=4, zorder=0)
            ax.plot(llm_C, np.abs(deriv_og[name][:, j]), color=color, linewidth=1.0)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, None)

            # Complexity annotation
            complexity = results_og[name][j][0]
            r2 = results_og[name][j][1]
            ax.text(0.97, 0.92, f"k={complexity} R²={r2:.2f}",
                    transform=ax.transAxes, fontsize=6, ha="right", va="top")

            # Highlight best method with a thick colored border
            if name == best_og_method_per_comp[j]:
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2.5)

            if mi == 0:
                ax.set_ylabel(f"PC{j}")

        # Random column (rightmost)
        ax_rd = axes[j, n_methods]
        method_rd = best_methods_rd[j][0]
        color_rd = method_colors.get(method_rd, "C7")
        ax_rd.scatter(llm_C, np.abs(num_deriv_rd_CR[:, j]), color="grey", alpha=1.0, s=4, zorder=0)
        ax_rd.plot(llm_C, np.abs(deriv_rd_CR[:, j]), color=color_rd, linewidth=1.0)
        ax_rd.grid(True, alpha=0.3)
        ax_rd.set_ylim(0, None)
        ax_rd.text(0.97, 0.92, method_rd, transform=ax_rd.transAxes,
                   fontsize=6, ha="right", va="top", color=color_rd)

    # Column headers
    for mi, name in enumerate(method_names):
        axes[0, mi].set_title(name, color=method_colors.get(name, "C7"))
    axes[0, n_methods].set_title("Random")

    # Bottom x-labels
    for col in range(n_cols):
        axes[-1, col].set_xlabel("z")

    fig.suptitle(
        f"{model_name} — {data_name}: Per-Component Derivative Magnitudes "
        r"$|d\Psi_j / dz|$",
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()


plot_per_component_derivatives(centroid_og_CR, centroid_rd_CR,
                               results_og, results_rd,
                               model_name, data_name)



# %% [markdown]
# # Distortion
# In this section we will measure distortion, compared to a ground truth kernel matrix.




















# %% [markdown]
# # Topological Characteristics
# Betti numbers via persistent homology reveal the topology of embedding manifolds:
# - β₀ = number of connected components (expect 1 for a single trajectory)
# - β₁ = number of 1-dimensional loops (expect >0 for cyclic concepts like months/colors)
# - β₂ = number of 2-dimensional voids
#
# We use Vietoris–Rips persistent homology (ripser) on PCA-truncated trajectories.

# Intuitively, at Birth we inflate all scatterpoints to big balls/spheres which all overlap each other in one giant blob.
# Then, we gradually deflate, and always count number of (beta0) consecutive pieces, (beta1) holes, (beta2) enclosed 2D spheres, ...

# %%
from ripser import ripser


def compute_persistence_CR(points_CR, maxdim=2):
    """Run Vietoris-Rips persistent homology on a single (C, R) point cloud.

    Returns the ripser result dict containing 'dgms' (list of persistence
    diagrams per homology dimension).
    """
    return ripser(points_CR, maxdim=maxdim)


def betti_from_diagrams(dgms, rel_threshold=0.1):
    """Extract Betti numbers from persistence diagrams.

    A feature counts as 'real' if (death - birth) / max_persistence > rel_threshold.

    Returns:
        betti_H: list of Betti numbers per homology dimension
        lifetimes_per_dim: list of lifetime arrays per homology dimension
    """
    # Compute max persistence across all finite features
    all_lifetimes = []
    for dgm in dgms:
        finite_mask = np.isfinite(dgm[:, 1])
        if np.any(finite_mask):
            all_lifetimes.append(dgm[finite_mask, 1] - dgm[finite_mask, 0])
    max_persistence = max(np.max(lt) for lt in all_lifetimes) if all_lifetimes else 1.0

    betti_H = []
    lifetimes_per_dim = []
    for dgm in dgms:
        finite_mask = np.isfinite(dgm[:, 1])
        lifetimes = dgm[finite_mask, 1] - dgm[finite_mask, 0]
        significant = lifetimes > rel_threshold * max_persistence
        betti_H.append(int(np.sum(significant)))
        lifetimes_per_dim.append(lifetimes)
    return betti_H, lifetimes_per_dim


def compute_betti_per_template_BH(llm_BCR, maxdim=2, rel_threshold=0.1):
    """Compute Betti numbers per template via persistent homology.

    Args:
        llm_BCR: shape (B, C, R) — PCA-truncated trajectories per template
        maxdim: max homology dimension to compute
        rel_threshold: relative lifetime threshold for significance

    Returns:
        betti_BH: shape (B, maxdim+1) — Betti numbers per template
        all_dgms_B: list of B diagram lists (for downstream analysis)
    """
    B = llm_BCR.shape[0]
    betti_BH = np.empty((B, maxdim + 1), dtype=int)
    all_dgms_B = []
    for b in range(B):
        result = compute_persistence_CR(llm_BCR[b], maxdim=maxdim)
        dgms = result["dgms"]
        all_dgms_B.append(dgms)
        betti_H, _ = betti_from_diagrams(dgms, rel_threshold=rel_threshold)
        betti_BH[b, :] = betti_H
    return betti_BH, all_dgms_B


def compute_noise_floor(all_dgms_B, maxdim=2):
    """Max lifetime per homology dim across all templates (noise baseline).

    Features in the original data exceeding this floor are topologically significant.

    Returns:
        noise_floor_H: shape (maxdim+1,) — max lifetime per homology dim
    """
    noise_floor_H = np.zeros(maxdim + 1)
    for dgms in all_dgms_B:
        for h, dgm in enumerate(dgms):
            if h > maxdim:
                break
            finite_mask = np.isfinite(dgm[:, 1])
            if np.any(finite_mask):
                lifetimes = dgm[finite_mask, 1] - dgm[finite_mask, 0]
                noise_floor_H[h] = max(noise_floor_H[h], np.max(lifetimes))
    return noise_floor_H


# %%
# Compute persistent homology for per-template and centroid trajectories

betti_og_BH, dgms_og_B = compute_betti_per_template_BH(llm_og_BCR)
betti_rd_BH, dgms_rd_B = compute_betti_per_template_BH(llm_rd_BCR)

result_centroid_og = compute_persistence_CR(centroid_og_CR)
result_centroid_rd = compute_persistence_CR(centroid_rd_CR)

betti_centroid_og_H, lifetimes_centroid_og = betti_from_diagrams(result_centroid_og["dgms"])
betti_centroid_rd_H, lifetimes_centroid_rd = betti_from_diagrams(result_centroid_rd["dgms"])

noise_floor_H = compute_noise_floor(dgms_rd_B)

print(f"{model_name} — {data_name}: Centroid Betti numbers")
print(f"  Original: β₀={betti_centroid_og_H[0]}, β₁={betti_centroid_og_H[1]}, β₂={betti_centroid_og_H[2]}")
print(f"  Random:   β₀={betti_centroid_rd_H[0]}, β₁={betti_centroid_rd_H[1]}, β₂={betti_centroid_rd_H[2]}")
print(f"  Noise floor (max random lifetime): H0={noise_floor_H[0]:.4f}, H1={noise_floor_H[1]:.4f}, H2={noise_floor_H[2]:.4f}")

#%% [markdown]
# ## Persistence Diagrams
# Each point represents a topological feature: its x-coordinate is the filtration
# radius at which it appears (birth), and its y-coordinate is where it disappears
# (death). Points far above the diagonal are long-lived, topologically significant
# features; points near the diagonal are short-lived noise. Dotted horizontal lines
# show the noise floor (max lifetime observed in random baselines per homology dim).
# Structured data should show points clearly above both the diagonal and the noise floor.

# %%


def plot_persistence_diagrams_comparison(dgms_og, dgms_rd, noise_floor_H,
                                         betti_og_H, betti_rd_H,
                                         model_name, data_name):
    """Birth-vs-death scatter for centroid persistence, original vs random."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    labels = ['H₀', 'H₁', 'H₂']

    for ax, dgms, betti_H, title in [
        (axes[0], dgms_og, betti_og_H, f"Original ({data_name})"),
        (axes[1], dgms_rd, betti_rd_H, "Random"),
    ]:
        # Collect all finite points for axis limits
        all_vals = []
        for h, dgm in enumerate(dgms):
            finite_mask = np.isfinite(dgm[:, 1])
            if np.any(finite_mask):
                all_vals.extend(dgm[finite_mask].ravel())

        if all_vals:
            max_val = max(all_vals) * 1.1
        else:
            max_val = 1.0

        # Diagonal line
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)

        for h, dgm in enumerate(dgms):
            finite_mask = np.isfinite(dgm[:, 1])
            dgm_finite = dgm[finite_mask]
            # Infinite features (H0 component that never dies) — plot at top
            inf_mask = ~finite_mask
            if np.any(inf_mask):
                ax.scatter(dgm[inf_mask, 0], [max_val] * np.sum(inf_mask),
                           c=colors[h], marker='^', s=60, alpha=0.8,
                           label=f'{labels[h]} (∞)')
            if len(dgm_finite) > 0:
                ax.scatter(dgm_finite[:, 0], dgm_finite[:, 1],
                           c=colors[h], s=30, alpha=0.7, label=labels[h])

            # Noise floor band
            if h < len(noise_floor_H):
                ax.axhline(y=noise_floor_H[h], color=colors[h], linestyle=':',
                           alpha=0.4, linewidth=1)

        betti_str = ", ".join(f"β{h}={betti_H[h]}" for h in range(len(betti_H)))
        ax.set_title(f"{title}\n{betti_str}")
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{model_name} — {data_name}: Persistence Diagrams (Centroid)", fontsize=13)
    plt.tight_layout()
    plt.show()


plot_persistence_diagrams_comparison(
    result_centroid_og["dgms"], result_centroid_rd["dgms"], noise_floor_H,
    betti_centroid_og_H, betti_centroid_rd_H,
    model_name, data_name)


#%% [markdown]
# ## Persistence Barcodes
# Alternative view of the same persistence data: each horizontal bar spans from
# birth to death of one topological feature. Bars are grouped by homology dimension
# and sorted by lifetime (longest first). Hatched bars represent the single H₀
# component that never dies (the connected component containing all points).
# Long bars = real topological structure; short bars = noise.

# %%


def plot_persistence_barcodes(dgms_og, dgms_rd, betti_og_H, betti_rd_H,
                              model_name, data_name):
    """Horizontal barcode for centroid persistence, original vs random."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    labels = ['H₀', 'H₁', 'H₂']

    for ax, dgms, betti_H, title in [
        (axes[0], dgms_og, betti_og_H, f"Original ({data_name})"),
        (axes[1], dgms_rd, betti_rd_H, "Random"),
    ]:
        y_pos = 0
        yticks = []
        ytick_labels = []

        for h in range(len(dgms) - 1, -1, -1):
            dgm = dgms[h]
            # Sort by lifetime (longest first)
            finite_mask = np.isfinite(dgm[:, 1])
            dgm_finite = dgm[finite_mask]
            if len(dgm_finite) > 0:
                lifetimes = dgm_finite[:, 1] - dgm_finite[:, 0]
                order = np.argsort(-lifetimes)
                for idx in order:
                    ax.barh(y_pos, lifetimes[idx], left=dgm_finite[idx, 0],
                            height=0.8, color=colors[h], alpha=0.7)
                    y_pos += 1
            # Infinite features
            inf_mask = ~finite_mask
            if np.any(inf_mask):
                max_death = max(dgm_finite[:, 1]) if len(dgm_finite) > 0 else 1.0
                for idx in np.where(inf_mask)[0]:
                    ax.barh(y_pos, max_death - dgm[idx, 0], left=dgm[idx, 0],
                            height=0.8, color=colors[h], alpha=0.4, hatch='//')
                    y_pos += 1
            yticks.append(y_pos - 0.5)
            ytick_labels.append(labels[h])
            y_pos += 1  # gap between homology dims

        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
        ax.set_xlabel("Filtration value")
        betti_str = ", ".join(f"β{h}={betti_H[h]}" for h in range(len(betti_H)))
        ax.set_title(f"{title}\n{betti_str}")
        ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle(f"{model_name} — {data_name}: Persistence Barcodes (Centroid)", fontsize=13)
    plt.tight_layout()
    plt.show()


plot_persistence_barcodes(
    result_centroid_og["dgms"], result_centroid_rd["dgms"],
    betti_centroid_og_H, betti_centroid_rd_H,
    model_name, data_name)


#%% [markdown]
# ## Per-Template Betti Number Distributions
# Histograms of Betti numbers across all B templates. Each template's trajectory
# (C points in R-dimensional PCA space) is analyzed independently. β₀ should be 1
# for well-connected trajectories. β₁ > 0 indicates loop structure (expected for
# cyclic concepts like months, days, colors). Differences between original and
# random reveal whether the topological structure is a property of the data or
# an artifact of the embedding geometry.

# %%


def plot_betti_comparison(betti_og_BH, betti_rd_BH, model_name, data_name):
    """Per-template Betti number histograms, one panel per homology dim."""
    maxdim = betti_og_BH.shape[1]
    fig, axes = plt.subplots(1, maxdim, figsize=(5 * maxdim, 4))
    if maxdim == 1:
        axes = [axes]

    for h in range(maxdim):
        ax = axes[h]
        all_vals = np.concatenate([betti_og_BH[:, h], betti_rd_BH[:, h]])
        max_val = max(int(np.max(all_vals)), 1)
        bins = np.arange(-0.5, max_val + 1.5, 1)
        ax.hist(betti_og_BH[:, h], bins=bins, alpha=0.6, color='tab:blue',
                label=f'Original ({data_name})')
        ax.hist(betti_rd_BH[:, h], bins=bins, alpha=0.6, color='tab:orange',
                label='Random')
        ax.set_xlabel(f"β{h}")
        ax.set_ylabel("Count")
        ax.set_title(f"β{h} (H{h})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        max_ticks = 15
        ticks = np.arange(max_val + 1)
        if len(ticks) > max_ticks:
            step = int(np.ceil(len(ticks) / max_ticks))
            ticks = ticks[::step]
        ax.set_xticks(ticks)

    fig.suptitle(f"{model_name} — {data_name}: Per-Template Betti Numbers (R={R})", fontsize=13)
    plt.tight_layout()
    plt.show()


plot_betti_comparison(betti_og_BH, betti_rd_BH, model_name, data_name)


#%% [markdown]
# ## Persistence Lifetime Distributions
# Box plots of all feature lifetimes (death − birth) pooled across templates,
# split by homology dimension. Longer lifetimes indicate more robust topological
# features. If original lifetimes are systematically larger than random, the
# embedding encodes genuine topological structure beyond what random geometry
# produces. Outliers (dots beyond whiskers) are candidate significant features.

# %%


def plot_persistence_lifetimes(dgms_og_B, dgms_rd_B, model_name, data_name, maxdim=2):
    """Box plots of feature lifetimes per homology dim, original vs random."""
    fig, axes = plt.subplots(1, maxdim + 1, figsize=(5 * (maxdim + 1), 4))
    if maxdim == 0:
        axes = [axes]

    for h in range(maxdim + 1):
        ax = axes[h]
        # Collect lifetimes across all templates
        og_lifetimes = []
        rd_lifetimes = []
        for dgms in dgms_og_B:
            dgm = dgms[h]
            finite_mask = np.isfinite(dgm[:, 1])
            if np.any(finite_mask):
                og_lifetimes.extend(dgm[finite_mask, 1] - dgm[finite_mask, 0])
        for dgms in dgms_rd_B:
            dgm = dgms[h]
            finite_mask = np.isfinite(dgm[:, 1])
            if np.any(finite_mask):
                rd_lifetimes.extend(dgm[finite_mask, 1] - dgm[finite_mask, 0])

        data_to_plot = []
        box_labels = []
        if og_lifetimes:
            data_to_plot.append(og_lifetimes)
            box_labels.append(f"Original\n(n={len(og_lifetimes)})")
        if rd_lifetimes:
            data_to_plot.append(rd_lifetimes)
            box_labels.append(f"Random\n(n={len(rd_lifetimes)})")

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=box_labels, patch_artist=True)
            box_colors = ['tab:blue', 'tab:orange']
            for patch, color in zip(bp['boxes'], box_colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
        else:
            ax.text(0.5, 0.5, "No features", ha='center', va='center',
                    transform=ax.transAxes)

        ax.set_title(f"H{h} lifetimes")
        ax.set_ylabel("Lifetime (death − birth)")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{model_name} — {data_name}: Persistence Lifetimes (R={R})", fontsize=13)
    plt.tight_layout()
    plt.show()


plot_persistence_lifetimes(dgms_og_B, dgms_rd_B, model_name, data_name)

# %%
