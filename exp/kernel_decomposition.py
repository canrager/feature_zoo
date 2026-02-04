# %% [markdown]
# # Analyzing the weekday representation in language models
#
# All experiment code is in this notebook, functions outside this notebook are only used for caching activations.
#
# This notebook compares original data (e.g., colors6, days7) with random baselines
# to determine if observed structure is due to semantic relationships or template artifacts.

# %%
# Notebook setup
%cd /home/can/feature_zoo/
%load_ext autoreload
%autoreload 2

# %%
# Initialize Experiment config (defines dataset, model, sae)

from src.config import load_config

overrides_list = [
    # ["llm=gpt2", "data=colors101"],
    # ["llm=gpt2", "data=integers100"],
    # ["llm=gpt2", "data=integers1000"],
    # ["llm=gpt2", "data=integers1000step5"],
    ["llm=llama3.1-8b-base", "data=integers100"],
    # ["llm=llama3.1-8b-base", "data=integers1000step5"],
    # ["llm=llama3.1-8b-base", "data=years100"],
    # ["llm=llama3.1-8b-base", "data=integers500"],
    # ["llm=llama3.1-8b-base", "data=integers1000"],
    # ["llm=llama3.1-8b-base", "data=days7"],
    # ["llm=llama3.1-8b-base", "data=months12"],
    # ["llm=llama3.1-8b-base", "data=colors101"],
    # ["llm=olmo3-32b-base", "data=integers100"],
    # ["llm=gpt2", "data=years100"],
    # ["llm=gpt2", "data=colors6"],
    # "llm=llama3.1-8b-base", # Uncomment to switch models or select multiple models.
    # "llm=olmo3-32b-base",
]

# %%
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
# Cache data for all models - both original and random baselines
experiment_data = {}  # {model_name: {"original": {...}, "random": {...}}}

for override in overrides_list:
    # Clear CUDA cache before loading new model
    th.cuda.empty_cache()

    cfg = load_config(overrides=override)
    model_name = cfg.llm.name
    data_name = cfg.data.name

    print(f"Loading {model_name} - {data_name} (original)...")
    original_dict = load_short_trajectory_acts(cfg, force_recompute=False)

    # Generate and load random baseline
    print(f"Loading {model_name} - random baseline...")
    random_cfg = ensure_random_baseline_exists(cfg)
    random_dict = load_short_trajectory_acts(random_cfg, force_recompute=False)

    # Move to CPU immediately to free GPU memory
    def process_return_dict(return_dict):
        if isinstance(return_dict["llm_BCD"], th.Tensor):
            llm_BCD = to_numpy(return_dict["llm_BCD"])
            input_ids_BT = return_dict["input_ids_BT"].cpu() if isinstance(return_dict["input_ids_BT"], th.Tensor) else return_dict["input_ids_BT"]
        else:
            llm_BCD = return_dict["llm_BCD"]
            input_ids_BT = return_dict["input_ids_BT"]

        return {
            "elements_C": return_dict["elements_C"],
            "labels_B": return_dict["labels_B"],
            "texts_B": return_dict["texts_B"],
            "input_ids_BT": input_ids_BT,
            "llm_BCD": llm_BCD,
        }

    experiment_data[model_name] = {
        "original": process_return_dict(original_dict),
        "random": process_return_dict(random_dict),
        "data_name": data_name,
    }

    del original_dict, random_dict
    th.cuda.empty_cache()

    print(f"  {model_name} original llm_BCD shape: {experiment_data[model_name]['original']['llm_BCD'].shape}")
    print(f"  {model_name} random llm_BCD shape: {experiment_data[model_name]['random']['llm_BCD'].shape}")

# %% [markdown]
# The dataset contains short webtext sentence templates that end with a concept (e.g., weekday, color).
# Example: The artist finally decided to paint the bedroom walls {placeholder}
#
# We extract representations at the final token of the fill-in concept. Every template occurs N times
# in the dataset, once for each element.
#
# Importantly, we minimize the template-related information by subtracting the mean per template.
#
# **Random Baseline**: Uses the same templates but replaces semantic elements with random words
# to isolate whether structure comes from meaning or from template artifacts.

# %%
def get_tick_positions_and_labels(labels, num_ticklabels):
    """Return evenly-spaced tick positions and corresponding labels."""
    n = len(labels)
    if n <= num_ticklabels:
        return range(n), labels
    # Select evenly-spaced indices
    indices = np.linspace(0, n - 1, num_ticklabels, dtype=int)
    return indices, [labels[i] for i in indices]


def compute_kernel_and_eigenvalues(llm_BCD):
    """Compute kernel matrix and eigenvalues from activations.

    Args:
        llm_BCD: Activations of shape (templates, concepts, features)

    Returns:
        K_CC: Kernel matrix (concepts x concepts)
        eigenvalues: Eigenvalues of the kernel matrix
        eigenvectors: Eigenvectors of the kernel matrix
    """
    # Subtract mean per template to remove template-specific information
    llm_BCD = llm_BCD - np.mean(llm_BCD, axis=1, keepdims=True)

    # Average over templates
    llm_CD = np.mean(llm_BCD, axis=0)
    llm_CD = llm_CD / np.linalg.norm(llm_CD, axis=-1, keepdims=True) # TODO

    # Compute kernel matrix
    K_CC = llm_CD @ llm_CD.T

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(K_CC)

    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return K_CC, eigenvalues, eigenvectors

# %%
def plot_comparison(model_name, original_data, random_data, data_name, vmin=None, vmax=None, num_ticklabels=10, comp_max=15):
    """Plot side-by-side comparison of original and random baseline kernels.

    Args:
        model_name: Name of the model
        original_data: Dict with original dataset activations
        random_data: Dict with random baseline activations
        data_name: Name of the original dataset (e.g., 'colors6')
        num_ticklabels: Maximum number of tick labels to display (default 10)
    """
    # Compute kernels and eigenvalues
    orig_K, orig_eig, orig_vec = compute_kernel_and_eigenvalues(original_data["llm_BCD"])
    rand_K, rand_eig, rand_vec = compute_kernel_and_eigenvalues(random_data["llm_BCD"])

    # Get labels
    orig_labels = original_data["elements_C"]
    rand_labels = random_data["elements_C"]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{model_name} - {data_name} vs Random Baseline", fontsize=14)

    # Original kernel matrix
    im0 = axes[0].imshow(orig_K, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Original ({data_name})")
    orig_tick_pos, orig_tick_labels = get_tick_positions_and_labels(orig_labels, num_ticklabels)
    axes[0].set_xticks(orig_tick_pos)
    axes[0].set_yticks(orig_tick_pos)
    axes[0].set_xticklabels(orig_tick_labels, rotation=45, ha='right')
    axes[0].set_yticklabels(orig_tick_labels)
    plt.colorbar(im0, ax=axes[0])

    # Random kernel matrix
    im1 = axes[1].imshow(rand_K, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Random Baseline")
    rand_tick_pos, rand_tick_labels = get_tick_positions_and_labels(rand_labels, num_ticklabels)
    axes[1].set_xticks(rand_tick_pos)
    axes[1].set_yticks(rand_tick_pos)
    axes[1].set_xticklabels(rand_tick_labels, rotation=45, ha='right')
    axes[1].set_yticklabels(rand_tick_labels)
    plt.colorbar(im1, ax=axes[1])

    # Relative energy comparison (cumulative)
    orig_rel_energy = orig_eig**2 / np.sum(orig_eig**2)
    rand_rel_energy = rand_eig**2 / np.sum(rand_eig**2)
    orig_cumsum = np.cumsum(orig_rel_energy)
    rand_cumsum = np.cumsum(rand_rel_energy)

    x = np.arange(len(orig_eig))
    axes[2].scatter(x[:comp_max], orig_cumsum[:comp_max], c='tab:blue', marker='o', s=60, label=f'Original ({data_name})')
    axes[2].plot(x[:comp_max], orig_cumsum[:comp_max], c='tab:blue', alpha=0.5)
    axes[2].scatter(x[:comp_max], rand_cumsum[:comp_max], c='tab:orange', marker='s', s=60, label='Random Baseline')
    axes[2].plot(x[:comp_max], rand_cumsum[:comp_max], c='tab:orange', alpha=0.5)
    axes[2].set_xlabel("Eigenvalue Index")
    axes[2].set_ylabel("Cumulative Relative Energy")
    axes[2].set_title("Cumulative Relative Energy")
    axes[2].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    axes[2].set_xticks(x[:comp_max])
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\n{model_name} - {data_name}:")
    print(f"  Original eigenvalues: {orig_eig}")
    print(f"  Random eigenvalues:   {rand_eig}")
    print(f"  Original trace (total variance): {np.sum(orig_eig):.4f}")
    print(f"  Random trace (total variance):   {np.sum(rand_eig):.4f}")
    print(f"  Original top-1 explained variance: {orig_eig[0]/np.sum(orig_eig)*100:.1f}%")
    print(f"  Random top-1 explained variance:   {rand_eig[0]/np.sum(rand_eig)*100:.1f}%")

# %%
# Run comparison for all models
for model_name, data in experiment_data.items():
    plot_comparison(
        model_name,
        data["original"],
        data["random"],
        data["data_name"],
        vmin=None,
        vmax=None,
    )

# %%
# Spectral reordering for similarity matrices
def spectral_reorder(K_CC):
    """Reorder similarity matrix using Fiedler vector (spectral ordering).

    The Fiedler vector (second-smallest eigenvector of the graph Laplacian)
    provides an optimal 1D embedding that minimizes cut distances, revealing
    block structure in similarity matrices.
    """
    # Compute graph Laplacian: L = D - K_CC
    D = np.diag(K_CC.sum(axis=1))
    L = D - K_CC

    # Eigendecompose (smallest eigenvalues first)
    eigvals, eigvecs = np.linalg.eigh(L)

    # Fiedler vector is 2nd eigenvector (1st is constant)
    fiedler = eigvecs[:, 1]

    # Sort by Fiedler vector values
    order = np.argsort(fiedler)
    return order


def plot_comparison_sorted(model_name, original_data, random_data, data_name, vmin=None, vmax=None, num_ticklabels=10):
    """Plot side-by-side comparison with spectral reordering applied.

    Same as plot_comparison but reorders rows/columns using the Fiedler vector
    to reveal block structure in the similarity matrices.

    Args:
        num_ticklabels: Maximum number of tick labels to display (default 10)
    """
    # Compute kernels and eigenvalues
    orig_K, orig_eig, orig_vec = compute_kernel_and_eigenvalues(original_data["llm_BCD"])
    rand_K, rand_eig, rand_vec = compute_kernel_and_eigenvalues(random_data["llm_BCD"])

    # Get labels
    orig_labels = np.array(original_data["elements_C"])
    rand_labels = np.array(random_data["elements_C"])

    # Compute spectral ordering for each matrix
    orig_order = spectral_reorder(orig_K)
    rand_order = spectral_reorder(rand_K)

    # Reorder matrices and labels
    orig_K_sorted = orig_K[orig_order][:, orig_order]
    rand_K_sorted = rand_K[rand_order][:, rand_order]
    orig_labels_sorted = orig_labels[orig_order]
    rand_labels_sorted = rand_labels[rand_order]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{model_name} - {data_name} vs Random (Spectral Reordering)", fontsize=14)

    # Original kernel matrix (reordered)
    im0 = axes[0].imshow(orig_K_sorted, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Original ({data_name}) - Sorted")
    orig_tick_pos, orig_tick_labels = get_tick_positions_and_labels(list(orig_labels_sorted), num_ticklabels)
    axes[0].set_xticks(orig_tick_pos)
    axes[0].set_yticks(orig_tick_pos)
    axes[0].set_xticklabels(orig_tick_labels, rotation=45, ha='right')
    axes[0].set_yticklabels(orig_tick_labels)
    plt.colorbar(im0, ax=axes[0])

    # Random kernel matrix (reordered)
    im1 = axes[1].imshow(rand_K_sorted, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Random Baseline - Sorted")
    rand_tick_pos, rand_tick_labels = get_tick_positions_and_labels(list(rand_labels_sorted), num_ticklabels)
    axes[1].set_xticks(rand_tick_pos)
    axes[1].set_yticks(rand_tick_pos)
    axes[1].set_xticklabels(rand_tick_labels, rotation=45, ha='right')
    axes[1].set_yticklabels(rand_tick_labels)
    plt.colorbar(im1, ax=axes[1])

    # Relative energy comparison (cumulative) - same as original
    orig_rel_energy = orig_eig / np.sum(orig_eig)
    rand_rel_energy = rand_eig / np.sum(rand_eig)
    orig_cumsum = np.cumsum(orig_rel_energy)
    rand_cumsum = np.cumsum(rand_rel_energy)

    x = np.arange(len(orig_eig))
    axes[2].scatter(x, orig_cumsum, c='tab:blue', marker='o', s=60, label=f'Original ({data_name})')
    axes[2].plot(x, orig_cumsum, c='tab:blue', alpha=0.5)
    axes[2].scatter(x, rand_cumsum, c='tab:orange', marker='s', s=60, label='Random Baseline')
    axes[2].plot(x, rand_cumsum, c='tab:orange', alpha=0.5)
    axes[2].set_xlabel("Eigenvalue Index")
    axes[2].set_ylabel("Cumulative Relative Energy")
    axes[2].set_title("Cumulative Relative Energy")
    axes[2].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    axes[2].set_xticks(x)
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print reordering info
    print(f"\n{model_name} - {data_name} (Spectral Reordering):")
    print(f"  Original order: {list(orig_labels_sorted)}")
    print(f"  Random order:   {list(rand_labels_sorted)}")


# %%
# # Run comparison with spectral reordering for all models
# for model_name, data in experiment_data.items():
#     plot_comparison_sorted(
#         model_name,
#         data["original"],
#         data["random"],
#         data["data_name"],
#         vmin=None,
#         vmax=None
#     )

# %%
from typing import List, Tuple

def plot_eigenfunction_projections_multi(
    experiment_data: dict,
    eigenfunction_pairs: List[Tuple[int, int]],
    figsize: tuple | None = None,
    axlim: float | None = None,
    num_plotted_trajectories: int = 20,
    centroid_arrow_mode: str = "cyclic",
    centroid_arrow_size: float = 15,
    centroid_marker_size: float = 150,
):
    """
    Plot 2D projections of data onto kernel eigenfunctions.

    For each model, plots original data (odd rows) and random data (even rows)
    projected onto their respective eigenfunctions. Each column shows a different
    pair of eigenfunction indices.

    Args:
        experiment_data: dict mapping model_name -> {"original": {...}, "random": {...}, "data_name": ...}
        eigenfunction_pairs: List of (idx1, idx2) tuples specifying which eigenfunction pairs to plot.
                            e.g., [(0,1), (1,2), (2,3)] creates 3 columns
        figsize: Optional figure size tuple
        axlim: Fixed axis limit. If None, computed dynamically per subplot.
        num_plotted_trajectories: Number of individual trajectories to plot per concept
        centroid_arrow_mode: "cyclic" (arrows between consecutive concepts), or "none"
        centroid_arrow_size: Size of centroid arrows
    """
    n_cols = len(eigenfunction_pairs)
    n_models = len(experiment_data)
    n_rows = 2 * n_models  # 2 rows per model: original and random

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for model_idx, (model_name, data) in enumerate(experiment_data.items()):
        data_name = data["data_name"]

        for data_type_idx, (data_type, data_dict) in enumerate([("original", data["original"]),
                                                                  ("random", data["random"])]):
            row = model_idx * 2 + data_type_idx

            # Compute kernel eigenvectors for this data
            llm_BCD = data_dict["llm_BCD"]
            labels = data_dict["elements_C"]
            C = len(labels)

            # Subtract mean per template
            llm_BCD_centered = llm_BCD - np.mean(llm_BCD, axis=1, keepdims=True)

            # Get mean representation per concept
            llm_CD = np.mean(llm_BCD_centered, axis=0)

            # Compute kernel and eigenvectors
            K_CC = llm_CD @ llm_CD.T
            eigenvalues, eigenvectors = np.linalg.eigh(K_CC)

            # Sort by descending eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Colors for concepts
            colors = plt.cm.rainbow(np.linspace(0, 1, C))

            for col, (ef_idx1, ef_idx2) in enumerate(eigenfunction_pairs):
                ax = axes[row, col]

                # Project data onto eigenfunctions
                # eigenvectors[:, i] is the i-th eigenvector (C-dimensional)
                # llm_BCD_centered is (B, C, D), we want projection onto eigenvectors
                # The projection of concept c onto eigenfunction i is eigenvectors[c, i]
                # For each template b, the projection is: sum over c of llm_BCD_centered[b,c,:] dotted with something
                # Actually, the eigenvectors are in concept space, so projection is simply eigenvectors[:, i]

                # Project each (template, concept) point:
                # For concept c, template b: project llm_BCD_centered[b, c, :] onto the direction
                # But eigenvectors are C-dimensional (concept space), not D-dimensional (feature space)

                # The natural projection is: for each template b, project the C-dimensional
                # representation (one value per concept) onto the eigenvector
                # proj_B = llm_BCD_centered[b, :, d] @ eigenvectors[:, i] for fixed d

                # Simpler: just use the eigenvector values directly as coordinates
                # proj_x_C = eigenvectors[:, ef_idx1]  # projection of each concept onto eigenfunction ef_idx1
                # proj_y_C = eigenvectors[:, ef_idx2]

                # For individual trajectories, we can project each template's representation
                # proj_x_BC[b, c] = how much does template b's concept c align with eigenfunction ef_idx1
                # This is: (llm_BCD_centered[b, :, :] @ llm_CD.T) @ eigenvectors[:, ef_idx1] / eigenvalues[ef_idx1]
                # Simplified: project in feature space then transform to eigenspace

                # Actually, the cleaner approach:
                # Project each point in feature space onto the principal directions
                # Principal direction in feature space: llm_CD.T @ eigenvectors[:, i] (D-dimensional)
                # Then normalize by sqrt(eigenvalue)

                if eigenvalues[ef_idx1] > 1e-10:
                    principal_dir1 = llm_CD.T @ eigenvectors[:, ef_idx1] / np.sqrt(eigenvalues[ef_idx1])
                else:
                    principal_dir1 = llm_CD.T @ eigenvectors[:, ef_idx1]

                if eigenvalues[ef_idx2] > 1e-10:
                    principal_dir2 = llm_CD.T @ eigenvectors[:, ef_idx2] / np.sqrt(eigenvalues[ef_idx2])
                else:
                    principal_dir2 = llm_CD.T @ eigenvectors[:, ef_idx2]

                # Project all data points: (B, C, D) @ (D,) -> (B, C)
                proj_x_BC = llm_BCD_centered @ principal_dir1
                proj_y_BC = llm_BCD_centered @ principal_dir2

                # Determine axis limits
                if axlim is not None:
                    current_axlim = axlim
                else:
                    current_axlim = np.max(np.abs([proj_x_BC, proj_y_BC])) * 1.1

                # Plot individual trajectories (gray)
                B = proj_x_BC.shape[0]
                for b in range(min(num_plotted_trajectories, B)):
                    x_coords_C = np.clip(proj_x_BC[b], -current_axlim, current_axlim)
                    y_coords_C = np.clip(proj_y_BC[b], -current_axlim, current_axlim)
                    for c in range(C):
                        next_c = (c + 1) % C
                        ax.annotate('', xy=(x_coords_C[next_c], y_coords_C[next_c]),
                                xytext=(x_coords_C[c], y_coords_C[c]),
                                arrowprops=dict(arrowstyle='-', alpha=0.2, color='gray'),
                                annotation_clip=True)
                        ax.scatter(x_coords_C[c], y_coords_C[c], c='grey', s=3, alpha=0.5)

                # Plot mean trajectory (colored centroids)
                mean_x_C = proj_x_BC.mean(axis=0)
                mean_y_C = proj_y_BC.mean(axis=0)
                for c in range(C):
                    label = labels[c] if (col == n_cols - 1 and data_type_idx == 0) else None
                    ax.scatter(mean_x_C[c], mean_y_C[c], c=[colors[c]], s=centroid_marker_size,
                            label=label, zorder=5, edgecolors='black')

                # Draw centroid arrows
                if centroid_arrow_mode == "cyclic":
                    for c in range(C):
                        next_c = (c + 1) % C
                        ax.annotate('', xy=(mean_x_C[next_c], mean_y_C[next_c]),
                                xytext=(mean_x_C[c], mean_y_C[c]),
                                arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                                              mutation_scale=centroid_arrow_size),
                                annotation_clip=True)
                elif centroid_arrow_mode == "sequential":
                    for c in range(C - 1):
                        next_c = c + 1
                        ax.annotate('', xy=(mean_x_C[next_c], mean_y_C[next_c]),
                                xytext=(mean_x_C[c], mean_y_C[c]),
                                arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                                              mutation_scale=centroid_arrow_size),
                                annotation_clip=True)

                ax.set_xlim((-current_axlim, current_axlim))
                ax.set_ylim((-current_axlim, current_axlim))

                # Labels
                ax.set_xlabel(f'Eigenfunction {ef_idx1}')
                if col == 0:
                    row_label = f"{model_name}\n{data_type.capitalize()}"
                    if data_type == "original":
                        row_label = f"{model_name}\n({data_name})"
                    else:
                        row_label = f"{model_name}\n(random)"
                    ax.set_ylabel(f'{row_label}\n\nEigenfunction {ef_idx2}')
                else:
                    ax.set_ylabel(f'Eigenfunction {ef_idx2}')

                if row == 0:
                    ax.set_title(f'EF {ef_idx1} vs {ef_idx2}')

                # Legend on rightmost column for original data rows
                if col == n_cols - 1 and data_type_idx == 0:
                    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=6)

                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)

    plt.suptitle("Projections onto Kernel Eigenfunctions: Original vs Random", fontsize=14)
    plt.tight_layout()
    plt.show()

# %%
# Plot eigenfunction projections for all models
plot_eigenfunction_projections_multi(
    experiment_data,
    eigenfunction_pairs=[(0, 1), (1, 2), (2, 3), (4, 5)],
    # eigenfunction_pairs=[(0, 1), (1, 2), (2, 3), (4, 5), (6, 7)],
    num_plotted_trajectories=0,
    centroid_arrow_mode="none",
    centroid_marker_size=50
)

# %% [markdown]
# ## Eigenvectors as Basis Functions
#
# Following the Feature Fields framework (Yocum et al., 2025), the eigenvectors of the
# kernel matrix K_CC define the **basis functions** of the RKHS of representable functions.
#
# For a feature field f(x,z) = ⟨Φ(x), Ψ(z)⟩, any realization can be written as:
#   f(x, z) = Σ_j √λ_j a_j(x) ψ_j(z)
#
# where ψ_j are the eigenfunctions (eigenvectors when Z is discrete).
#
# **Key diagnostic**: Plot eigenvectors over concept index. Smooth eigenvectors indicate
# the model represents smooth functions over concepts (semantic structure).
# Noisy eigenvectors suggest independent/random treatment of concepts.

# %%
def plot_eigenvectors_as_basis_functions(
    experiment_data: dict,
    n_eigenvectors: int = 4,
    figsize: tuple | None = None,
    show_fourier_comparison: bool = True,
    num_ticklabels: int = 10,
):
    """
    Plot kernel eigenvectors as basis functions over the concept space.

    For each model, compares original vs random data eigenvectors to assess
    whether smoothness reflects semantic structure or template artifacts.

    Args:
        experiment_data: dict with original/random data per model
        n_eigenvectors: number of leading eigenvectors to plot
        figsize: optional figure size
        show_fourier_comparison: if True, overlay Fourier modes for cyclic comparison
        num_ticklabels: maximum number of tick labels to display (default 10)
    """
    n_models = len(experiment_data)

    if figsize is None:
        figsize = (14, 4 * n_models)

    fig, axes = plt.subplots(n_models, 2, figsize=figsize)
    if n_models == 1:
        axes = axes.reshape(1, -1)

    for model_idx, (model_name, data) in enumerate(experiment_data.items()):
        data_name = data["data_name"]

        for col, (data_type, data_dict) in enumerate([("original", data["original"]),
                                                       ("random", data["random"])]):
            ax = axes[model_idx, col]

            # Compute kernel eigenvectors
            llm_BCD = data_dict["llm_BCD"]
            labels = data_dict["elements_C"]
            C = len(labels)

            llm_BCD_centered = llm_BCD - np.mean(llm_BCD, axis=1, keepdims=True)
            llm_CD = np.mean(llm_BCD_centered, axis=0)
            K_CC = llm_CD @ llm_CD.T

            eigenvalues, eigenvectors = np.linalg.eigh(K_CC)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Normalize eigenvalues for display
            rel_energy = eigenvalues / np.sum(np.abs(eigenvalues))

            # X-axis: concept indices
            x = np.arange(C)

            # Plot leading eigenvectors
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_eigenvectors))
            for j in range(min(n_eigenvectors, C)):
                psi_j = eigenvectors[:, j]
                # Normalize for visual comparison
                psi_j = psi_j / np.max(np.abs(psi_j)) if np.max(np.abs(psi_j)) > 0 else psi_j
                ax.plot(x, psi_j, 'o-', color=colors[j], linewidth=2, markersize=8,
                       label=f'ψ_{j} ({rel_energy[j]*100:.1f}%)')

            # Optionally show Fourier modes for comparison (cyclic domains)
            if show_fourier_comparison and C >= 4:
                theta = 2 * np.pi * x / C
                ax.plot(x, np.cos(theta) / np.max(np.abs(np.cos(theta))),
                       '--', color='gray', alpha=0.5, label='cos(2πc/C)')
                if C > 2:
                    ax.plot(x, np.cos(2*theta) / np.max(np.abs(np.cos(2*theta))),
                           ':', color='gray', alpha=0.5, label='cos(4πc/C)')

            tick_pos, tick_labels = get_tick_positions_and_labels(labels, num_ticklabels)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
            ax.set_xlabel('Concept')
            ax.set_ylabel('ψ_j(z) (normalized)')
            ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
            ax.grid(True, alpha=0.3)

            title = f"{model_name} - {data_name}" if data_type == "original" else f"{model_name} - Random"
            ax.set_title(f"{title}\nEigenvectors as Basis Functions")

            # Legend only on rightmost subplot (random column)
            if col == 1:
                ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)

    plt.suptitle("Kernel Eigenvectors: Basis Functions of the Representable Function Space", fontsize=14)
    plt.tight_layout()
    plt.show()

# %%
# plot_eigenvectors_as_basis_functions(experiment_data, n_eigenvectors=4)

# %%
def plot_eigenvectors_separate_rows(
    experiment_data: dict,
    n_eigenvectors: int = 4,
    figsize: tuple | None = None,
    num_ticklabels: int = 10,
    xlim: int | None = None,
    vmin: int | None = None,
    vmax: int | None = None,
):
    """
    Plot kernel eigenvectors as basis functions, one eigenfunction per row.

    Each row shows one eigenfunction, columns are [original data, random data].

    Args:
        experiment_data: dict with original/random data per model
        n_eigenvectors: number of leading eigenvectors to plot (one per row)
        figsize: optional figure size
        num_ticklabels: maximum number of tick labels to display (default 10)
    """
    # For now, assume single model (take first)
    model_name, data = next(iter(experiment_data.items()))
    data_name = data["data_name"]

    n_rows = n_eigenvectors
    n_cols = 2  # original, random

    if figsize is None:
        figsize = (10, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Compute eigenvectors for both original and random
    eigenvector_data = {}
    for data_type, data_dict in [("original", data["original"]), ("random", data["random"])]:
        llm_BCD = data_dict["llm_BCD"]
        labels = data_dict["elements_C"]
        C = len(labels)

        llm_BCD_centered = llm_BCD - np.mean(llm_BCD, axis=1, keepdims=True)
        llm_CD = np.mean(llm_BCD_centered, axis=0)
        K_CC = llm_CD @ llm_CD.T

        eigenvalues, eigenvectors = np.linalg.eigh(K_CC)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Slice to xlim elements if specified
        if xlim is not None:
            labels = labels[:xlim]
            eigenvectors = eigenvectors[:xlim, :]
            C = len(labels)

        rel_energy = eigenvalues / np.sum(np.abs(eigenvalues))

        eigenvector_data[data_type] = {
            "labels": labels,
            "eigenvectors": eigenvectors,
            "rel_energy": rel_energy,
            "C": C,
        }

    # Plot each eigenfunction in a separate row
    for row in range(n_eigenvectors):
        for col, data_type in enumerate(["original", "random"]):
            ax = axes[row, col]

            ev_data = eigenvector_data[data_type]
            labels = ev_data["labels"]
            eigenvectors = ev_data["eigenvectors"]
            rel_energy = ev_data["rel_energy"]
            C = ev_data["C"]

            x = np.arange(C)
            psi_j = eigenvectors[:, row]
            # Normalize for visual comparison
            psi_j = psi_j / np.max(np.abs(psi_j)) if np.max(np.abs(psi_j)) > 0 else psi_j

            color = plt.cm.viridis(0.2 + 0.6 * row / max(n_eigenvectors - 1, 1))
            ax.plot(x, psi_j, 'o-', color=color, linewidth=2, markersize=8)

            tick_pos, tick_labels = get_tick_positions_and_labels(labels, num_ticklabels)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
            ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
            ax.set_ylim((vmin, vmax))
            ax.grid(True, alpha=0.3)

            # Column titles (only on first row)
            if row == 0:
                if data_type == "original":
                    ax.set_title(f"Original ({data_name})")
                else:
                    ax.set_title("Random")

            # Row labels (only on first column)
            if col == 0:
                ax.set_ylabel(f"ψ_{row} ({rel_energy[row]*100:.1f}%)")
            else:
                # Show energy for random too
                ax.text(1.02, 0.5, f"({rel_energy[row]*100:.1f}%)",
                       transform=ax.transAxes, va='center', fontsize=9)

            # X-axis label only on bottom row
            if row == n_eigenvectors - 1:
                ax.set_xlabel('Concept')

    plt.suptitle(f"{model_name}: Eigenvectors as Basis Functions", fontsize=14)
    plt.tight_layout()
    plt.show()

# %%
# plot_eigenvectors_separate_rows(experiment_data, n_eigenvectors=15, xlim=40, vmin=-1, vmax=1)

# %%
def compute_fourier_decomposition(
    eigenvector: np.ndarray,
    n_components: int,
) -> dict:
    """
    Decompose eigenvector into dominant Fourier components.

    Args:
        eigenvector: shape (C,) - the eigenvector to decompose
        n_components: number of dominant frequency components to keep

    Returns:
        dict with:
        - "approximation": reconstructed signal from top n components
        - "frequencies": list of dominant frequency indices (positive only)
        - "error": relative RMSE as percentage
        - "fft_coeffs": full FFT coefficients for reference
    """
    C = len(eigenvector)
    fft_coeffs = np.fft.fft(eigenvector)
    magnitudes = np.abs(fft_coeffs)

    # For real signals, frequencies k and C-k are conjugates
    # We'll select by magnitude but report only positive frequencies
    sorted_indices = np.argsort(magnitudes)[::-1]

    # Keep track of which frequencies are selected
    selected_mask = np.zeros(C, dtype=bool)
    frequencies_selected = []
    n_selected = 0

    for idx in sorted_indices:
        if n_selected >= n_components:
            break
        if selected_mask[idx]:
            continue

        # Select this frequency
        selected_mask[idx] = True
        n_selected += 1

        # Map to canonical frequency (0 to C//2)
        canonical_freq = idx if idx <= C // 2 else C - idx
        if canonical_freq not in frequencies_selected:
            frequencies_selected.append(canonical_freq)

        # For non-DC and non-Nyquist, also select the conjugate
        conjugate_idx = (C - idx) % C
        if conjugate_idx != idx and not selected_mask[conjugate_idx]:
            selected_mask[conjugate_idx] = True

    # Zero out non-selected frequencies
    filtered_coeffs = np.zeros_like(fft_coeffs)
    filtered_coeffs[selected_mask] = fft_coeffs[selected_mask]

    # Inverse FFT to get approximation
    approximation = np.fft.ifft(filtered_coeffs).real

    # Compute relative RMSE
    residual = eigenvector - approximation
    rmse = np.sqrt(np.mean(residual**2))
    norm = np.sqrt(np.mean(eigenvector**2))
    rel_error = (rmse / norm * 100) if norm > 0 else 0.0

    return {
        "approximation": approximation,
        "frequencies": sorted(frequencies_selected),
        "error": rel_error,
        "fft_coeffs": fft_coeffs,
    }


def plot_fourier_approximations(
    experiment_data: dict,
    n_eigenvectors: int = 4,
    n_fourier_components_list: list[int] = [1, 2, 3],
    figsize: tuple | None = None,
    num_ticklabels: int = 10,
    xlim: int | None = None,
):
    """
    Plot eigenvectors with Fourier approximations overlaid.

    For each eigenvector, plots:
    - Original datapoints (scatter)
    - Multiple approximations (lines) for each n in n_fourier_components_list
    - Legend showing frequency indices and approximation error

    Args:
        experiment_data: dict with original/random data per model
        n_eigenvectors: number of leading eigenvectors to plot (one per row)
        n_fourier_components_list: list of component counts for approximations
        figsize: optional figure size
        num_ticklabels: maximum number of tick labels to display
        xlim: optional limit on number of elements to show
    """
    # For now, assume single model (take first)
    model_name, data = next(iter(experiment_data.items()))
    data_name = data["data_name"]

    n_rows = n_eigenvectors
    n_cols = 2  # original, random

    if figsize is None:
        figsize = (12, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Colors for different approximations
    approx_colors = plt.cm.tab10(np.linspace(0, 1, len(n_fourier_components_list)))

    # Compute eigenvectors for both original and random
    eigenvector_data = {}
    for data_type, data_dict in [("original", data["original"]), ("random", data["random"])]:
        llm_BCD = data_dict["llm_BCD"]
        labels = data_dict["elements_C"]
        C = len(labels)

        llm_BCD_centered = llm_BCD - np.mean(llm_BCD, axis=1, keepdims=True)
        llm_CD = np.mean(llm_BCD_centered, axis=0)
        K_CC = llm_CD @ llm_CD.T

        eigenvalues, eigenvectors = np.linalg.eigh(K_CC)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Slice to xlim elements if specified
        if xlim is not None:
            labels = labels[:xlim]
            eigenvectors = eigenvectors[:xlim, :]
            C = len(labels)

        rel_energy = eigenvalues / np.sum(np.abs(eigenvalues))

        eigenvector_data[data_type] = {
            "labels": labels,
            "eigenvectors": eigenvectors,
            "rel_energy": rel_energy,
            "C": C,
        }

    # Plot each eigenfunction in a separate row
    for row in range(n_eigenvectors):
        for col, data_type in enumerate(["original", "random"]):
            ax = axes[row, col]

            ev_data = eigenvector_data[data_type]
            labels = ev_data["labels"]
            eigenvectors = ev_data["eigenvectors"]
            rel_energy = ev_data["rel_energy"]
            C = ev_data["C"]

            x = np.arange(C)
            psi_j = eigenvectors[:, row]
            # Normalize for visual comparison
            psi_j = psi_j / np.max(np.abs(psi_j)) if np.max(np.abs(psi_j)) > 0 else psi_j

            # Plot original as scatter
            # ax.scatter(x, psi_j, color='black', s=30, zorder=10, label='Original', linewidth=0.5)
            ax.plot(x, psi_j, color='black', markersize=3, marker='o', zorder=10, label='Original', linewidth=0.5)

            # Plot Fourier approximations
            for approx_idx, n_components in enumerate(n_fourier_components_list):
                decomp = compute_fourier_decomposition(psi_j, n_components)
                approx = decomp["approximation"]
                freqs = decomp["frequencies"]
                error = decomp["error"]

                freq_str = ",".join(map(str, freqs))
                label = f"k=[{freq_str}], err={error:.1f}%"
                ax.plot(x, approx, '-', color=approx_colors[approx_idx],
                       linewidth=2, alpha=0.8, label=label)

            tick_pos, tick_labels = get_tick_positions_and_labels(labels, num_ticklabels)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
            ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
            # ax.grid(True, alpha=0.3)

            # Column titles (only on first row)
            if row == 0:
                if data_type == "original":
                    ax.set_title(f"Original ({data_name})")
                else:
                    ax.set_title("Random")

            # Row labels (only on first column)
            if col == 0:
                ax.set_ylabel(f"ψ_{row} ({rel_energy[row]*100:.1f}%)")
            else:
                ax.text(1.02, 0.5, f"({rel_energy[row]*100:.1f}%)",
                       transform=ax.transAxes, va='center', fontsize=9)

            # X-axis label only on bottom row
            if row == n_eigenvectors - 1:
                ax.set_xlabel('Concept')

            # Legend (compact, on each subplot)
            ax.legend(loc='upper right', fontsize=7, framealpha=0.9)

    plt.suptitle(f"{model_name}: Eigenvectors with Fourier Approximations", fontsize=14)
    plt.tight_layout()
    plt.show()


# %%
plot_fourier_approximations(experiment_data, n_eigenvectors=6, n_fourier_components_list=[2, 4], xlim=20, num_ticklabels=21)

# %%
def plot_eigenspectrum_comparison(
    experiment_data: dict,
    figsize: tuple | None = None,
    xmax: int | None = None,
):
    """
    Compare eigenvalue spectra between original and random data.

    A steeper drop in eigenvalues indicates lower effective dimensionality,
    suggesting more structured representations.
    """
    n_models = len(experiment_data)

    if figsize is None:
        figsize = (5 * n_models, 4)

    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    for model_idx, (model_name, data) in enumerate(experiment_data.items()):
        ax = axes[model_idx]
        data_name = data["data_name"]

        for data_type, data_dict, color, marker in [("original", data["original"], 'tab:blue', 'o'),
                                                     ("random", data["random"], 'tab:orange', 's')]:
            llm_BCD = data_dict["llm_BCD"]
            C = llm_BCD.shape[1]

            llm_BCD_centered = llm_BCD - np.mean(llm_BCD, axis=1, keepdims=True)
            llm_CD = np.mean(llm_BCD_centered, axis=0)
            K_CC = llm_CD @ llm_CD.T

            eigenvalues, _ = np.linalg.eigh(K_CC)
            eigenvalues = np.sort(eigenvalues)[::-1]
            rel_energy = eigenvalues**2 / np.sum(np.abs(eigenvalues**2))

            x = np.arange(len(eigenvalues[:xmax]))
            rel_energy = rel_energy[:xmax]
            label = f"{data_name}" if data_type == "original" else "Random"
            ax.bar(x + (0.2 if data_type == "original" else -0.2), rel_energy,
                   width=0.4, color=color, alpha=0.7, label=label)

        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel('Relative Energy')
        ax.set_title(f'{model_name}')
        ax.set_xticks(np.arange(min(xmax, C)))
        ax.grid(True, alpha=0.3, axis='y')

        # Legend only on rightmost subplot
        if model_idx == n_models - 1:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.suptitle("Eigenspectrum: Original vs Random", fontsize=14)
    plt.tight_layout()
    plt.show()

# %%
plot_eigenspectrum_comparison(experiment_data, xmax=15)

# %%
def plot_fourier_power_spectrum(
    experiment_data: dict,
    figsize: tuple | None = None,
    normalize: bool = True,
    num_ticklabels: int = 10,
):
    """
    Plot power spectrum of concept representations in Fourier space.

    Computes FFT of mean representations along the concept axis,
    then averages power across feature dimensions.

    Args:
        experiment_data: dict with original/random data per model
        figsize: optional figure size
        normalize: if True, normalize power to sum to 1
        num_ticklabels: maximum number of x-axis tick labels to show
    """
    n_models = len(experiment_data)

    if figsize is None:
        figsize = (5 * n_models, 4)

    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    for model_idx, (model_name, data) in enumerate(experiment_data.items()):
        ax = axes[model_idx]
        data_name = data["data_name"]

        for data_type, data_dict, color in [("original", data["original"], 'tab:blue'),
                                             ("random", data["random"], 'tab:orange')]:
            llm_BCD = data_dict["llm_BCD"]
            C = llm_BCD.shape[1]

            # Center and average across batch
            llm_BCD_centered = llm_BCD - np.mean(llm_BCD, axis=1, keepdims=True)
            llm_CD = np.mean(llm_BCD_centered, axis=0)  # (C, D)

            # FFT along concept axis
            fft_CD = np.fft.fft(llm_CD, axis=0)  # (C, D)

            # Power spectrum: average |FFT|^2 across feature dimensions
            power_C = np.mean(np.abs(fft_CD) ** 2, axis=1)  # (C,)

            # Only keep frequencies 0 to C//2 (due to conjugate symmetry for real input)
            n_freqs = C // 2 + 1
            power_C = power_C[:n_freqs]

            if normalize:
                power_C = power_C / np.sum(power_C)

            x = np.arange(n_freqs)
            width = 0.35
            offset = -width / 2 if data_type == "original" else width / 2
            label = f"{data_name}" if data_type == "original" else "Random"
            ax.bar(x + offset, power_C, width=width, color=color, alpha=0.7, label=label)

        ax.set_xlabel('Frequency Index')
        ax.set_ylabel('Relative Power' if normalize else 'Power')
        ax.set_title(f'{model_name}')
        if n_freqs <= num_ticklabels:
            ax.set_xticks(np.arange(n_freqs))
        else:
            tick_step = max(1, n_freqs // num_ticklabels)
            ax.set_xticks(np.arange(0, n_freqs, tick_step))
        ax.grid(True, alpha=0.3, axis='y')

        # Legend only on rightmost subplot
        if model_idx == n_models - 1:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.suptitle("Fourier Power Spectrum: Original vs Random", fontsize=14)
    plt.tight_layout()
    plt.show()


# %%
plot_fourier_power_spectrum(experiment_data)

# %% [markdown]
# ## Fitting Continuous Embeddings for Cyclic Domains
#
# For cyclic domains like weekdays, we can fit a continuous embedding Ψ: S¹ → ℝ^D
# using Fourier features. This reveals:
# - How smooth the embedding is as a function of angle
# - What continuous functions over the cycle are representable




# %%
def fit_fourier_embedding(
    experiment_data: dict,
    n_harmonics: int = 3,
    n_interpolation_points: int = 100,
    figsize: tuple | None = None,
):
    """
    Fit continuous Fourier embeddings to discrete concept embeddings.

    For cyclic domains, fits Ψ(θ) = Σ_k [A_k cos(kθ) + B_k sin(kθ)]
    and visualizes the continuous kernel K(θ, θ').

    Compares original data vs random baseline to assess whether smooth
    structure reflects semantic relationships or template artifacts.

    Args:
        experiment_data: dict with original/random data per model
        n_harmonics: number of Fourier harmonics to use
        n_interpolation_points: points for continuous interpolation
        figsize: optional figure size
    """
    n_models = len(experiment_data)
    n_rows = 2 * n_models  # 2 rows per model: original and random

    if figsize is None:
        figsize = (12, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Define fourier_features outside the loop to avoid redefining it
    def fourier_features(theta, n_harm):
        """Return Fourier features for angle(s) theta."""
        theta = np.atleast_1d(theta)
        features = [np.ones_like(theta)]  # DC component
        for k in range(1, n_harm + 1):
            features.append(np.cos(k * theta))
            features.append(np.sin(k * theta))
        return np.stack(features, axis=-1)  # (n_points, 2*n_harm + 1)

    for model_idx, (model_name, data) in enumerate(experiment_data.items()):
        data_name = data["data_name"]

        for data_type_idx, (data_type, data_dict) in enumerate([("original", data["original"]),
                                                                  ("random", data["random"])]):
            row = model_idx * 2 + data_type_idx

            llm_BCD = data_dict["llm_BCD"]
            labels = data_dict["elements_C"]
            C = len(labels)
            D = llm_BCD.shape[2]

            llm_BCD_centered = llm_BCD - np.mean(llm_BCD, axis=1, keepdims=True)
            llm_CD = np.mean(llm_BCD_centered, axis=0)  # (C, D)

            # Assign angles to concepts (cyclic)
            theta_discrete = 2 * np.pi * np.arange(C) / C  # (C,)

            # Design matrix for discrete points
            F_discrete = fourier_features(theta_discrete, n_harmonics)  # (C, 2*n_harm+1)

            # Fit coefficients: llm_CD ≈ F_discrete @ Coeffs
            # Coeffs = (F^T F)^{-1} F^T llm_CD
            Coeffs = np.linalg.lstsq(F_discrete, llm_CD, rcond=None)[0]  # (2*n_harm+1, D)

            # Continuous interpolation
            theta_continuous = np.linspace(0, 2*np.pi, n_interpolation_points, endpoint=False)
            F_continuous = fourier_features(theta_continuous, n_harmonics)
            Psi_continuous = F_continuous @ Coeffs  # (n_points, D)

            # Compute continuous kernel
            K_continuous = Psi_continuous @ Psi_continuous.T  # (n_points, n_points)

            # Reconstruction error
            Psi_reconstructed = F_discrete @ Coeffs
            reconstruction_error = np.linalg.norm(llm_CD - Psi_reconstructed) / np.linalg.norm(llm_CD)

            # Row label for y-axis
            if data_type == "original":
                row_label = f"{model_name}\n({data_name})"
            else:
                row_label = f"{model_name}\n(random)"

            # Plot 1: Continuous kernel
            ax0 = axes[row, 0]
            im = ax0.imshow(K_continuous, extent=[0, 360, 360, 0], aspect='auto', cmap='RdBu_r')
            plt.colorbar(im, ax=ax0)
            ax0.set_xlabel('θ\' (degrees)')
            ax0.set_ylabel(f'{row_label}\n\nθ (degrees)')
            if row == 0:
                ax0.set_title('Continuous Kernel K(θ,θ\')')

            # Plot 2: Kernel as function of angle difference (should be stationary for cyclic)
            ax1 = axes[row, 1]
            # Extract diagonal slices
            angle_diffs = []
            kernel_values = []
            for i in range(n_interpolation_points):
                for j in range(n_interpolation_points):
                    diff = (theta_continuous[j] - theta_continuous[i]) % (2 * np.pi)
                    if diff > np.pi:
                        diff = 2 * np.pi - diff
                    angle_diffs.append(diff * 180 / np.pi)
                    kernel_values.append(K_continuous[i, j])

            ax1.scatter(angle_diffs, kernel_values, alpha=0.1, s=1)
            ax1.set_xlabel('|θ - θ\'| (degrees)')
            ax1.set_ylabel('K(θ, θ\')')
            if row == 0:
                ax1.set_title('Kernel vs Angle Difference')
            ax1.text(0.95, 0.95, f'Error: {reconstruction_error:.3f}',
                    transform=ax1.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax1.grid(True, alpha=0.3)

            # Plot 3: First few PCA components of continuous embedding
            ax2 = axes[row, 2]
            U, S, Vh = np.linalg.svd(Psi_continuous, full_matrices=False)

            # Plot trajectory in PC1-PC2 space
            pc1 = U[:, 0] * S[0]
            pc2 = U[:, 1] * S[1]
            ax2.plot(pc1, pc2, 'b-', alpha=0.5, linewidth=1)

            # Recompute for discrete points
            Psi_discrete_proj = Psi_reconstructed @ Vh.T[:, :2]
            colors = plt.cm.rainbow(np.linspace(0, 1, C))
            for c in range(C):
                ax2.scatter(Psi_discrete_proj[c, 0], Psi_discrete_proj[c, 1],
                           c=[colors[c]], s=150, edgecolors='black', zorder=5)
                ax2.annotate(labels[c], (Psi_discrete_proj[c, 0], Psi_discrete_proj[c, 1]),
                            fontsize=8, ha='center', va='bottom')

            ax2.set_xlabel('PC1')
            ax2.set_ylabel('PC2')
            if row == 0:
                ax2.set_title(f'Continuous Embedding (Fourier, k≤{n_harmonics})')
            ax2.set_aspect('equal')
            ax2.grid(True, alpha=0.3)

    plt.suptitle("Continuous Fourier Embedding: Original vs Random", fontsize=14)
    plt.tight_layout()
    plt.show()

# %%
fit_fourier_embedding(experiment_data, n_harmonics=10)

# %% [markdown]
# ## Interpretation
#
# If the original data shows strong structure (high eigenvalue concentration, clear clustering)
# but the random baseline does NOT, then the structure is likely due to **semantic relationships**
# between the fill-in terms.
#
# If BOTH show similar structure, then the structure might be due to **template artifacts**
# (e.g., certain templates producing similar activations regardless of fill-in content).