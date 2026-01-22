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
    "llm=gpt2",
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

    cfg = load_config(overrides=[override])
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
def plot_comparison(model_name, original_data, random_data, data_name):
    """Plot side-by-side comparison of original and random baseline kernels.

    Args:
        model_name: Name of the model
        original_data: Dict with original dataset activations
        random_data: Dict with random baseline activations
        data_name: Name of the original dataset (e.g., 'colors6')
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
    im0 = axes[0].imshow(orig_K, cmap='RdBu_r')
    axes[0].set_title(f"Original ({data_name})")
    axes[0].set_xticks(range(len(orig_labels)))
    axes[0].set_yticks(range(len(orig_labels)))
    axes[0].set_xticklabels(orig_labels, rotation=45, ha='right')
    axes[0].set_yticklabels(orig_labels)
    plt.colorbar(im0, ax=axes[0])

    # Random kernel matrix
    im1 = axes[1].imshow(rand_K, cmap='RdBu_r')
    axes[1].set_title(f"Random Baseline")
    axes[1].set_xticks(range(len(rand_labels)))
    axes[1].set_yticks(range(len(rand_labels)))
    axes[1].set_xticklabels(rand_labels, rotation=45, ha='right')
    axes[1].set_yticklabels(rand_labels)
    plt.colorbar(im1, ax=axes[1])

    # Eigenvalue comparison
    x = np.arange(1, len(orig_eig) + 1)
    axes[2].plot(x, orig_eig, 'o-', label=f'Original ({data_name})', linewidth=2, markersize=8)
    axes[2].plot(x, rand_eig, 's--', label='Random Baseline', linewidth=2, markersize=8)
    axes[2].set_xlabel("Eigenvalue Index")
    axes[2].set_ylabel("Eigenvalue")
    axes[2].set_title("Eigenvalue Spectrum")
    axes[2].legend()
    axes[2].set_xticks(x)
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
        data["data_name"]
    )

# %% [markdown]
# ## Interpretation
#
# If the original data shows strong structure (high eigenvalue concentration, clear clustering)
# but the random baseline does NOT, then the structure is likely due to **semantic relationships**
# between the fill-in terms.
#
# If BOTH show similar structure, then the structure might be due to **template artifacts**
# (e.g., certain templates producing similar activations regardless of fill-in content).

# %%
# Individual kernel matrices (for detailed inspection)
for model_name, data in experiment_data.items():
    orig_labels = data["original"]["elements_C"]
    print(f"\n{model_name} - Original labels: {orig_labels}")
    print(f"{model_name} - Random labels: {data['random']['elements_C']}")

    llm_BCD = data["original"]["llm_BCD"]
    llm_BCD = llm_BCD - np.mean(llm_BCD, axis=1, keepdims=True)
    llm_CD = np.mean(llm_BCD, axis=0)
    K_CC = llm_CD @ llm_CD.T

    plt.figure(figsize=(6, 5))
    plt.imshow(K_CC)
    plt.xticks(range(len(orig_labels)), orig_labels, rotation=45, ha='right')
    plt.yticks(range(len(orig_labels)), orig_labels)
    plt.colorbar()
    plt.title(f"{model_name} - {data['data_name']} Kernel")
    plt.tight_layout()
    plt.show()

# %%
# Eigenvalue analysis
for model_name, data in experiment_data.items():
    llm_BCD = data["original"]["llm_BCD"]
    llm_BCD = llm_BCD - np.mean(llm_BCD, axis=1, keepdims=True)
    llm_CD = np.mean(llm_BCD, axis=0)
    K_CC = llm_CD @ llm_CD.T

    eigenvalues, eigenvectors = np.linalg.eigh(K_CC)
    eigenvalues = eigenvalues[::-1]  # Sort descending

    plt.figure(figsize=(8, 4))
    plt.plot(eigenvalues, 'o-')
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Eigenvalue")
    plt.title(f"{model_name} - {data['data_name']} Eigenvalue Spectrum")
    plt.grid(True, alpha=0.3)
    plt.show()
# %%
