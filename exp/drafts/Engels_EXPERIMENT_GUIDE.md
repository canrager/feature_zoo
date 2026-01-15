# Multi-Dimensional SAE Features: Experimental Pipeline

This document provides a detailed explanation of the experimental pipeline for discovering and visualizing multi-dimensional features in Sparse Autoencoders (SAEs) trained on language models.

## Table of Contents
1. [Overview](#overview)
2. [Starting Materials](#starting-materials)
3. [Pipeline Stage 1: SAE Neuron Clustering](#pipeline-stage-1-sae-neuron-clustering)
4. [Pipeline Stage 2: Reducibility Metrics](#pipeline-stage-2-reducibility-metrics)
5. [Pipeline Stage 3: Visualization](#pipeline-stage-3-visualization)
6. [Mathematical Details](#mathematical-details)
7. [Interpretation](#interpretation)

## Overview

This experiment demonstrates that Sparse Autoencoders (SAEs) learn multi-dimensional, non-linear representations of semantic concepts like days of the week, months, and years. The key insight is that these concepts are represented not by single neurons but by **clusters of neurons** whose activations form circular or spiral manifolds in high-dimensional space.

**Research Question**: Do SAEs learn interpretable multi-dimensional features, or only sparse 1D features?

**Answer**: The visualizations show clear evidence of multi-dimensional circular structures for periodic concepts.

## Starting Materials

### 1. Pre-trained SAE Weights

**Source**: [SAELens](https://github.com/jbloomAus/SAELens) - GPT-2 Small SAEs
- **Model**: `gpt2-small-res-jb`
- **Layer**: 7 (residual stream)
- **SAE Architecture**:
  - Input dimension: 768 (GPT-2's residual stream)
  - Hidden dimension: 24,576 (32× expansion)
  - Decoder matrix `W_dec`: shape `[24576, 768]`

**Loading code** ([gpt2_days_months_years.py:26-31](gpt2_days_months_years.py#L26-L31)):
```python
def get_gpt2_sae(device, layer):
    return SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{layer}.hook_resid_pre",
        device=device
    )[0]
```

Each row of `W_dec` represents the **decoder direction** for one SAE neuron - the direction in activation space that the neuron "writes" to.

### 2. LLM Activation Data

**File**: `sae_activations_big_layer-7.npz`

This file contains SAE activations collected from running GPT-2 on a large text corpus. The data structure:

```python
sparse_sae_activations = np.load("sae_activations_big_layer-7.npz")
# Contains:
# - sparse_sae_values: activation values (only non-zero entries)
# - sparse_sae_indices: which SAE neuron fired
# - all_token_indices: which token position in the corpus
# - all_tokens: the actual token IDs
```

**Format**: Sparse representation to save memory
- Instead of storing a `[num_tokens, 24576]` dense matrix
- Stores only the non-zero activations as `(token_idx, neuron_idx, value)` tuples

**Data collection**: The corpus was processed through GPT-2, and at layer 7's residual stream, activations were passed through the SAE encoder to get sparse neuron activations.

---

## Pipeline Stage 1: SAE Neuron Clustering

**Objective**: Group the 24,576 SAE neurons into 1,000 clusters based on similarity of their decoder directions.

**Script**: [clustering.py](sae_multid_feature_discovery/clustering.py)

### Step 1.1: Compute Similarity Matrix

```python
all_sae_features = sae.W_dec  # Shape: [24576, 768]
all_sims = all_sae_features @ all_sae_features.T  # Shape: [24576, 24576]
all_sims.fill_diagonal_(0)
```

**What this computes**: The cosine similarity between every pair of SAE decoder directions.

- `all_sims[i, j]` = cosine similarity between decoder vectors i and j
- Range: [-1, 1] where:
  - 1.0 = vectors point in same direction
  - 0.0 = orthogonal vectors
  - -1.0 = opposite directions

**Intuition**: SAE neurons with similar decoder directions tend to activate in similar contexts and may work together to represent related concepts.

### Step 1.2: Transform to Affinity Measure

```python
all_sims = torch.clamp(all_sims, -1, 1)
all_sims = 1 - torch.arccos(all_sims) / torch.pi
```

**Transformation**:
- `arccos(sim)` gives the angle θ between vectors (in [0, π])
- Dividing by π normalizes to [0, 1]
- Subtracting from 1 makes it an affinity: higher = more similar

**Result**: Affinity matrix where:
- 1.0 = identical directions (0° apart)
- 0.5 = orthogonal (90° apart)
- 0.0 = opposite directions (180° apart)

### Step 1.3: Spectral Clustering

```python
sc = SpectralClustering(n_clusters=1000, affinity="precomputed")
labels = sc.fit_predict(all_sims).tolist()
```

**Algorithm**:
1. Construct graph with SAE neurons as nodes, affinities as edge weights
2. Compute normalized graph Laplacian: `L = D^(-1/2) (D - A) D^(-1/2)`
   - `D` = degree matrix (sum of edge weights per node)
   - `A` = affinity matrix
3. Compute bottom k=1000 eigenvectors of L
4. Run k-means on these eigenvectors to assign cluster labels

**Why spectral clustering?**
- Works well for non-convex clusters
- Captures manifold structure in high dimensions
- Groups neurons that are "connected" in the similarity graph

### Step 1.4: Organize Clusters

```python
clusters = [[] for _ in range(n_clusters)]
for i, label in enumerate(labels):
    clusters[label].append(i)

pickle.dump(clusters, open(f"gpt-2_layer_7_clusters_spectral_n1000.pkl", "wb"))
```

**Output**: A list of 1,000 lists, where `clusters[i]` contains the SAE neuron indices belonging to cluster i.

**Example**:
```python
clusters[138] = [1423, 5672, 8901, ...]  # SAE neurons for "days of week"
clusters[251] = [2341, 7823, 9102, ...]  # SAE neurons for "months"
clusters[212] = [3456, 6789, 10234, ...] # SAE neurons for "years"
```

---

## Pipeline Stage 2: Reducibility Metrics

**Objective**: Quantify which clusters represent multi-dimensional features vs 1D features.

**Script**: [gpt2_compute_reducibility.py](sae_multid_feature_discovery/gpt2_compute_reducibility.py)

### Step 2.1: Reconstruct Activations for a Cluster

**Function**: `get_cluster_activations()` ([gpt2_compute_reducibility.py:43-67](gpt2_compute_reducibility.py#L43-L67))

**Input**:
- Sparse SAE activations from the corpus
- Set of SAE neuron indices in the cluster
- Decoder vectors `W_dec`

**Algorithm**:
```python
for each token in corpus:
    reconstruction = zeros(768)
    for each active SAE neuron i at this token:
        if i in cluster:
            reconstruction += activation_value[i] * W_dec[i]

    if any cluster neurons were active:
        save reconstruction
```

**Mathematical formula**:

For token t, the reconstruction from cluster C is:

```
r_t = Σ_{i ∈ C} a_i(t) · W_dec[i]
```

where:
- `a_i(t)` = activation of SAE neuron i at token t
- `W_dec[i]` = 768-dimensional decoder vector for neuron i
- `r_t` = 768-dimensional reconstruction vector

**Output**:
- `reconstructions`: array of shape `[N, 768]` where N ≤ num_tokens
- Only includes tokens where at least one cluster neuron was active

**Example for cluster 138 (days)**:
```python
reconstructions_days, token_indices_days = get_cluster_activations(
    sparse_sae_activations,
    set(cluster_days),
    decoder_vecs
)
# reconstructions_days.shape = [20000, 768]
# These 20,000 tokens are where day-of-week neurons activated
```

### Step 2.2: PCA Dimensionality Reduction

```python
pca = PCA(n_components=min(5, len(cluster)))
reconstructions_pca = pca.fit_transform(reconstructions)
```

**PCA Algorithm (from first principles)**:

1. **Center the data**:
   ```
   μ = (1/N) Σ_t r_t
   X = [r_1 - μ, r_2 - μ, ..., r_N - μ]  # Shape: [N, 768]
   ```

2. **Compute covariance matrix**:
   ```
   C = (1/(N-1)) X^T X  # Shape: [768, 768]
   ```

3. **Eigendecomposition**:
   ```
   C v_k = λ_k v_k  for k = 1, 2, ..., 768
   ```
   Find eigenvectors v_1, v_2, ..., v_5 with largest eigenvalues

4. **Project data**:
   ```
   V = [v_1, v_2, v_3, v_4, v_5]  # Shape: [768, 5]
   reconstructions_pca = X @ V    # Shape: [N, 5]
   ```

**Result**: Each 768-dim reconstruction is projected to 5 dimensions:
- PC1 (column 0): direction of maximum variance
- PC2 (column 1): direction of 2nd most variance (orthogonal to PC1)
- PC3 (column 2): direction of 3rd most variance
- etc.

### Step 2.3: Compute Mixture Index

**Metric**: `M_ε(f)` - measures if data lies in a low-dimensional subspace

**Algorithm** ([gpt2_compute_reducibility.py:134-161](gpt2_compute_reducibility.py#L134-L161)):

For each pair of consecutive PCs (e.g., PC1-PC2, PC2-PC3):

1. **Find optimal 1D subspace**: Use gradient descent to find vector `a` and offset `b` such that projections `a·x + b` are maximally concentrated near zero

2. **Compute normalized projections**:
   ```python
   proj = (xy @ a + b) / ||a||
   z = proj / sqrt(mean(proj²))
   ```

3. **Measure concentration**:
   ```python
   M_ε = fraction of points where |z| < ε
   ```

**Interpretation**:
- **M_ε ≈ 0**: Data is 2D, doesn't fit in any 1D subspace (good for multi-D features)
- **M_ε ≈ 1**: Data lies along a line (1D feature)

### Step 2.4: Compute Separability Index

**Metric**: `S(f)` - measures minimum mutual information over rotations

**Algorithm** ([gpt2_compute_reducibility.py:117-131](gpt2_compute_reducibility.py#L117-L131)):

For each pair of consecutive PCs:

1. **Rotate the 2D data** by angles θ ∈ [0, 2π]:
   ```python
   R(θ) = [[cos θ, -sin θ],
           [sin θ,  cos θ]]
   xy_rotated = xy @ R(θ)
   ```

2. **Compute 2D histogram**:
   ```python
   hist, _, _ = histogram2d(xy_rotated[:, 0], xy_rotated[:, 1], bins=10)
   ```

3. **Compute mutual information**:
   ```
   p(x,y) = hist / sum(hist)
   p(x) = sum_y p(x,y)
   p(y) = sum_x p(x,y)
   MI = Σ_{x,y} p(x,y) log(p(x,y) / (p(x)p(y)))
   ```

4. **Take minimum over all angles**:
   ```python
   S(f) = min_{θ} MI(θ)
   ```

**Interpretation**:
- **S(f) ≈ 0**: Variables are independent after rotation (separable)
- **S(f) >> 0**: Variables are entangled (non-linear relationship)

### Step 2.5: Rank Clusters

**Script**: [analyze_reducibilities.py](sae_multid_feature_discovery/analyze_reducibilities.py)

**Ranking score**:
```python
score = (1 - mixture_index) × separability_index
```

**Ideal multi-dimensional features have**:
- Low mixture index (not 1D) → (1 - M_ε) is high
- High separability (entangled) → S(f) is high
- High score

**Results**:
- Cluster 138 (days of week): High score, circular structure
- Cluster 251 (months): High score, circular structure
- Cluster 212 (years): High score, spiral structure

These were manually inspected and found to represent temporal concepts.

---

## Pipeline Stage 3: Visualization

**Objective**: Create scatter plots showing the circular/spiral structure of multi-dimensional features.

**Script**: [gpt2_days_months_years.py](sae_multid_feature_discovery/gpt2_days_months_years.py)

### Step 3.1: Load Clusters and Compute Reconstructions

```python
# Load clusters
with open("gpt-2_layer_7_clusters_spectral_n1000.pkl", "rb") as f:
    clusters = pickle.load(f)

cluster_days = clusters[138]  # SAE neuron indices for days cluster

# Load activations
sparse_sae_activations = np.load("sae_activations_big_layer-7.npz")

# Get reconstructions
reconstructions_days, token_indices_days = get_cluster_activations(
    sparse_sae_activations,
    set(cluster_days),
    decoder_vecs
)
```

### Step 3.2: Apply PCA

```python
pca = PCA(n_components=min(5, len(cluster_days)))
reconstructions_pca = pca.fit_transform(reconstructions_days)
# Shape: [20000, 5]
```

**Which PCs to plot?**
- Often **skip PC1** because it captures overall magnitude/common features
- Plot **PC2 vs PC3** (indices 1 and 2) to reveal circular structure

### Step 3.3: Extract Token Context

```python
token_strs_days = tokenizer.batch_decode(sparse_sae_activations['all_tokens'])

contexts_days = []
for token_index in token_indices_days:
    # Get 10 tokens before the current token for context
    contexts_days.append(
        token_strs_days[max(0, token_index-10):token_index]
    )
```

**Why extract context?**
- To see what token triggered the SAE neurons
- The last token in context is what we'll color by

### Step 3.4: Assign Colors Based on Semantic Content

```python
days_of_week = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    # Also include variations: "mon", "mondays", etc.
}

colorwheel = plt.cm.tab10(np.linspace(0, 1, 10))

colors = []
for context in contexts_days:
    token = context[-1]  # Last token before activation
    if token.lower().strip() in days_of_week:
        color = colorwheel[days_of_week[token.lower().strip()]]
    else:
        color = "#BBB"  # Grey for non-day tokens
    colors.append(color)
```

**Coloring scheme**:
- Monday → Color 0 (from colorwheel)
- Tuesday → Color 1
- ...
- Sunday → Color 6
- Other tokens → Grey

### Step 3.5: Create Scatter Plot

```python
plt.scatter(
    reconstructions_pca[:, 1],  # PC2 (x-axis)
    reconstructions_pca[:, 2],  # PC3 (y-axis)
    s=1,                         # Point size
    color=colors,                # Color by day of week
    alpha=0.6                    # Transparency
)
plt.xlabel("PCA axis 2")
plt.ylabel("PCA axis 3")
plt.title("Days of the Week")
```

### Step 3.6: Add Legend

```python
legend_elements = [
    Line2D([0], [0], marker='o', color='w',
           label='Monday', markerfacecolor=colorwheel[0], markersize=3),
    Line2D([0], [0], marker='o', color='w',
           label='Tuesday', markerfacecolor=colorwheel[1], markersize=3),
    # ... etc for all days
    Line2D([0], [0], marker='o', color='w',
           label='Other', markerfacecolor='#BBB', markersize=3)
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=4)
```

### Step 3.7: Repeat for Months and Years

**Months** (cluster 251):
- Same process as days
- 12 colors for 12 months
- Use rainbow colormap

**Years** (cluster 212):
- Filter for tokens that are 4-digit numbers in range [1900, 1999]
- Color by year value on a continuous scale (viridis colormap)
- Plot **PC3 vs PC4** (different axes may capture different patterns)

---

## Mathematical Details

### Why Circular Representations?

**Mathematical reason**: Periodic concepts with period n naturally map to n-dimensional circular representations.

For days of the week (period 7), an ideal representation would be:
```
Monday    → (cos(0·2π/7), sin(0·2π/7))
Tuesday   → (cos(1·2π/7), sin(1·2π/7))
Wednesday → (cos(2·2π/7), sin(2·2π/7))
...
Sunday    → (cos(6·2π/7), sin(6·2π/7))
```

This ensures:
- Each day is equally spaced on a circle
- Sunday wraps back to Monday (periodic)
- Days are not linearly separable

**What PCA reveals**: The SAE has learned something close to this circular representation in the high-dimensional space, and PCA projects it down to 2D where we can see it.

### Why Skip PC1?

PC1 often captures:
- **Overall activation strength**: How strongly any day neuron fired
- **Common features**: Aspects shared by all days (e.g., "this is a temporal concept")

PC2 and PC3 capture the **discriminative circular structure** that distinguishes which day it is.

### Reconstruction Formula

For a token where SAE neurons `i₁, i₂, ..., iₖ` from cluster C are active:

```
reconstruction = Σⱼ aᵢⱼ · W_dec[iⱼ]
```

This is a **weighted sum of decoder directions**, which can be thought of as:
- A point in the 768-dimensional residual stream space
- The model's "internal representation" of this concept
- A vector that, when added to the residual stream, helps predict what comes next

### Why These Clusters Activate Together

**Hypothesis**: The SAE has learned a **compositional code** where:
1. Different neurons in the cluster represent different "angles" on the circle
2. The specific combination of which neurons fire and how strongly encodes the precise position on the circle
3. This allows representing all 7 days with ~5-20 neurons instead of needing 7 separate neurons

**Evidence**: The PCA shows clear separation of days, meaning different combinations of cluster neurons activate for different days.

---

## Interpretation

### What We Learn About SAEs

**Traditional view**: SAEs learn monosemantic features (one neuron = one concept)

**This experiment shows**: SAEs also learn **polysemantic compositions** where:
- Multiple neurons work together
- The pattern of activation encodes position on a manifold
- The manifold structure reflects semantic relationships (cyclic for days/months, sequential for years)

### Why This Matters

1. **Interpretability**: Understanding SAEs requires looking at neuron clusters, not just individual neurons

2. **Capacity**: Multi-dimensional features are more expressive than 1D features
   - 2D circular feature can represent 7 days
   - 7 separate 1D features would be needed otherwise

3. **Geometry**: The model has learned the **geometric structure** of semantic concepts
   - Days form a 7-cycle
   - Months form a 12-cycle
   - Years form a helix (combining sequential and periodic)

### Limitations

1. **Cluster selection**: The choice of clusters 138, 251, 212 involved manual inspection after ranking

2. **Imperfect circles**: The scatter plots show approximate circles with noise, not perfect mathematical circles

3. **PC interpretation**: PCA finds directions of maximum variance, which may not perfectly align with the "true" circular axes

4. **Causality**: We observe correlation between tokens and cluster activations, but don't prove the cluster causes day-of-week behavior

---

## Running the Experiment

### Prerequisites

```bash
pip install sae-lens transformers datasets sklearn matplotlib numpy torch
```

### Step 1: Generate Clusters

```bash
python sae_multid_feature_discovery/clustering.py \
    --model_name gpt-2 \
    --layer 7 \
    --method spectral
```

Output: `gpt-2_layer_7_clusters_spectral_n1000.pkl`

### Step 2: Compute Metrics (for all clusters)

```bash
for cluster_id in {0..999}; do
    python sae_multid_feature_discovery/gpt2_compute_reducibility.py \
        --layer 7 \
        --cluster $cluster_id \
        --save_dir metrics/$cluster_id
done
```

### Step 3: Analyze and Rank

```bash
python sae_multid_feature_discovery/analyze_reducibilities.py
```

Output: Scatter plot of mixture vs separability indices, with top clusters highlighted

### Step 4: Create Visualizations

```bash
python sae_multid_feature_discovery/gpt2_days_months_years.py
```

Output:
- `gpt2nonlinears.pdf` - 3 panel figure showing days, months, years
- `gpt2nonlinears3projs.pdf` - 9 panel figure showing multiple PC projections

---

## Files Reference

- **[clustering.py](sae_multid_feature_discovery/clustering.py)**: Spectral clustering of SAE neurons
- **[gpt2_compute_reducibility.py](sae_multid_feature_discovery/gpt2_compute_reducibility.py)**: Compute mixture and separability metrics
- **[analyze_reducibilities.py](sae_multid_feature_discovery/analyze_reducibilities.py)**: Rank clusters by reducibility scores
- **[gpt2_days_months_years.py](sae_multid_feature_discovery/gpt2_days_months_years.py)**: Create final visualizations

## Citation

If you use this code or build upon this work, please cite the original research paper.

---

**Last updated**: 2025-12-13
