# Feature Fields and Kernel Geometry

Notes connecting kernel decomposition analysis to the Feature Fields framework (Yocum et al., 2025).

## 1. The Feature Fields Framework

### Core Idea

Traditional interpretability treats features as **scalar-valued**: a single number extracted from activations via a linear probe. The feature fields framework generalizes this to **function-valued** features defined over a domain space Z.

**Feature Variable (scalar):**
```
f_i(x) = ⟨Φ(x), Ψ_i⟩
```
- Φ(x) ∈ ℝ^d = neural activation for input x
- Ψ_i ∈ ℝ^d = probe direction (a single vector)
- Output: a scalar

**Feature Field (function-valued):**
```
f(x, z) = ⟨Φ(x), Ψ(z)⟩   for all z ∈ Z
```
- Ψ: Z → ℝ^d = domain embedding (a continuous map)
- Output: a function over Z

### The Domain Embedding

The key object is Ψ(Z) ⊆ ℝ^d, the image of the domain space in activation space. For weekdays with Z = {Mon, Tue, ..., Sun}, this is 7 vectors in activation space.

**Theorem (Domain Homeomorphism):** If the feature field is continuous and satisfies mild conditions, then Ψ(Z) ≅ Z topologically. A circle domain embeds as a circle in activation space.

### The Feature Kernel and RKHS

The geometry of Ψ(Z) determines what functions can be represented:

```
K(z, z') = ⟨Ψ(z), Ψ(z')⟩
```

This kernel defines a Reproducing Kernel Hilbert Space H_K. The key theorem states:

**Theorem (Field Geometry Equivalence):** The space of linearly representable realizations equals the RKHS H_K, with basis functions:
```
ψ_j(z) = (1/√λ_j) ⟨Ψ(z), e_j⟩
```
where e_j are principal directions of the embedding and λ_j are eigenvalues.

## 2. Connecting to Kernel Decomposition

In `kernel_decomposition.py`, we compute:
```python
llm_CD = mean activations per concept  # This is Ψ(Z) in the paper's notation
K_CC = llm_CD @ llm_CD.T               # This is the feature kernel K(z,z')
eigenvalues, eigenvectors = eigh(K_CC) # These give the basis functions
```

### Interpretation

| Our Code | Paper Notation | Meaning |
|----------|----------------|---------|
| `llm_CD[c, :]` | Ψ(z_c) | Embedding of concept c in activation space |
| `K_CC[c, c']` | K(z_c, z_c') | Inner product of embeddings |
| `eigenvectors[:, j]` | ψ_j(z) | j-th basis function over concepts |
| `eigenvalues[j]` | λ_j | Variance explained by j-th basis |

### Eigenvectors as Basis Functions

The eigenvectors of K_CC, when viewed as functions over the concept index, are the **basis functions** of the RKHS. Any linearly representable function over concepts can be written as:

```
f(z) = Σ_j √λ_j a_j ψ_j(z)
```

For weekdays, if ψ_1 looks like a sinusoid with period 7, that means "smooth cyclic functions" are representable.

## 3. Fitting Continuous Embeddings for Cyclic Domains

### Why Go Continuous?

For discrete concepts like weekdays, we only have 7 points. But weekdays have implicit cyclic structure: Sunday is "close" to Monday. To leverage this:

1. **Parameterize Z as S¹**: Map weekdays to angles θ_c = 2πc/7
2. **Fit a continuous embedding**: Ψ(θ) that interpolates between discrete points
3. **Analyze the continuous kernel**: K(θ, θ') = ⟨Ψ(θ), Ψ(θ')⟩

### Fourier Basis Approach

For cyclic domains, a natural parameterization uses Fourier features:

```python
def continuous_embedding(theta, coefficients):
    """
    Ψ(θ) = Σ_k [a_k cos(kθ) + b_k sin(kθ)] · v_k

    where v_k are directions in activation space.
    """
    features = []
    for k in range(num_harmonics):
        features.extend([np.cos(k * theta), np.sin(k * theta)])
    return coefficients @ np.array(features)
```

### Fitting Procedure

1. **Start with discrete embeddings**: Ψ(z_c) = llm_CD[c, :] for c = 0, ..., 6
2. **Assign angles**: θ_c = 2πc/7
3. **Fit Fourier coefficients**: Minimize ||Ψ_continuous(θ_c) - Ψ_discrete(c)||²
4. **Evaluate continuous kernel**: Now K(θ, θ') is defined for any angles

### What This Reveals

- **Effective dimensionality**: How many Fourier harmonics are needed?
- **Smoothness**: Does the embedding vary smoothly with θ?
- **Representable functions**: The RKHS of the fitted kernel

## 4. Checking Eigenfunction Smoothness

### The Key Diagnostic

Plot each eigenvector ψ_j as a function over concept index (or angle for cyclic domains):

```
ψ_j: {0, 1, 2, ..., C-1} → ℝ
     c ↦ eigenvectors[c, j]
```

### What to Look For

**Smooth eigenfunctions** (low-frequency variation):
- Indicate the model represents smooth functions over concepts
- Suggest semantic similarity is encoded geometrically
- For cyclic domains: should look like sinusoids

**Non-smooth eigenfunctions** (high-frequency/random):
- Indicate the model treats concepts more independently
- May arise from template artifacts rather than semantic structure
- Random baseline should show this pattern

### Comparing Original vs Random

If original data shows smooth eigenfunctions but random data shows noisy ones:
→ The smoothness reflects **semantic structure**, not template artifacts

If both show similar patterns:
→ Structure may come from **template effects**

## 5. Example Analysis Pipeline

```python
# 1. Compute kernel and eigenvectors
K_CC = llm_CD @ llm_CD.T
eigenvalues, eigenvectors = np.linalg.eigh(K_CC)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

# 2. Plot eigenvectors as basis functions
for j in range(min(4, len(eigenvalues))):
    plt.plot(eigenvectors[:, j], label=f'ψ_{j} (λ={eigenvalues[j]:.3f})')

# 3. For cyclic domains, check if eigenvectors match Fourier modes
theta = np.linspace(0, 2*np.pi, C, endpoint=False)
# Compare ψ_j to cos(k·θ) and sin(k·θ) patterns

# 4. Compute explained variance
rel_energy = eigenvalues / eigenvalues.sum()
# If first 2-3 eigenfunctions capture most variance → low-dimensional structure
```

## 6. Key Insights

1. **Geometry encodes function space**: The shape of Ψ(Z) in activation space determines which functions f(z) can be linearly read out.

2. **Eigenfunctions are basis functions**: The principal components of the kernel matrix give an orthonormal basis for representable functions.

3. **Smoothness implies structure**: If leading eigenfunctions are smooth over the domain, the network has learned to represent smooth relationships between concepts.

4. **Dimension of H_K ≤ d**: At most d basis functions can be non-trivial, where d = activation dimension. In practice, often much fewer are significant.

5. **Random baseline test**: Comparing eigenfunction structure between semantic and random concepts isolates what structure comes from meaning vs. artifacts.
