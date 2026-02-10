# %% [markdown]
# # Free Text Trajectory Experiment
#
# This notebook analyzes emotional story trajectories projected onto the emotion UMAP manifold.
# Stories are processed token-by-token, and each token's activation is projected into the
# same 3D UMAP space as the 200 emotion concepts from emotions200.

# %%
# Notebook setup
%cd /home/can/feature_zoo/
%load_ext autoreload
%autoreload 2

# %%
# Initialize Experiment config
from src.config import load_config

# Load emotions200 config for the emotion manifold
emotions_cfg = load_config(overrides=["llm=llama3.1-8b-base-layer15", "data=emotions200"])

# Load emotional_stories config for the stories
stories_cfg = load_config(overrides=["llm=llama3.1-8b-base-layer15", "data=emotional_stories"])

# %%
# Imports
import torch as th
import numpy as np
import umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.cache_llm import load_short_trajectory_acts, load_story_trajectory_acts
from src.random_baseline import ensure_random_baseline_exists


def to_numpy(t: th.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


# %%
# Load emotions200 data
print("Loading emotions200 data...")
emotions_dict = load_short_trajectory_acts(emotions_cfg, force_recompute=False)

# Move to CPU immediately
llm_BCD = to_numpy(emotions_dict["llm_BCD"])
elements_C = emotions_dict["elements_C"]

print(f"Emotions shape: {llm_BCD.shape}")  # (B templates, C concepts, D features)
print(f"Number of emotions: {len(elements_C)}")

# %%
# Load random baseline for finding significant components
print("Loading random baseline...")
random_cfg = ensure_random_baseline_exists(emotions_cfg)
random_dict = load_short_trajectory_acts(random_cfg, force_recompute=False)
random_BCD = to_numpy(random_dict["llm_BCD"])

del emotions_dict, random_dict
th.cuda.empty_cache()


# %%
# Preprocessing functions (same as kernel_decomposition.py)
def preprocess_activations(llm_BCD):
    """Subtract template mean, average over templates, and center across features."""
    llm_BCD = llm_BCD - np.mean(llm_BCD, axis=1, keepdims=True)
    llm_CD = np.mean(llm_BCD, axis=0)
    llm_CD = llm_CD - np.mean(llm_CD, axis=-1, keepdims=True)
    return llm_CD


def find_significant_components(orig_CD, rand_CD):
    """Find number of significant SVD components by comparing normalized singular values."""
    U_orig, S_orig, Vt_orig = np.linalg.svd(orig_CD, full_matrices=False)
    U_rand, S_rand, Vt_rand = np.linalg.svd(rand_CD, full_matrices=False)

    S_orig_norm = S_orig**2 / (S_orig**2).sum()
    S_rand_norm = S_rand**2 / (S_rand**2).sum()

    n_significant = len(S_orig_norm)
    for i in range(1, len(S_orig_norm)):
        if S_orig_norm[i - 1] > S_rand_norm[i - 1] and S_orig_norm[i] < S_rand_norm[i]:
            n_significant = i
            break

    return n_significant, U_orig, S_orig, Vt_orig, U_rand, S_rand, Vt_rand, S_orig_norm, S_rand_norm


# %%
# Compute emotion manifold UMAP with projection components
def compute_emotion_umap_with_projection(llm_BCD, random_BCD, n_sig_override=None):
    """Compute UMAP and return projection components for projecting new data.

    Returns:
        dict with:
        - orig_3d: 3D UMAP coordinates for emotions
        - orig_labels: emotion labels
        - mean_D: global mean for centering new data
        - Vt_truncated: right singular vectors for projection
        - S_truncated: singular values for scaling
        - reducer: fitted UMAP reducer
        - n_sig: number of significant components
    """
    orig_CD = preprocess_activations(llm_BCD)
    rand_CD = preprocess_activations(random_BCD)

    n_sig, U_orig, S_orig, Vt_orig, _, _, _, _, _ = find_significant_components(orig_CD, rand_CD)

    if n_sig_override is not None:
        n_sig = n_sig_override
        print(f"Using {n_sig} significant components (override)")
    else:
        print(f"Found {n_sig} significant components")

    # Store mean for centering new data
    mean_D = np.mean(orig_CD, axis=0)

    # Truncate to significant subspace (scores = U * S)
    orig_truncated = U_orig[:, :n_sig] * S_orig[:n_sig]

    # UMAP to 3D
    reducer = umap.UMAP(n_components=3, n_neighbors=min(15, len(elements_C) - 1), random_state=42)
    orig_3d = reducer.fit_transform(orig_truncated)

    return {
        "orig_3d": orig_3d,
        "orig_labels": elements_C,
        "mean_D": mean_D,
        "Vt_truncated": Vt_orig[:n_sig, :],
        "S_truncated": S_orig[:n_sig],
        "reducer": reducer,
        "n_sig": n_sig,
        "orig_CD": orig_CD,  # Keep for reference
    }


# %%
# Compute emotion UMAP
n_sig_override = 25  # Same as kernel_decomposition.py
umap_data = compute_emotion_umap_with_projection(llm_BCD, random_BCD, n_sig_override=n_sig_override)

print(f"UMAP shape: {umap_data['orig_3d'].shape}")
print(f"Vt_truncated shape: {umap_data['Vt_truncated'].shape}")


# %%
# Load story trajectory activations
print("Loading story trajectory activations...")
story_data = load_story_trajectory_acts(stories_cfg, force_recompute=False)

print(f"Number of stories: {len(story_data['story_acts_list'])}")
for i, acts in enumerate(story_data["story_acts_list"]):
    print(f"  Story {i}: {acts.shape[0]} tokens, labels: {story_data['labels_list'][i][:3]}...")


# %%
# Project story trajectories into UMAP space
def project_story_to_umap(story_acts_TD, umap_data):
    """Project story token activations into the emotion UMAP space.

    Args:
        story_acts_TD: (T, D) tensor of per-token activations
        umap_data: dict from compute_emotion_umap_with_projection

    Returns:
        trajectory_T3: (T, 3) array of UMAP coordinates
    """
    # Convert to numpy
    if isinstance(story_acts_TD, th.Tensor):
        story_TD = to_numpy(story_acts_TD)
    else:
        story_TD = story_acts_TD

    # Center features (same as preprocessing for emotions)
    story_TD = story_TD - np.mean(story_TD, axis=-1, keepdims=True)

    # Center using emotions200's global mean
    story_centered = story_TD - umap_data["mean_D"]

    # Project to SVD basis: (T, D) @ (D, n_sig) -> (T, n_sig)
    story_svd_coords = story_centered @ umap_data["Vt_truncated"].T

    # Project to existing UMAP using transform
    story_3d = umap_data["reducer"].transform(story_svd_coords)

    return story_3d


# Project all stories
story_trajectories = []
for i, acts in enumerate(story_data["story_acts_list"]):
    traj_3d = project_story_to_umap(acts, umap_data)
    story_trajectories.append(traj_3d)
    print(f"Story {i} trajectory shape: {traj_3d.shape}")


# %%
# Select 8 archetype emotions (evenly spaced from 200)
archetype_indices = [0, 25, 50, 75, 100, 125, 150, 175]
archetype_labels = [elements_C[i] for i in archetype_indices]
archetype_coords = umap_data["orig_3d"][archetype_indices]

print("Archetype emotions:")
for i, (idx, label) in enumerate(zip(archetype_indices, archetype_labels)):
    print(f"  {i}: {label} (idx {idx})")


# %%
# Create interactive trajectory plot
def plot_trajectory_with_archetypes(
    story_idx,
    trajectory_T3,
    archetype_coords,
    archetype_labels,
    tokens,
    sentences,
    labels,
    token_to_sentence,
    title="Story Trajectory",
):
    """Create interactive 3D plot with story trajectory and archetype emotions.

    Args:
        story_idx: index of story
        trajectory_T3: (T, 3) trajectory coordinates
        archetype_coords: (8, 3) archetype emotion coordinates
        archetype_labels: list of 8 archetype labels
        tokens: list of token strings
        sentences: list of sentences
        labels: list of emotion labels per sentence
        token_to_sentence: list mapping token idx to sentence idx
    """
    fig = go.Figure()

    # Add archetype emotions as labeled scatter points
    fig.add_trace(
        go.Scatter3d(
            x=archetype_coords[:, 0],
            y=archetype_coords[:, 1],
            z=archetype_coords[:, 2],
            mode="markers+text",
            text=archetype_labels,
            textposition="top center",
            textfont=dict(size=12, color="black"),
            marker=dict(
                size=12,
                color="red",
                symbol="diamond",
                opacity=0.9,
            ),
            name="Archetype Emotions",
            hoverinfo="text",
            hovertext=archetype_labels,
        )
    )

    # Build hover text for trajectory
    hover_texts = []
    for t_idx in range(len(tokens)):
        sent_idx = token_to_sentence[t_idx] if t_idx < len(token_to_sentence) else len(sentences) - 1
        sent_idx = min(sent_idx, len(sentences) - 1)

        current_sentence = sentences[sent_idx]
        current_label = labels[sent_idx] if sent_idx < len(labels) else labels[-1]
        current_token = tokens[t_idx]

        # Build sentence with underlined token
        hover_text = (
            f"<b>Label:</b> {current_label}<br>"
            f"<b>Token:</b> '{current_token}'<br>"
            f"<b>Sentence:</b> {current_sentence}<br>"
            f"<b>Token idx:</b> {t_idx}"
        )
        hover_texts.append(hover_text)

    # Add trajectory as connected scatter with color gradient
    T = len(trajectory_T3)
    fig.add_trace(
        go.Scatter3d(
            x=trajectory_T3[:, 0],
            y=trajectory_T3[:, 1],
            z=trajectory_T3[:, 2],
            mode="lines+markers",
            marker=dict(
                size=4,
                color=np.arange(T),
                colorscale="Viridis",
                opacity=0.8,
                colorbar=dict(title="Token Index", x=1.02),
            ),
            line=dict(
                color="rgba(100, 100, 100, 0.5)",
                width=2,
            ),
            name=f"Story {story_idx} Trajectory",
            hoverinfo="text",
            hovertext=hover_texts,
        )
    )

    # Mark sentence boundaries with larger markers
    sentence_starts = []
    sentence_labels_at_start = []
    current_sent = -1
    for t_idx, sent_idx in enumerate(token_to_sentence):
        if sent_idx != current_sent:
            sentence_starts.append(t_idx)
            sentence_labels_at_start.append(labels[sent_idx] if sent_idx < len(labels) else labels[-1])
            current_sent = sent_idx

    if sentence_starts:
        fig.add_trace(
            go.Scatter3d(
                x=trajectory_T3[sentence_starts, 0],
                y=trajectory_T3[sentence_starts, 1],
                z=trajectory_T3[sentence_starts, 2],
                mode="markers",
                marker=dict(
                    size=8,
                    color="blue",
                    symbol="circle",
                    opacity=1.0,
                    line=dict(width=2, color="white"),
                ),
                name="Sentence Starts",
                hoverinfo="text",
                hovertext=[f"Sentence {i}: {lbl}" for i, lbl in enumerate(sentence_labels_at_start)],
            )
        )

    fig.update_layout(
        title=title,
        height=700,
        width=900,
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
        ),
        legend=dict(x=0, y=1),
    )

    return fig


# %%
# Plot trajectories for each story
for i in range(len(story_trajectories)):
    fig = plot_trajectory_with_archetypes(
        story_idx=i,
        trajectory_T3=story_trajectories[i],
        archetype_coords=archetype_coords,
        archetype_labels=archetype_labels,
        tokens=story_data["tokens_list"][i],
        sentences=story_data["sentences_list"][i],
        labels=story_data["labels_list"][i],
        token_to_sentence=story_data["token_to_sentence_list"][i],
        title=f"Story {i}: Emotional Trajectory in UMAP Space",
    )
    fig.show()


# %%
# Plot all emotions with archetypes highlighted
def plot_emotion_manifold_with_archetypes(umap_data, archetype_indices, archetype_labels):
    """Plot full emotion manifold with archetype emotions highlighted."""
    fig = go.Figure()

    # All emotions (faded)
    all_labels = umap_data["orig_labels"]
    coords_3d = umap_data["orig_3d"]

    # Non-archetype emotions
    non_archetype_mask = np.ones(len(all_labels), dtype=bool)
    non_archetype_mask[archetype_indices] = False

    fig.add_trace(
        go.Scatter3d(
            x=coords_3d[non_archetype_mask, 0],
            y=coords_3d[non_archetype_mask, 1],
            z=coords_3d[non_archetype_mask, 2],
            mode="markers",
            marker=dict(
                size=4,
                color=np.arange(len(all_labels))[non_archetype_mask],
                colorscale="Viridis",
                opacity=0.4,
            ),
            name="All Emotions",
            hoverinfo="text",
            hovertext=[all_labels[i] for i in range(len(all_labels)) if non_archetype_mask[i]],
        )
    )

    # Archetype emotions (highlighted)
    fig.add_trace(
        go.Scatter3d(
            x=coords_3d[archetype_indices, 0],
            y=coords_3d[archetype_indices, 1],
            z=coords_3d[archetype_indices, 2],
            mode="markers+text",
            text=archetype_labels,
            textposition="top center",
            textfont=dict(size=12, color="black"),
            marker=dict(
                size=12,
                color="red",
                symbol="diamond",
                opacity=1.0,
            ),
            name="Archetype Emotions",
            hoverinfo="text",
            hovertext=archetype_labels,
        )
    )

    fig.update_layout(
        title="Emotion Manifold with Archetype Emotions",
        height=700,
        width=900,
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
        ),
    )

    return fig


fig = plot_emotion_manifold_with_archetypes(umap_data, archetype_indices, archetype_labels)
fig.show()


# %%
# Combined plot: emotion manifold + story trajectory
def plot_combined_trajectory_and_manifold(
    story_idx,
    trajectory_T3,
    umap_data,
    archetype_indices,
    archetype_labels,
    tokens,
    sentences,
    labels,
    token_to_sentence,
):
    """Plot story trajectory overlaid on the full emotion manifold."""
    fig = go.Figure()

    coords_3d = umap_data["orig_3d"]
    all_labels = umap_data["orig_labels"]

    # All emotions (very faded)
    fig.add_trace(
        go.Scatter3d(
            x=coords_3d[:, 0],
            y=coords_3d[:, 1],
            z=coords_3d[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color="gray",
                opacity=0.2,
            ),
            name="Emotion Manifold",
            hoverinfo="text",
            hovertext=all_labels,
        )
    )

    # Archetype emotions (highlighted)
    fig.add_trace(
        go.Scatter3d(
            x=coords_3d[archetype_indices, 0],
            y=coords_3d[archetype_indices, 1],
            z=coords_3d[archetype_indices, 2],
            mode="markers+text",
            text=archetype_labels,
            textposition="top center",
            textfont=dict(size=10, color="darkred"),
            marker=dict(
                size=10,
                color="red",
                symbol="diamond",
                opacity=0.9,
            ),
            name="Archetype Emotions",
            hoverinfo="text",
            hovertext=archetype_labels,
        )
    )

    # Build hover text for trajectory
    hover_texts = []
    for t_idx in range(len(tokens)):
        sent_idx = token_to_sentence[t_idx] if t_idx < len(token_to_sentence) else len(sentences) - 1
        sent_idx = min(sent_idx, len(sentences) - 1)

        current_sentence = sentences[sent_idx]
        current_label = labels[sent_idx] if sent_idx < len(labels) else labels[-1]
        current_token = tokens[t_idx]

        hover_text = (
            f"<b>Label:</b> {current_label}<br>"
            f"<b>Token:</b> '{current_token}'<br>"
            f"<b>Sentence:</b> {current_sentence}"
        )
        hover_texts.append(hover_text)

    # Story trajectory
    T = len(trajectory_T3)
    fig.add_trace(
        go.Scatter3d(
            x=trajectory_T3[:, 0],
            y=trajectory_T3[:, 1],
            z=trajectory_T3[:, 2],
            mode="lines+markers",
            marker=dict(
                size=5,
                color=np.arange(T),
                colorscale="Plasma",
                opacity=0.9,
                colorbar=dict(title="Token Index", x=1.02),
            ),
            line=dict(
                color="rgba(50, 50, 50, 0.6)",
                width=3,
            ),
            name=f"Story {story_idx}",
            hoverinfo="text",
            hovertext=hover_texts,
        )
    )

    fig.update_layout(
        title=f"Story {story_idx} Trajectory on Emotion Manifold",
        height=700,
        width=900,
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
        ),
    )

    return fig


# %%
# Plot combined views
for i in range(len(story_trajectories)):
    fig = plot_combined_trajectory_and_manifold(
        story_idx=i,
        trajectory_T3=story_trajectories[i],
        umap_data=umap_data,
        archetype_indices=archetype_indices,
        archetype_labels=archetype_labels,
        tokens=story_data["tokens_list"][i],
        sentences=story_data["sentences_list"][i],
        labels=story_data["labels_list"][i],
        token_to_sentence=story_data["token_to_sentence_list"][i],
    )
    fig.show()


# %%
# ============================================================================
# Baseline: Story-Based UMAP
# ============================================================================
# Instead of projecting stories onto the emotion manifold, here we:
# 1. Fit UMAP on the story data itself
# 2. Project archetype emotions into the story-derived UMAP space
# This provides a baseline to compare with the emotion-centric view above.

# %%
# Concatenate all story activations into a single array
print("Preparing story-based UMAP baseline...")

# Convert story activations to numpy and concatenate
story_acts_list_np = []
story_lengths = []
for acts in story_data["story_acts_list"]:
    if isinstance(acts, th.Tensor):
        acts_np = to_numpy(acts)
    else:
        acts_np = acts
    story_acts_list_np.append(acts_np)
    story_lengths.append(acts_np.shape[0])

# Concatenate all stories: (T_total, D)
all_story_acts = np.concatenate(story_acts_list_np, axis=0)
print(f"Total story tokens: {all_story_acts.shape[0]}, features: {all_story_acts.shape[1]}")


# %%
# Preprocess story activations (center features like in preprocess_activations)
# Center each token's features (subtract mean across features)
all_story_centered = all_story_acts - np.mean(all_story_acts, axis=-1, keepdims=True)

# Compute global mean across all tokens for later use
story_mean_D = np.mean(all_story_centered, axis=0)

# Center across tokens (subtract global mean)
all_story_centered = all_story_centered - story_mean_D

print(f"Story activations preprocessed: {all_story_centered.shape}")


# %%
# Compute SVD on story data
n_sig = umap_data["n_sig"]  # Use same number of components as emotion UMAP
U_story, S_story, Vt_story = np.linalg.svd(all_story_centered, full_matrices=False)

# Truncate to n_sig components
story_svd_coords = U_story[:, :n_sig] * S_story[:n_sig]
Vt_story_truncated = Vt_story[:n_sig, :]

print(f"Story SVD: {story_svd_coords.shape} (using {n_sig} components)")


# %%
# Fit UMAP on story SVD coordinates
print("Fitting UMAP on story data...")
story_reducer = umap.UMAP(
    n_components=3,
    n_neighbors=min(15, len(all_story_centered) - 1),
    random_state=42,
)
story_umap_3d = story_reducer.fit_transform(story_svd_coords)
print(f"Story UMAP shape: {story_umap_3d.shape}")

# Split back into individual story trajectories
story_trajectories_baseline = []
offset = 0
for length in story_lengths:
    story_trajectories_baseline.append(story_umap_3d[offset : offset + length])
    offset += length


# %%
# Project archetype emotions into story-based UMAP space
# Get archetype emotion activations from preprocessed emotions
archetype_acts_CD = umap_data["orig_CD"][archetype_indices]  # (8, D)

# Center using story's global mean (not emotion mean)
archetype_centered = archetype_acts_CD - story_mean_D

# Project through story's SVD basis
archetype_svd_coords = archetype_centered @ Vt_story_truncated.T

# Transform through story-fitted UMAP
archetype_in_story_umap = story_reducer.transform(archetype_svd_coords)

print(f"Archetype emotions projected to story UMAP: {archetype_in_story_umap.shape}")


# %%
# Plot story trajectories and archetypes in story-based UMAP space
def plot_story_based_baseline(
    story_idx,
    trajectory_T3,
    archetype_coords,
    archetype_labels,
    tokens,
    sentences,
    labels,
    token_to_sentence,
    title="Story-Based UMAP Baseline",
):
    """Create interactive 3D plot with story trajectory as native data and archetypes projected."""
    fig = go.Figure()

    # Add archetype emotions (projected into story space)
    fig.add_trace(
        go.Scatter3d(
            x=archetype_coords[:, 0],
            y=archetype_coords[:, 1],
            z=archetype_coords[:, 2],
            mode="markers+text",
            text=archetype_labels,
            textposition="top center",
            textfont=dict(size=12, color="black"),
            marker=dict(
                size=12,
                color="red",
                symbol="diamond",
                opacity=0.9,
            ),
            name="Archetype Emotions (projected)",
            hoverinfo="text",
            hovertext=archetype_labels,
        )
    )

    # Build hover text for trajectory
    hover_texts = []
    for t_idx in range(len(tokens)):
        sent_idx = token_to_sentence[t_idx] if t_idx < len(token_to_sentence) else len(sentences) - 1
        sent_idx = min(sent_idx, len(sentences) - 1)

        current_sentence = sentences[sent_idx]
        current_label = labels[sent_idx] if sent_idx < len(labels) else labels[-1]
        current_token = tokens[t_idx]

        hover_text = (
            f"<b>Label:</b> {current_label}<br>"
            f"<b>Token:</b> '{current_token}'<br>"
            f"<b>Sentence:</b> {current_sentence}<br>"
            f"<b>Token idx:</b> {t_idx}"
        )
        hover_texts.append(hover_text)

    # Add trajectory
    T = len(trajectory_T3)
    fig.add_trace(
        go.Scatter3d(
            x=trajectory_T3[:, 0],
            y=trajectory_T3[:, 1],
            z=trajectory_T3[:, 2],
            mode="lines+markers",
            marker=dict(
                size=4,
                color=np.arange(T),
                colorscale="Viridis",
                opacity=0.8,
                colorbar=dict(title="Token Index", x=1.02),
            ),
            line=dict(
                color="rgba(100, 100, 100, 0.5)",
                width=2,
            ),
            name=f"Story {story_idx} Trajectory (native)",
            hoverinfo="text",
            hovertext=hover_texts,
        )
    )

    # Mark sentence boundaries
    sentence_starts = []
    sentence_labels_at_start = []
    current_sent = -1
    for t_idx, sent_idx in enumerate(token_to_sentence):
        if sent_idx != current_sent:
            sentence_starts.append(t_idx)
            sentence_labels_at_start.append(labels[sent_idx] if sent_idx < len(labels) else labels[-1])
            current_sent = sent_idx

    if sentence_starts:
        fig.add_trace(
            go.Scatter3d(
                x=trajectory_T3[sentence_starts, 0],
                y=trajectory_T3[sentence_starts, 1],
                z=trajectory_T3[sentence_starts, 2],
                mode="markers",
                marker=dict(
                    size=8,
                    color="blue",
                    symbol="circle",
                    opacity=1.0,
                    line=dict(width=2, color="white"),
                ),
                name="Sentence Starts",
                hoverinfo="text",
                hovertext=[f"Sentence {i}: {lbl}" for i, lbl in enumerate(sentence_labels_at_start)],
            )
        )

    fig.update_layout(
        title=title,
        height=700,
        width=900,
        scene=dict(
            xaxis_title="Story UMAP 1",
            yaxis_title="Story UMAP 2",
            zaxis_title="Story UMAP 3",
        ),
        legend=dict(x=0, y=1),
    )

    return fig


# %%
# Plot story-based baseline for each story
print("\n=== Story-Based UMAP Baseline Plots ===")
print("Note: Stories are 'native' here; emotions are projected into story space")
for i in range(len(story_trajectories_baseline)):
    fig = plot_story_based_baseline(
        story_idx=i,
        trajectory_T3=story_trajectories_baseline[i],
        archetype_coords=archetype_in_story_umap,
        archetype_labels=archetype_labels,
        tokens=story_data["tokens_list"][i],
        sentences=story_data["sentences_list"][i],
        labels=story_data["labels_list"][i],
        token_to_sentence=story_data["token_to_sentence_list"][i],
        title=f"Story {i} - Story-Based UMAP (Baseline)",
    )
    fig.show()
