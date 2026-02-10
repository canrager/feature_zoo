# Text Data

This directory contains template-generated text datasets used for probing semantic structure in LLM hidden states. Downstream experiments (`exp/manifold_metrics.py`, `exp/kernel_decomposition.py`) extract activations from these texts and analyze the geometry of the resulting representations.

## Generation Pipeline

**1. Define elements** — A `.txt` file lists one semantic concept per line (colors, emotions, integers, etc.).

**2. Define templates** — A `_templates.txt` file contains natural-language sentences with a `{placeholder}` slot.

**3. Generate trajectories** — `src/generate_dataset.py` fills every template with every element, producing a `_trajectories.json` file: the final input to the LLM.

**4. Generate random baselines** — `src/random_baseline.py` repeats step 3 but swaps the semantic elements for semantically-diverse words drawn from `random_pool.txt`. This controls for template artifacts.

## File Types

### Elements (`.txt`)

One value per line. These are the concepts whose representations we study.

```
# colors6.txt          # integers100.txt       # emotions200.txt
green                   1                       happy
yellow                  2                       pleased
red                     3                       glad
pink                    4                       delighted
darkblue                5                       joyful
cyan                    6                       elated
```

### Templates (`_templates.txt`)

Numbered sentences with a `{placeholder}` slot. Each domain has its own template set (88 for colors, 100 for emotions, 123 for days, etc.).

```
1→The artist finally decided to paint the bedroom walls {placeholder}
2→According to the manufacturer, the limited edition model will only be available in {placeholder}
3→She couldn't decide between the two dresses, but eventually chose the one in {placeholder}
```

### Trajectories (`_trajectories.json`)

The cartesian product of templates x elements. Each template key maps to an array of `[index, element, filled_sentence]` tuples:

```json
{
  "The artist finally decided to paint the bedroom walls {placeholder}": [
    [0, "green",  "The artist finally decided to paint the bedroom walls green"],
    [1, "yellow", "The artist finally decided to paint the bedroom walls yellow"],
    [2, "red",    "The artist finally decided to paint the bedroom walls red"]
  ]
}
```

### Random Baselines (`random*_trajectories.json`)

Same templates, but elements replaced with words from `random_pool.txt` (1001 diverse English words: *magnets, ratchet, evening, nursery, ...*). Naming convention: `random{N}_from_{source}_trajectories.json`.

### Other Formats

| File | Format | Description |
|------|--------|-------------|
| `days_filtered_21k.csv` | CSV (`label,text`) | Real-world text corpus labeled by weekday |
| `emotional_stories_*.jsonl` | JSONL | Short narratives with per-sentence emotion labels following Russell's circumplex |
| `random_pool.txt` | One word per line | Source pool for random baseline generation |

## Datasets

| Domain | Elements | Templates | Semantic structure |
|--------|----------|-----------|-------------------|
| `colors6` | 6 | 88 | Categorical |
| `colors101` / `colors101_lch` | 101 | 88 | Categorical / circular hue |
| `colors195_lch` | 195 | 88 | Circular hue |
| `colors606` | 606 | 88 | Categorical |
| `days7` | 7 | 123 | Circular (weekday cycle) |
| `emotions200` | 200 | 100 | 2D circumplex (valence x arousal) |
| `integers100` / `500` / `1000` | 100–1000 | 100 | Linear ordinal |
| `integers1000step5` | 200 | 100 | Linear ordinal (step 5) |
| `months12` | 12 | — | Circular (annual cycle) |
| `years100` | 100 | — | Linear temporal |

Each dataset has a corresponding `random{N}_from_{name}` baseline.

## Downstream Usage

Experiments load trajectories via `src/cache_llm.py:load_short_trajectory_acts()`, which:

1. Reads a `_trajectories.json` file
2. Tokenizes all filled sentences
3. Extracts LLM hidden states at a specified layer
4. Reshapes activations into a `B x C x D` tensor (templates x elements x embedding dim)

These activation tensors are then analyzed for manifold geometry, kernel structure, and topological properties.
