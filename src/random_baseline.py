"""
Utils for generating random baseline datasets for comparison experiments.

Random baselines use the same templates as the original dataset but substitute
semantically diverse random terms instead of the original fill-in elements.
This isolates whether observed structure comes from semantic relationships
or from template artifacts.
"""

import copy
import json
from pathlib import Path
from typing import List

from src.config import Config, DataConfig


def get_random_elements(n: int, pool_path: Path | str | None = None) -> List[str]:
    """Get first N elements from the random pool.

    Args:
        n: Number of elements to get
        pool_path: Path to random_pool.txt. If None, uses default location.

    Returns:
        List of first N terms from the pool
    """
    if pool_path is None:
        pool_path = Path(__file__).parent.parent / "data" / "texts" / "random_pool.txt"
    else:
        pool_path = Path(pool_path)

    with open(pool_path, "r") as f:
        all_elements = [line.strip() for line in f.readlines() if line.strip()]

    if n > len(all_elements):
        raise ValueError(
            f"Requested {n} elements but pool only has {len(all_elements)} elements"
        )

    return all_elements[:n]


def derive_random_config(source_cfg: Config) -> Config:
    """Create a random baseline config from an original config.

    The random config:
    - Uses the same templates as the original
    - Uses first N elements from random pool (where N = source num_elements)
    - Has naming convention: random{N}_from_{source_name}

    Args:
        source_cfg: Original config (e.g., colors6)

    Returns:
        New Config object for the random baseline
    """
    random_cfg = copy.deepcopy(source_cfg)
    n = source_cfg.data.num_elements
    source_name = source_cfg.data.name

    # Update data config fields
    random_cfg.data.name = f"random{n}_from_{source_name}"
    random_cfg.data.elements_filename = f"random{n}_from_{source_name}.txt"
    random_cfg.data.trajectories_filename = f"random{n}_from_{source_name}_trajectories.json"
    # Random baseline reuses the same templates as the source
    random_cfg.data.template_filename = source_cfg.data.template_filename

    return random_cfg


def generate_random_elements_file(source_cfg: Config, output_dir: Path | str | None = None) -> Path:
    """Generate a random elements file for a given source config.

    Args:
        source_cfg: Original config (e.g., colors6)
        output_dir: Directory to write the file. If None, uses default texts_dir.

    Returns:
        Path to the generated elements file
    """
    n = source_cfg.data.num_elements
    source_name = source_cfg.data.name

    if output_dir is None:
        output_dir = Path(source_cfg.env.texts_dir)
    else:
        output_dir = Path(output_dir)

    elements = get_random_elements(n)
    output_path = output_dir / f"random{n}_from_{source_name}.txt"

    with open(output_path, "w") as f:
        for element in elements:
            f.write(f"{element}\n")

    return output_path


def generate_random_trajectories(source_cfg: Config, output_dir: Path | str | None = None) -> Path:
    """Generate trajectories for random baseline using original templates.

    Uses the same templates as the source dataset but fills them with
    random elements from the pool.

    Args:
        source_cfg: Original config (e.g., colors6)
        output_dir: Directory to write the file. If None, uses default texts_dir.

    Returns:
        Path to the generated trajectories file
    """
    n = source_cfg.data.num_elements
    source_name = source_cfg.data.name

    if output_dir is None:
        output_dir = Path(source_cfg.env.texts_dir)
    else:
        output_dir = Path(output_dir)

    # Load original templates (reuse from source dataset)
    templates_path = output_dir / source_cfg.data.template_filename
    with open(templates_path, "r") as f:
        templates = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle numbered format: "N→template"
            if "→" in line:
                template = line.split("→", 1)[1]
            else:
                template = line
            templates.append(template)

    # Get random elements
    random_elements = get_random_elements(n)

    # Generate trajectories (same structure as generate_dataset)
    dataset = {}
    placeholder = "{placeholder}"

    for template in templates:
        filled_strings = []
        for idx, element in enumerate(random_elements):
            filled = template.replace(placeholder, element)
            filled_strings.append([idx, element, filled])
        dataset[template] = filled_strings

    # Save trajectories
    output_path = output_dir / f"random{n}_from_{source_name}_trajectories.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    return output_path


def ensure_random_baseline_exists(source_cfg: Config, force_regenerate: bool = False) -> Config:
    """Ensure random baseline files exist for a source config.

    Generates the random elements file and trajectories file if they don't exist.
    Returns the random baseline config that can be used with load_short_trajectory_acts.

    Args:
        source_cfg: Original config (e.g., colors6)
        force_regenerate: If True, regenerate files even if they exist

    Returns:
        Config object for the random baseline
    """
    random_cfg = derive_random_config(source_cfg)

    texts_dir = Path(source_cfg.env.texts_dir)

    # Check if files exist
    elements_path = texts_dir / random_cfg.data.elements_filename
    trajectories_path = texts_dir / random_cfg.data.trajectories_filename

    # Generate elements file if needed
    if not elements_path.exists() or force_regenerate:
        generate_random_elements_file(source_cfg, texts_dir)
        print(f"Generated: {elements_path}")

    # Generate trajectories file if needed
    if not trajectories_path.exists() or force_regenerate:
        generate_random_trajectories(source_cfg, texts_dir)
        print(f"Generated: {trajectories_path}")

    return random_cfg


if __name__ == "__main__":
    # Test with colors6 config
    from src.config import load_config

    cfg = load_config(overrides=["data=colors6"])
    print(f"Source config: {cfg.data.name}")

    random_cfg = ensure_random_baseline_exists(cfg)
    print(f"Random config: {random_cfg.data.name}")
    print(f"Elements file: {random_cfg.data.elements_filename}")
    print(f"Trajectories file: {random_cfg.data.trajectories_filename}")
