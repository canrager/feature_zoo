import json
from itertools import product
from pathlib import Path
from config import load_config


def load_categories_from_txt(txt_path: str) -> list[str]:
    """
    Load category values from a text file (one value per line).

    Args:
        txt_path: Path to the text file containing category values

    Returns:
        List of category values (stripped, non-empty lines)
    """
    categories = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                categories.append(line)
    return categories


def generate_dataset(
    template_txt_path: str | None = None,
    output_json_path: str | None = None,
    categories: dict[str, list[str] | None] | None = None,
    placeholder: str = "{placeholder}",
    output_order: list[str] | None = None,
    cfg=None,
) -> None:
    """
    Generate a dataset from template text file with placeholder substitution.

    This version uses a single placeholder in the template and concatenates
    multiple category values to fill it.

    Args:
        template_txt_path: Path to the template text file (optional if cfg is provided)
        output_json_path: Path for the output JSON file (optional if cfg is provided)
        categories: Ordered dict of category names to lists of values (optional if cfg is provided)
                   The order determines hierarchy (last changes fastest)
                   (e.g., {'weekday': [...], 'time': [...], 'adverb': [...]})
                   Use None to skip a category entirely.
        placeholder: The placeholder string in templates (default: "{placeholder}")
        output_order: Optional list specifying the order of categories in the output string
                     (e.g., ['adverb', 'weekday', 'time'] for "early Monday morning")
                     If None, uses the order from categories dict.
        cfg: Optional Config object. If provided, paths and categories are derived from cfg.filter.regex_file
    """
    # If config is provided, derive paths and categories from it
    if cfg is not None:
        regex_file = cfg.filter.regex_file
        root_dir = Path(__file__).parent.parent

        # Construct paths based on regex_file
        category_txt_path = root_dir / cfg.env.texts_dir / f"{regex_file}.txt"
        template_txt_path = root_dir / cfg.env.texts_dir / f"{regex_file}_templates.txt"
        output_json_path = root_dir / cfg.env.texts_dir / f"{regex_file}_trajectories.json"

        # Load categories from the text file
        category_values = load_categories_from_txt(str(category_txt_path))
        categories = {regex_file: category_values}
        output_order = [regex_file]
    # Read templates from text file
    templates = []
    with open(template_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split by → separator if present (numbered format), otherwise use the whole line
            if "→" in line:
                template = line.split("→", 1)[1]
            else:
                template = line
            templates.append(template)

    # Filter out None values from categories
    active_categories = {k: v for k, v in categories.items() if v is not None}

    # Get category names and values in order
    category_names = list(active_categories.keys())
    category_values = list(active_categories.values())

    # Determine output order
    if output_order is None:
        output_order = category_names
    else:
        # Filter output_order to only include active categories
        output_order = [name for name in output_order if name in active_categories]

    # Generate all combinations in hierarchical order
    # product() makes the rightmost argument change fastest,
    # so pass categories in order (last category changes fastest)
    combinations = list(product(*category_values)) if category_values else [()]

    # Build the dataset
    dataset = {}

    for template in templates:
        filled_strings = []

        for idx, combo in enumerate(combinations):
            # Create dict mapping category names to values
            combo_dict = {name: value for name, value in zip(category_names, combo)}

            # Concatenate values in the specified output order
            temporal_parts = [combo_dict[name] for name in output_order]
            temporal_value = " ".join(temporal_parts).strip()

            # Fill in the template
            filled = template.replace(placeholder, temporal_value)

            # Store as [idx, placeholder_value, filled_string]
            filled_strings.append([idx, temporal_value, filled])

        # Use the template as the key
        dataset[template] = filled_strings

    # Save to JSON
    # Format: {template: [[idx, placeholder_value, filled_string], ...]}
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Dataset generated successfully!")
    print(f"Templates processed: {len(templates)}")
    print(f"Combinations per template: {len(combinations)}")
    print(f"Total strings generated: {len(templates) * len(combinations)}")
    print(f"Output saved to: {output_json_path}")
    print(f"Format: {{template: [[idx, placeholder_value, filled_string], ...]}}")


if __name__ == "__main__":
    # Load config and generate dataset using cfg.filter.regex_file
    cfg = load_config()
    generate_dataset(cfg=cfg)

    # Alternative: Manual mode (backward compatible)
    # categories = {
    #     "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    #     "time": None,  # Skip time
    #     "adverb": None,  # Skip adverb
    # }
    # output_order = ["adverb", "weekday", "time"]
    # template_path = "/home/can/feature_zoo/data/texts/days_templates_123_single_placeholder.txt"
    # output_path = "/home/can/feature_zoo/data/texts/days_trajectories_weekday_only.json"
    # generate_dataset(template_path, output_path, categories, output_order=output_order)
