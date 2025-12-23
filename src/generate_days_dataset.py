import csv
import json
from itertools import product
from pathlib import Path


def generate_dataset(
    template_csv_path: str,
    output_json_path: str,
    categories: dict[str, list[str] | None],
    placeholder: str = "{temporal}",
    output_order: list[str] | None = None,
) -> None:
    """
    Generate a dataset from template CSV with placeholder substitution.

    This version uses a single placeholder in the template and concatenates
    multiple category values to fill it.

    Args:
        template_csv_path: Path to the template CSV file
        output_json_path: Path for the output JSON file
        categories: Ordered dict of category names to lists of values
                   The order determines hierarchy (last changes fastest)
                   (e.g., {'weekday': [...], 'time': [...], 'adverb': [...]})
                   Use None to skip a category entirely.
        placeholder: The placeholder string in templates (default: "{temporal}")
        output_order: Optional list specifying the order of categories in the output string
                     (e.g., ['adverb', 'weekday', 'time'] for "early Monday morning")
                     If None, uses the order from categories dict.
    """
    # Read templates from CSV
    templates = []
    with open(template_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            templates.append(row["template"])

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

            # Clean up any multiple spaces
            filled = " ".join(filled.split())

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
    # Define the ordered categories with their possible values
    # Set to None to skip a category
    # NOTE: Order in dict determines hierarchy (last changes fastest)
    categories = {
        "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "time": None,  # Skip time
        "adverb": None,  # Skip adverb
    }

    # Alternative: use all categories (adverb changes fastest in hierarchy)
    # categories = {
    #     "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    #     "time": ["morning", "afternoon", "evening"],
    #     "adverb": ["early", "late"],
    # }

    # Define output order (how values appear in the string)
    # For "early Monday morning", use: ['adverb', 'weekday', 'time']
    output_order = ["adverb", "weekday", "time"]

    # Set paths
    template_path = "/home/can/feature_zoo/data/texts/days_templates_100_single_placeholder.csv"
    output_path = "/home/can/feature_zoo/data/texts/days_dataset_weekday_only.json"

    # Generate the dataset
    generate_dataset(template_path, output_path, categories, output_order=output_order)
