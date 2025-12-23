import pytest
import tempfile
import json
import csv
from pathlib import Path

from src.generate_days_dataset import generate_dataset as generate_dataset_v1
from src.generate_days_dataset_v2 import generate_dataset as generate_dataset_v2


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_template_csv_multi_placeholder(temp_dir):
    """Create a sample CSV with multiple placeholders for v1 tests."""
    csv_path = temp_dir / "templates_multi.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["template"])
        writer.writeheader()
        writer.writerow({"template": "Event happened on {adverb} {weekday} {time}"})
        writer.writerow({"template": "Meeting scheduled for {adverb} {weekday} {time}"})
    return csv_path


@pytest.fixture
def sample_template_csv_single_placeholder(temp_dir):
    """Create a sample CSV with single placeholder for v2 tests."""
    csv_path = temp_dir / "templates_single.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["template"])
        writer.writeheader()
        writer.writerow({"template": "Event happened on {temporal}"})
        writer.writerow({"template": "Meeting scheduled for {temporal}"})
    return csv_path


class TestGenerateDatasetV1:
    """Tests for the original multi-placeholder version."""

    def test_all_categories(self, sample_template_csv_multi_placeholder, temp_dir):
        """Test generation with all categories active."""
        output_path = temp_dir / "output_v1_all.json"
        categories = {
            "weekday": ["Monday", "Tuesday"],
            "time": ["morning", "afternoon"],
            "adverb": ["early", "late"],
        }

        generate_dataset_v1(
            str(sample_template_csv_multi_placeholder), str(output_path), categories
        )

        # Load and verify output
        with open(output_path, "r") as f:
            data = json.load(f)

        # Should have 2 templates
        assert len(data) == 2

        # Each template should have 2*2*2 = 8 combinations
        for template, filled_strings in data.items():
            assert len(filled_strings) == 8

        # Verify ordering: adverb (last) changes fastest
        first_template = list(data.keys())[0]
        values = data[first_template]
        assert "early Monday morning" in values[0]
        assert "late Monday morning" in values[1]
        assert "early Monday afternoon" in values[2]
        assert "late Monday afternoon" in values[3]

    def test_skip_categories_with_none(
        self, sample_template_csv_multi_placeholder, temp_dir
    ):
        """Test skipping categories with None value."""
        output_path = temp_dir / "output_v1_skip.json"
        categories = {
            "weekday": ["Monday", "Tuesday"],
            "time": None,
            "adverb": None,
        }

        generate_dataset_v1(
            str(sample_template_csv_multi_placeholder), str(output_path), categories
        )

        # Load and verify output
        with open(output_path, "r") as f:
            data = json.load(f)

        # Each template should have only 2 combinations (weekdays only)
        first_template = list(data.keys())[0]
        values = data[first_template]
        assert len(values) == 2

        # Verify no extra spaces and placeholders are removed
        assert "Event happened on Monday" in values[0]
        assert "Event happened on Tuesday" in values[1]
        assert "{time}" not in values[0]
        assert "{adverb}" not in values[0]
        # Check no double spaces
        assert "  " not in values[0]

    def test_empty_categories_edge_case(
        self, sample_template_csv_multi_placeholder, temp_dir
    ):
        """Test edge case with all categories set to None."""
        output_path = temp_dir / "output_v1_empty.json"
        categories = {
            "weekday": None,
            "time": None,
            "adverb": None,
        }

        generate_dataset_v1(
            str(sample_template_csv_multi_placeholder), str(output_path), categories
        )

        # Load and verify output
        with open(output_path, "r") as f:
            data = json.load(f)

        # Should have 1 combination per template (empty)
        first_template = list(data.keys())[0]
        values = data[first_template]
        assert len(values) == 1


class TestGenerateDatasetV2:
    """Tests for the single-placeholder version with output ordering."""

    def test_all_categories_with_output_order(
        self, sample_template_csv_single_placeholder, temp_dir
    ):
        """Test generation with all categories and custom output order."""
        output_path = temp_dir / "output_v2_all.json"
        categories = {
            "weekday": ["Monday", "Tuesday"],
            "time": ["morning", "afternoon"],
            "adverb": ["early", "late"],
        }
        output_order = ["adverb", "weekday", "time"]

        generate_dataset_v2(
            str(sample_template_csv_single_placeholder),
            str(output_path),
            categories,
            output_order=output_order,
        )

        # Load and verify output
        with open(output_path, "r") as f:
            data = json.load(f)

        # Should have 2 templates
        assert len(data) == 2

        # Each template should have 2*2*2 = 8 combinations
        first_template = list(data.keys())[0]
        entries = data[first_template]
        assert len(entries) == 8

        # Verify format: [idx, placeholder_value, filled_string]
        first_entry = entries[0]
        assert len(first_entry) == 3
        assert isinstance(first_entry[0], int)  # idx
        assert isinstance(first_entry[1], str)  # placeholder_value
        assert isinstance(first_entry[2], str)  # filled_string

        # Verify ordering: adverb (last in categories) changes fastest
        assert entries[0][1] == "early Monday morning"
        assert entries[1][1] == "late Monday morning"
        assert entries[2][1] == "early Monday afternoon"
        assert entries[3][1] == "late Monday afternoon"

        # Verify filled strings are correct
        assert "Event happened on early Monday morning" in entries[0][2]
        assert "Event happened on late Monday morning" in entries[1][2]

    def test_weekday_only(self, sample_template_csv_single_placeholder, temp_dir):
        """Test generation with only weekday category."""
        output_path = temp_dir / "output_v2_weekday.json"
        categories = {
            "weekday": ["Monday", "Tuesday", "Wednesday"],
            "time": None,
            "adverb": None,
        }
        output_order = ["adverb", "weekday", "time"]

        generate_dataset_v2(
            str(sample_template_csv_single_placeholder),
            str(output_path),
            categories,
            output_order=output_order,
        )

        # Load and verify output
        with open(output_path, "r") as f:
            data = json.load(f)

        first_template = list(data.keys())[0]
        entries = data[first_template]

        # Should have 3 combinations
        assert len(entries) == 3

        # Verify format and content
        assert entries[0] == [0, "Monday", "Event happened on Monday"]
        assert entries[1] == [1, "Tuesday", "Event happened on Tuesday"]
        assert entries[2] == [2, "Wednesday", "Event happened on Wednesday"]

        # No extra spaces
        for entry in entries:
            assert "  " not in entry[2]

    def test_output_order_default(
        self, sample_template_csv_single_placeholder, temp_dir
    ):
        """Test that output_order defaults to categories dict order."""
        output_path = temp_dir / "output_v2_default.json"
        categories = {
            "weekday": ["Monday"],
            "time": ["morning"],
            "adverb": ["early"],
        }

        # No output_order specified
        generate_dataset_v2(
            str(sample_template_csv_single_placeholder),
            str(output_path),
            categories,
            output_order=None,
        )

        # Load and verify output
        with open(output_path, "r") as f:
            data = json.load(f)

        first_template = list(data.keys())[0]
        entries = data[first_template]

        # Should use dict order: weekday, time, adverb
        assert entries[0][1] == "Monday morning early"

    def test_hierarchical_ordering(
        self, sample_template_csv_single_placeholder, temp_dir
    ):
        """Test that hierarchical ordering works correctly (last category changes fastest)."""
        output_path = temp_dir / "output_v2_hierarchy.json"
        categories = {
            "weekday": ["Monday", "Tuesday"],
            "time": ["morning", "afternoon"],
            "adverb": ["early", "late"],
        }
        output_order = ["adverb", "weekday", "time"]

        generate_dataset_v2(
            str(sample_template_csv_single_placeholder),
            str(output_path),
            categories,
            output_order=output_order,
        )

        with open(output_path, "r") as f:
            data = json.load(f)

        first_template = list(data.keys())[0]
        entries = data[first_template]

        # Verify indexes
        for i, entry in enumerate(entries):
            assert entry[0] == i

        # Verify last category (adverb) changes fastest
        assert entries[0][1] == "early Monday morning"
        assert entries[1][1] == "late Monday morning"  # adverb changed

        # Verify second-to-last (time) changes next
        assert entries[2][1] == "early Monday afternoon"  # time changed
        assert entries[3][1] == "late Monday afternoon"  # adverb changed again

        # Verify first category (weekday) changes slowest
        assert entries[4][1] == "early Tuesday morning"  # weekday changed

    def test_json_format_structure(
        self, sample_template_csv_single_placeholder, temp_dir
    ):
        """Test that JSON structure is exactly as specified."""
        output_path = temp_dir / "output_v2_format.json"
        categories = {
            "weekday": ["Monday"],
            "time": ["morning"],
            "adverb": ["early"],
        }
        output_order = ["adverb", "weekday", "time"]

        generate_dataset_v2(
            str(sample_template_csv_single_placeholder),
            str(output_path),
            categories,
            output_order=output_order,
        )

        with open(output_path, "r") as f:
            data = json.load(f)

        # Verify top-level structure: dict with template strings as keys
        assert isinstance(data, dict)

        # Verify each value is a list of [idx, value, filled_string]
        for template, entries in data.items():
            assert isinstance(entries, list)
            for entry in entries:
                assert isinstance(entry, list)
                assert len(entry) == 3
                assert isinstance(entry[0], int)  # idx
                assert isinstance(entry[1], str)  # placeholder_value
                assert isinstance(entry[2], str)  # filled_string
                # Verify filled_string contains the placeholder_value
                assert entry[1] in entry[2]


class TestEdgeCases:
    """Test edge cases for both versions."""

    def test_special_characters_in_template(self, temp_dir):
        """Test templates with special characters like quotes."""
        csv_path = temp_dir / "special_chars.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["template"])
            writer.writeheader()
            writer.writerow(
                {"template": 'He said "it happened on {temporal}" yesterday'}
            )

        output_path = temp_dir / "output_special.json"
        categories = {"weekday": ["Monday"]}

        generate_dataset_v2(
            str(csv_path), str(output_path), categories, output_order=None
        )

        with open(output_path, "r") as f:
            data = json.load(f)

        first_template = list(data.keys())[0]
        entries = data[first_template]
        assert 'He said "it happened on Monday" yesterday' == entries[0][2]

    def test_multiple_spaces_cleanup(self, temp_dir):
        """Test that multiple spaces are properly cleaned up."""
        csv_path = temp_dir / "spaces.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["template"])
            writer.writeheader()
            writer.writerow({"template": "Event  on   {temporal}  today"})

        output_path = temp_dir / "output_spaces.json"
        categories = {"weekday": ["Monday"]}

        generate_dataset_v2(
            str(csv_path), str(output_path), categories, output_order=None
        )

        with open(output_path, "r") as f:
            data = json.load(f)

        first_template = list(data.keys())[0]
        entries = data[first_template]
        # Multiple spaces should be collapsed to single spaces
        assert "  " not in entries[0][2]
        assert "Event on Monday today" == entries[0][2]
