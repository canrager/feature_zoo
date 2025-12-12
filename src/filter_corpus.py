"""
Filter corpus for occurences of tokens
"""

import re
import csv
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm

from src.config import Config
from src.loading import load_texts
from datasets import load_dataset
from collections import defaultdict


def filter_corpus(cfg: Config, keywords_labels: Optional[List[str]] = None):
    corpus = load_dataset(cfg.filter.corpus, split="train", streaming=True)
    # Load keywords from the regex file
    labels, keywords_list = load_texts(cfg, cfg.filter.regex_file)
    # Use provided labels or fall back to loaded labels
    if keywords_labels is None:
        keywords_labels = labels

    # Map keywords to their labels BEFORE converting to set (to preserve association)
    keyword_to_label = {}
    for i, keyword in enumerate(keywords_list):
        if keywords_labels and i < len(keywords_labels):
            keyword_to_label[keyword] = keywords_labels[i]
        else:
            keyword_to_label[keyword] = keyword  # Fallback to keyword itself

    keywords = set(keywords_list)
    initial_keyword_count = len(keywords)
    num_total_required = initial_keyword_count * cfg.filter.num_occurences

    matches = defaultdict(list)

    # Create progress bar
    pbar = tqdm(
        desc="Filtering corpus",
        unit="docs",
        dynamic_ncols=True,
    )

    for doc in corpus:
        doc_text = doc.get("text", "") if isinstance(doc, dict) else str(doc)

        for key in list(keywords):  # Use list() to avoid modification during iteration
            if key in matches and len(matches[key]) >= cfg.filter.num_occurences:
                # Keyword has reached target count
                if cfg.env.debug:
                    print(
                        f"\n[DEBUG] Keyword '{key}' has collected all {cfg.filter.num_occurences} occurrences"
                    )
                    print(f"[DEBUG] Matched sequences for '{key}':")
                    for i, matched_data in enumerate(matches[key], 1):
                        matched_text, matched_label = matched_data
                        print(f"  {i}. {matched_text} (label: {matched_label})")
                keywords.remove(key)

            # Escape special regex characters in the keyword
            escaped_key = re.escape(key)
            # Create regex pattern: match all preceding text and first occurrence of
            # "[punctuation/space]keyword[punctuation/space]"
            # Pattern matches: (all preceding text)(punctuation/space or start)(keyword)(punctuation/space or end)
            pattern = rf"(.*?)((?:^|[\s\W]){escaped_key}(?:[\s\W]|$))"

            match = re.search(pattern, doc_text, re.IGNORECASE)
            if match:
                # Match group 0 is the full match (all preceding text + keyword with punctuation)
                # Match group 1 is the preceding text
                # Match group 2 is the keyword with surrounding punctuation/space
                matched_text = match.group(0)
                # Remove trailing punctuation or space
                matched_text = re.sub(r"[\s\W]+$", "", matched_text)
                # Store both text and label
                label = keyword_to_label.get(key, key)
                # Optionally filter for min number of characters
                if (cfg.filter.min_char_count is None) or (
                    len(matched_text) >= cfg.filter.min_char_count
                ):
                    matches[key].append((matched_text, label))

        # Update progress bar with current status
        completed_keywords = initial_keyword_count - len(keywords)
        total_matches = sum(len(m) for m in matches.values())
        pbar.set_postfix(
            {
                "docs": pbar.n + 1,
                "keywords": f"{completed_keywords}/{initial_keyword_count}",
                "matches": f"{total_matches}/{num_total_required}",
            }
        )
        pbar.update(1)

        # Check termination criterion
        if len(keywords) == 0:
            break

    pbar.close()
    print(
        f"\nFiltering complete: Found {sum(len(m) for m in matches.values())} total matches across {initial_keyword_count} keywords"
    )

    # Order matches by original keywords_list order
    ordered_matches = {key: matches[key] for key in keywords_list if key in matches}
    return ordered_matches


def save_filtered_corpus(
    cfg: Config, force_rerun: bool = False
) -> List[tuple[str, str]]:
    # Determine save name, save dir is in cfg.env.texts_dir
    save_dir = Path(cfg.env.texts_dir)
    save_path = save_dir / f"{cfg.filter.regex_file}_filtered.csv"

    # If file exists, print warning and exit. Else load keywords
    if (force_rerun == False) and (save_path.exists()):
        print(f"Warning: File {save_path} already exists. Exiting without overwriting.")
        return []

    # Load keywords and compute matches
    matches = filter_corpus(cfg)

    # Flatten matches into a single list of all matched texts
    # Flatten matches into a list of (text, label) tuples.
    matched_entries: List[tuple[str, str]] = []
    for key, matched_list in matches.items():
        matched_entries.extend(matched_list)

    # Save results
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "text"])
        for text, label in matched_entries:
            writer.writerow([label, text])

    return matched_entries


if __name__ == "__main__":
    from src.config import load_config

    cfg = load_config()
    save_filtered_corpus(cfg, force_rerun=True)
