#!/usr/bin/env python3
"""
Utility script for managing cached artifacts.

Usage:
    python scripts/manage_artifacts.py list             # List all artifacts with metadata
    python scripts/manage_artifacts.py orphans          # List artifacts without metadata
    python scripts/manage_artifacts.py clean --dry-run  # Preview what would be deleted
    python scripts/manage_artifacts.py clean            # Delete orphaned artifacts
    python scripts/manage_artifacts.py info <file>      # Show metadata for specific artifact
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
from datetime import datetime


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def find_all_artifacts(project_root: Path) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Find all artifact files in the project.

    Returns:
        (token_artifacts, activation_artifacts, metadata_files)
    """
    tokens_dir = project_root / "data" / "tokens"
    activations_dir = project_root / "data" / "activations"

    token_artifacts = []
    activation_artifacts = []
    metadata_files = []

    if tokens_dir.exists():
        token_artifacts = list(tokens_dir.glob("*.safetensors"))
        metadata_files.extend(tokens_dir.glob("*.json"))

    if activations_dir.exists():
        activation_artifacts = list(activations_dir.glob("*.safetensors"))
        metadata_files.extend(activations_dir.glob("*.json"))

    return token_artifacts, activation_artifacts, metadata_files


def find_orphaned_artifacts(project_root: Path) -> List[Path]:
    """Find artifacts that don't have corresponding metadata files."""
    token_artifacts, activation_artifacts, metadata_files = find_all_artifacts(project_root)

    all_artifacts = token_artifacts + activation_artifacts
    metadata_stems = {f.stem for f in metadata_files}

    orphaned = []
    for artifact in all_artifacts:
        if artifact.stem not in metadata_stems:
            orphaned.append(artifact)

    return orphaned


def list_artifacts(project_root: Path) -> None:
    """List all artifacts with their metadata."""
    token_artifacts, activation_artifacts, metadata_files = find_all_artifacts(project_root)

    print(f"\n{'='*80}")
    print(f"ARTIFACT SUMMARY")
    print(f"{'='*80}\n")

    print(f"Token artifacts:      {len(token_artifacts)}")
    print(f"Activation artifacts: {len(activation_artifacts)}")
    print(f"Metadata files:       {len(metadata_files)}")
    print(f"\nTotal artifacts:      {len(token_artifacts) + len(activation_artifacts)}")

    # Group by config hash
    metadata_by_hash = {}
    for metadata_path in metadata_files:
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                config_hash = metadata.get('config_hash', 'unknown')
                if config_hash not in metadata_by_hash:
                    metadata_by_hash[config_hash] = []
                metadata_by_hash[config_hash].append((metadata_path, metadata))
        except Exception as e:
            print(f"Warning: Could not read {metadata_path}: {e}")

    print(f"\n{'='*80}")
    print(f"ARTIFACTS BY CONFIG HASH")
    print(f"{'='*80}\n")

    for config_hash, items in sorted(metadata_by_hash.items()):
        print(f"\nConfig Hash: {config_hash}")
        print(f"{'-'*80}")
        for metadata_path, metadata in items:
            artifact_path = metadata_path.with_suffix('.safetensors')
            artifact_exists = artifact_path.exists()
            status = "✓" if artifact_exists else "✗ (missing)"

            print(f"  {status} {artifact_path.name}")
            print(f"      Type: {metadata.get('artifact_type', 'unknown')}")
            print(f"      Created: {metadata.get('created_at', 'unknown')}")

            # Show config snapshot summary
            config_snapshot = metadata.get('config_snapshot', {})
            if 'data' in config_snapshot:
                print(f"      Data: {config_snapshot['data'].get('name', 'unknown')}")
            if 'llm' in config_snapshot:
                print(f"      LLM: {config_snapshot['llm'].get('hf_name', 'unknown')} "
                      f"layer {config_snapshot['llm'].get('layer_idx', '?')}")

    # List orphaned artifacts
    orphaned = find_orphaned_artifacts(project_root)
    if orphaned:
        print(f"\n{'='*80}")
        print(f"ORPHANED ARTIFACTS (no metadata)")
        print(f"{'='*80}\n")
        for artifact in sorted(orphaned):
            size_mb = artifact.stat().st_size / (1024 * 1024)
            print(f"  {artifact.relative_to(project_root)} ({size_mb:.2f} MB)")


def list_orphans(project_root: Path) -> None:
    """List only orphaned artifacts."""
    orphaned = find_orphaned_artifacts(project_root)

    if not orphaned:
        print("\n✓ No orphaned artifacts found!")
        return

    total_size = sum(f.stat().st_size for f in orphaned)
    total_size_mb = total_size / (1024 * 1024)

    print(f"\n{'='*80}")
    print(f"ORPHANED ARTIFACTS ({len(orphaned)} files, {total_size_mb:.2f} MB total)")
    print(f"{'='*80}\n")

    for artifact in sorted(orphaned):
        size_mb = artifact.stat().st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(artifact.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {artifact.relative_to(project_root)}")
        print(f"    Size: {size_mb:.2f} MB | Modified: {modified}")


def clean_orphans(project_root: Path, dry_run: bool = True) -> None:
    """Delete orphaned artifacts."""
    orphaned = find_orphaned_artifacts(project_root)

    if not orphaned:
        print("\n✓ No orphaned artifacts to clean!")
        return

    total_size = sum(f.stat().st_size for f in orphaned)
    total_size_mb = total_size / (1024 * 1024)

    if dry_run:
        print(f"\n{'='*80}")
        print(f"DRY RUN - Would delete {len(orphaned)} files ({total_size_mb:.2f} MB)")
        print(f"{'='*80}\n")
        for artifact in sorted(orphaned):
            print(f"  Would delete: {artifact.relative_to(project_root)}")
        print(f"\nRun without --dry-run to actually delete these files.")
    else:
        print(f"\n{'='*80}")
        print(f"DELETING {len(orphaned)} orphaned artifacts ({total_size_mb:.2f} MB)")
        print(f"{'='*80}\n")

        deleted = 0
        failed = 0
        for artifact in sorted(orphaned):
            try:
                artifact.unlink()
                print(f"  ✓ Deleted: {artifact.relative_to(project_root)}")
                deleted += 1
            except Exception as e:
                print(f"  ✗ Failed to delete {artifact.relative_to(project_root)}: {e}")
                failed += 1

        print(f"\n{'='*80}")
        print(f"Summary: {deleted} deleted, {failed} failed")
        print(f"{'='*80}")


def show_artifact_info(project_root: Path, artifact_name: str) -> None:
    """Show detailed information about a specific artifact."""
    # Try to find the artifact
    tokens_dir = project_root / "data" / "tokens"
    activations_dir = project_root / "data" / "activations"

    artifact_path = None
    for directory in [tokens_dir, activations_dir]:
        potential_path = directory / artifact_name
        if potential_path.exists():
            artifact_path = potential_path
            break

    if artifact_path is None:
        print(f"\n✗ Artifact not found: {artifact_name}")
        return

    # Load metadata
    metadata_path = artifact_path.with_suffix('.json')
    if not metadata_path.exists():
        print(f"\n✗ No metadata found for {artifact_name}")
        print(f"  Artifact path: {artifact_path}")
        print(f"  This is an orphaned artifact.")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Display information
    size_mb = artifact_path.stat().st_size / (1024 * 1024)
    modified = datetime.fromtimestamp(artifact_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')

    print(f"\n{'='*80}")
    print(f"ARTIFACT INFORMATION")
    print(f"{'='*80}\n")

    print(f"File:         {artifact_path.name}")
    print(f"Path:         {artifact_path}")
    print(f"Size:         {size_mb:.2f} MB")
    print(f"Modified:     {modified}")
    print(f"\nMetadata:")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Manage cached artifacts in the feature_zoo project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/manage_artifacts.py list             # List all artifacts
  python scripts/manage_artifacts.py orphans          # List orphaned artifacts
  python scripts/manage_artifacts.py clean --dry-run  # Preview cleanup
  python scripts/manage_artifacts.py clean            # Actually clean orphans
  python scripts/manage_artifacts.py info my_artifact.safetensors
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # List command
    subparsers.add_parser('list', help='List all artifacts with metadata')

    # Orphans command
    subparsers.add_parser('orphans', help='List orphaned artifacts (no metadata)')

    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Delete orphaned artifacts')
    clean_parser.add_argument('--dry-run', action='store_true',
                             help='Preview what would be deleted without actually deleting')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show info about specific artifact')
    info_parser.add_argument('artifact', help='Artifact filename')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    project_root = get_project_root()

    if args.command == 'list':
        list_artifacts(project_root)
    elif args.command == 'orphans':
        list_orphans(project_root)
    elif args.command == 'clean':
        clean_orphans(project_root, dry_run=args.dry_run)
    elif args.command == 'info':
        show_artifact_info(project_root, args.artifact)


if __name__ == '__main__':
    main()
