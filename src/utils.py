"""
utils.py
--------
Small, reusable helper functions shared across the pipeline.

Keeping these in one place avoids duplicating logic (e.g. printing
progress banners, making sure output folders exist) inside every
other module.
"""

from pathlib import Path


def print_step(title: str) -> None:
    """Print a formatted banner so pipeline progress is easy to follow in the console.

    Args:
        title: Short description of the step currently running.
    """
    print("\n" + "=" * 60)
    print(f"STEP: {title}")
    print("=" * 60)


def ensure_directory_exists(path: Path) -> None:
    """Create a directory (and any missing parent folders) if it doesn't exist.

    Args:
        path: Directory path that must exist before writing files into it.
    """
    path.mkdir(parents=True, exist_ok=True)
