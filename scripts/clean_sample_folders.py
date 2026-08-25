#!/usr/bin/env python3
"""
Script to delete generated_samples and valid_samples folders
from savedmodels/experimental directory.
"""

import shutil
from pathlib import Path


def main():
    base_dir = Path(__file__).parent.parent / "savedmodels" / "experimental"

    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return

    folders_to_delete = ["generated_samples", "valid_samples"]
    deleted_count = 0

    for folder_name in folders_to_delete:
        for folder in base_dir.rglob(folder_name):
            if folder.is_dir():
                print(f"Deleting: {folder}")
                shutil.rmtree(folder)
                deleted_count += 1

    print(f"\nDeleted {deleted_count} folder(s)")


if __name__ == "__main__":
    main()
