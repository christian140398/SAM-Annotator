#!/usr/bin/env python3
"""
Script to clear all files from output/images and output/labels folders,
preserving .gitkeep files.
"""

from pathlib import Path


def clear_output_folders():
    """Clear all files from output folders except .gitkeep files."""
    # Get the project root directory (parent of test/)
    project_root = Path(__file__).parent.parent
    output_images = project_root / "output" / "images"
    output_labels = project_root / "output" / "labels"
    
    import shutil
    
    # Count items to be deleted
    images_count = 0
    if output_images.exists():
        for item in output_images.iterdir():
            if item.name != ".gitkeep":
                images_count += 1
    
    labels_count = 0
    if output_labels.exists():
        for item in output_labels.iterdir():
            if item.name != ".gitkeep":
                labels_count += 1
    
    # Ask for confirmation
    if images_count == 0 and labels_count == 0:
        print("No files to clear.")
        return
    
    total_items = images_count + labels_count
    response = input(f"This will delete {total_items} items ({images_count} images, {labels_count} labels). Continue? (y/n): ").strip().lower()
    
    if response != 'y' and response != 'yes':
        print("Cancelled.")
        return
    
    # Clear images folder
    images_deleted = 0
    if output_images.exists():
        for item in output_images.iterdir():
            if item.name != ".gitkeep":
                try:
                    if item.is_file():
                        item.unlink()
                        images_deleted += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        images_deleted += 1
                except Exception as e:
                    pass
    
    # Clear labels folder
    labels_deleted = 0
    if output_labels.exists():
        for item in output_labels.iterdir():
            if item.name != ".gitkeep":
                try:
                    if item.is_file():
                        item.unlink()
                        labels_deleted += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        labels_deleted += 1
                except Exception as e:
                    pass
    
    print(f"cleared {images_deleted} images")
    print(f"cleared {labels_deleted} labels")


if __name__ == "__main__":
    clear_output_folders()

