#!/usr/bin/env python3
"""
Script to clear all files from output/images, output/segment_labels, and output/bb_labels folders,
preserving .gitkeep files.
"""

from pathlib import Path


def clear_output_folders():
    """Clear all files from output folders except .gitkeep files."""
    # Get the project root directory (parent of test/)
    project_root = Path(__file__).parent.parent
    output_images = project_root / "output" / "images"
    output_segment_labels = project_root / "output" / "segment_labels"
    output_bb_labels = project_root / "output" / "bb_labels"
    
    import shutil
    
    # Count items to be deleted
    images_count = 0
    if output_images.exists():
        for item in output_images.iterdir():
            if item.name != ".gitkeep":
                images_count += 1
    
    segment_labels_count = 0
    if output_segment_labels.exists():
        for item in output_segment_labels.iterdir():
            if item.name != ".gitkeep":
                segment_labels_count += 1
    
    bb_labels_count = 0
    if output_bb_labels.exists():
        for item in output_bb_labels.iterdir():
            if item.name != ".gitkeep":
                bb_labels_count += 1
    
    # Ask for confirmation
    if images_count == 0 and segment_labels_count == 0 and bb_labels_count == 0:
        print("No files to clear.")
        return
    
    total_items = images_count + segment_labels_count + bb_labels_count
    response = input(f"This will delete {total_items} items ({images_count} images, {segment_labels_count} segment labels, {bb_labels_count} bb labels). Continue? (y/n): ").strip().lower()
    
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
    
    # Clear segment labels folder
    segment_labels_deleted = 0
    if output_segment_labels.exists():
        for item in output_segment_labels.iterdir():
            if item.name != ".gitkeep":
                try:
                    if item.is_file():
                        item.unlink()
                        segment_labels_deleted += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        segment_labels_deleted += 1
                except Exception as e:
                    pass
    
    # Clear bb labels folder
    bb_labels_deleted = 0
    if output_bb_labels.exists():
        for item in output_bb_labels.iterdir():
            if item.name != ".gitkeep":
                try:
                    if item.is_file():
                        item.unlink()
                        bb_labels_deleted += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        bb_labels_deleted += 1
                except Exception as e:
                    pass
    
    print(f"cleared {images_deleted} images")
    print(f"cleared {segment_labels_deleted} segment labels")
    print(f"cleared {bb_labels_deleted} bb labels")


if __name__ == "__main__":
    clear_output_folders()

