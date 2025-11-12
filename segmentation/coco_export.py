"""
COCO Export
Functions for exporting annotations in COCO format
"""

import json
import os
import numpy as np
from pycocotools import mask as mask_utils
from segmentation.sam_utils import mask_to_rle


class COCOExporter:
    """COCO format exporter for SAM annotations"""

    def __init__(self, categories: list):
        """
        Initialize COCO exporter

        Args:
            categories: List of category names (e.g., ["body", "rotor", "camera", "other"])
        """
        self.categories = categories
        self.images_json = []
        self.annotations_json = []
        self.categories_json = [
            {"id": i + 1, "name": name} for i, name in enumerate(categories)
        ]
        self.ann_id = 1

    def add_image(
        self,
        image_id: int,
        file_path: str,
        width: int,
        height: int,
        output_dir: str = None,
    ):
        """
        Add image metadata

        Args:
            image_id: Unique image ID
            file_path: Path to image file
            width: Image width
            height: Image height
            output_dir: Output directory (for relative paths, ignored - uses basename)
        """
        # Use just the filename for cleaner COCO format
        file_name = os.path.basename(file_path)

        self.images_json.append(
            {"id": image_id, "file_name": file_name, "width": width, "height": height}
        )

    def add_annotation(self, image_id: int, mask: np.ndarray, category_name: str):
        """
        Add annotation

        Args:
            image_id: Image ID this annotation belongs to
            mask: Boolean mask array
            category_name: Category name (must be in categories list)
        """
        if category_name not in self.categories:
            raise ValueError(f"Category '{category_name}' not in categories list")

        H, W = mask.shape[:2]
        rle = mask_to_rle(mask)
        bbox = mask_utils.toBbox(
            {"size": [H, W], "counts": rle["counts"].encode()}
        ).tolist()

        category_id = self.categories.index(category_name) + 1

        self.annotations_json.append(
            {
                "id": self.ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": rle,
                "area": float(mask.sum()),
                "bbox": bbox,
                "iscrowd": 0,
            }
        )
        self.ann_id += 1

    def export(self, output_path: str):
        """
        Export COCO JSON to file

        Args:
            output_path: Path to output JSON file
        """
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        coco_data = {
            "images": self.images_json,
            "annotations": self.annotations_json,
            "categories": self.categories_json,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2)

        print(f"✅ Saved COCO annotations to {output_path}")
        print(
            f"Images: {len(self.images_json)}  Annotations: {len(self.annotations_json)}"
        )
