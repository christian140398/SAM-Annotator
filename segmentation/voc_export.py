"""
VOC Export
Functions for exporting annotations in VOC (Pascal VOC) XML format
"""

import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


def load_input_objects(xml_path: Optional[str]) -> List[Dict]:
    """
    Load all objects with their bounding boxes from input XML file

    Args:
        xml_path: Path to input XML file, or None

    Returns:
        List of dicts with keys: 'name', 'bbox' (xmin, ymin, xmax, ymax), 'truncated', 'difficult'
    """
    input_objects = []
    if xml_path and os.path.isfile(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall("object"):
                name_elem = obj.find("name")
                name = name_elem.text if name_elem is not None else "unknown"

                bndbox_elem = obj.find("bndbox")
                if bndbox_elem is not None:
                    try:
                        xmin = int(float(bndbox_elem.find("xmin").text))
                        ymin = int(float(bndbox_elem.find("ymin").text))
                        xmax = int(float(bndbox_elem.find("xmax").text))
                        ymax = int(float(bndbox_elem.find("ymax").text))

                        truncated_elem = obj.find("truncated")
                        truncated = (
                            truncated_elem.text if truncated_elem is not None else "0"
                        )

                        difficult_elem = obj.find("difficult")
                        difficult = (
                            difficult_elem.text if difficult_elem is not None else "0"
                        )

                        input_objects.append(
                            {
                                "name": name,
                                "bbox": (xmin, ymin, xmax, ymax),
                                "truncated": truncated,
                                "difficult": difficult,
                            }
                        )
                    except (ValueError, AttributeError):
                        continue
        except Exception as e:
            print(f"Warning: Could not load input XML objects: {e}")

    return input_objects


class VOCExporter:
    """VOC format exporter for SAM annotations"""

    def __init__(self):
        """Initialize VOC exporter"""
        pass

    def export(
        self,
        xml_path: str,
        image_path: str,
        segments: List[Tuple[np.ndarray, str]],
        labels: List[Dict],
        input_xml_path: Optional[str] = None,
        image_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Export annotations to VOC XML format

        Args:
            xml_path: Path to save XML file
            image_path: Path to image file
            segments: List of (mask, label_id) tuples
            labels: List of label dicts with 'id' and 'name' keys
            input_xml_path: Optional path to input XML file to copy original objects from
            image_shape: Optional (height, width) tuple. If None, will load from image_path
        """
        # Load image to get dimensions
        h, w = None, None
        if image_shape is not None:
            h, w = image_shape
        else:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")
            h, w = img.shape[:2]

        # Load input objects if available
        input_objects = load_input_objects(input_xml_path)

        # Create XML root
        root = ET.Element("annotation")

        # Folder
        folder_elem = ET.SubElement(root, "folder")
        folder_elem.text = "images"

        # Filename
        filename_elem = ET.SubElement(root, "filename")
        filename_elem.text = os.path.basename(image_path)

        # Path
        path_elem = ET.SubElement(root, "path")
        path_elem.text = image_path

        # Source
        source_elem = ET.SubElement(root, "source")
        database_elem = ET.SubElement(source_elem, "database")
        database_elem.text = "SAM Annotator"

        # Size
        size_elem = ET.SubElement(root, "size")
        width_elem = ET.SubElement(size_elem, "width")
        width_elem.text = str(w)
        height_elem = ET.SubElement(size_elem, "height")
        height_elem.text = str(h)
        depth_elem = ET.SubElement(size_elem, "depth")
        depth_elem.text = "3"

        # Segmented
        segmented_elem = ET.SubElement(root, "segmented")
        segmented_elem.text = "1" if segments else "0"

        # First, copy all original objects from input XML file (with their original bounding boxes)
        if input_objects:
            print(f"Copying {len(input_objects)} original object(s) from input XML...")
            for inp_obj in input_objects:
                # Create object element for original object
                obj_elem = ET.SubElement(root, "object")

                name_elem = ET.SubElement(obj_elem, "name")
                name_elem.text = inp_obj["name"]

                pose_elem = ET.SubElement(obj_elem, "pose")
                pose_elem.text = "Unspecified"

                truncated_elem = ET.SubElement(obj_elem, "truncated")
                truncated_elem.text = inp_obj.get("truncated", "0")

                difficult_elem = ET.SubElement(obj_elem, "difficult")
                difficult_elem.text = inp_obj.get("difficult", "0")

                # Original bounding box
                xmin, ymin, xmax, ymax = inp_obj["bbox"]
                bndbox_elem = ET.SubElement(obj_elem, "bndbox")
                xmin_elem = ET.SubElement(bndbox_elem, "xmin")
                xmin_elem.text = str(max(0, min(w - 1, xmin)))
                ymin_elem = ET.SubElement(bndbox_elem, "ymin")
                ymin_elem.text = str(max(0, min(h - 1, ymin)))
                xmax_elem = ET.SubElement(bndbox_elem, "xmax")
                xmax_elem.text = str(max(0, min(w - 1, xmax)))
                ymax_elem = ET.SubElement(bndbox_elem, "ymax")
                ymax_elem.text = str(max(0, min(h - 1, ymax)))

        # Now add objects for each segment (with polygon segmentations)
        for mask, label_id in segments:
            # Get label name
            label_name = "UAV"  # Default
            for label in labels:
                if label["id"] == label_id:
                    label_name = label["name"]
                    break

            # Calculate bounding box directly from mask (faster than RLE conversion)
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if rows.any() and cols.any():
                seg_ymin, seg_ymax = np.where(rows)[0][[0, -1]]
                seg_xmin, seg_xmax = np.where(cols)[0][[0, -1]]
            else:
                # Fallback if mask is empty
                seg_xmin = seg_ymin = seg_xmax = seg_ymax = 0

            # Use calculated bounding box from mask for segmented objects
            xmin = seg_xmin
            ymin = seg_ymin
            xmax = seg_xmax
            ymax = seg_ymax
            truncated = "0"
            difficult = "0"

            # Create object element
            obj_elem = ET.SubElement(root, "object")

            name_elem = ET.SubElement(obj_elem, "name")
            name_elem.text = label_name

            pose_elem = ET.SubElement(obj_elem, "pose")
            pose_elem.text = "Unspecified"

            truncated_elem = ET.SubElement(obj_elem, "truncated")
            truncated_elem.text = truncated

            difficult_elem = ET.SubElement(obj_elem, "difficult")
            difficult_elem.text = difficult

            # Bounding box
            bndbox_elem = ET.SubElement(obj_elem, "bndbox")
            xmin_elem = ET.SubElement(bndbox_elem, "xmin")
            xmin_elem.text = str(max(0, min(w - 1, xmin)))
            ymin_elem = ET.SubElement(bndbox_elem, "ymin")
            ymin_elem.text = str(max(0, min(h - 1, ymin)))
            xmax_elem = ET.SubElement(bndbox_elem, "xmax")
            xmax_elem.text = str(max(0, min(w - 1, xmax)))
            ymax_elem = ET.SubElement(bndbox_elem, "ymax")
            ymax_elem.text = str(max(0, min(h - 1, ymax)))

            # Add segmentation polygon
            # Convert mask to polygon coordinates
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) > 0:
                # Create segmentation element (VOC format can include polygon segmentation)
                segmentation_elem = ET.SubElement(obj_elem, "segmentation")

                # Sort contours by area (largest first) and use only the largest one
                # This ensures we only save one polygon per segment, avoiding small artifacts
                contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
                main_contour = contours_sorted[0]

                # Only create polygon if contour has at least 3 points (minimum for a polygon)
                if len(main_contour) >= 3:
                    # Simplify contour if too many points (reduce to reasonable number)
                    # Use slightly higher epsilon for faster processing with minimal quality loss
                    epsilon = 0.002 * cv2.arcLength(main_contour, True)
                    approx = cv2.approxPolyDP(main_contour, epsilon, True)

                    # Only save if simplified polygon has at least 3 points
                    if len(approx) >= 3:
                        # Create polygon element
                        polygon_elem = ET.SubElement(segmentation_elem, "polygon")

                        # Add points as x1,y1 x2,y2 ... format
                        points = []
                        for point in approx:
                            x, y = point[0]
                            points.append(f"{x},{y}")

                        polygon_elem.text = " ".join(points)

        # Create output directory if needed
        os.makedirs(os.path.dirname(xml_path), exist_ok=True)

        # Write XML file with proper formatting
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

        print(f"✅ Saved VOC annotations to {xml_path}")
