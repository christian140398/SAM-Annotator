"""
Test script to visualize polygon segmentations from XML annotation files.
Usage: python test/segmentation_test.py <filename>
Example: python test/segmentation_test.py 00004
"""
import os
import sys
import argparse
import xml.etree.ElementTree as ET
import hashlib
import cv2
import numpy as np

# Optional matplotlib import for better visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Label file path (relative to project root)
LABEL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "label.txt")


def load_labels_from_file(label_file: str):
    """
    Load labels from a text file, one label per line.
    
    Args:
        label_file: Path to the label file
        
    Returns:
        List of label names (stripped of whitespace)
    """
    labels = []
    if os.path.isfile(label_file):
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    label = line.strip()
                    if label:  # Skip empty lines
                        labels.append(label)
        except Exception as e:
            print(f"Warning: Could not load labels from {label_file}: {e}")
    else:
        print(f"Warning: Label file not found: {label_file}")
    
    return labels


def generate_random_color_rgb(seed=None):
    """
    Generate a deterministic color as RGB tuple (0-255) based on seed.
    Colors are consistent across different images and sessions.
    
    Args:
        seed: Seed string for reproducible colors (e.g., label name)
        
    Returns:
        Tuple of (R, G, B) values from 0-255
    """
    if seed is None:
        seed = "default"
    
    # Use deterministic hash (MD5) to generate consistent colors
    hash_obj = hashlib.md5(seed.encode('utf-8'))
    hash_bytes = hash_obj.digest()
    
    # Generate bright, visible colors (avoid too dark colors)
    # Map hash bytes to color range 50-255
    r = 50 + (hash_bytes[0] % 206)  # 206 = 255 - 50 + 1
    g = 50 + (hash_bytes[1] % 206)
    b = 50 + (hash_bytes[2] % 206)
    
    return (r, g, b)


def load_color_palette(label_file: str):
    """
    Load labels from file and generate random colors for each.
    
    Args:
        label_file: Path to the label file
        
    Returns:
        Dict mapping label name (lowercase) to RGB tuple
    """
    labels = load_labels_from_file(label_file)
    color_palette = {}
    
    for label in labels:
        # Use label name as seed for consistent color generation
        color_palette[label.lower()] = generate_random_color_rgb(label)
    
    # Default color for unknown labels (red)
    color_palette["default"] = (255, 0, 0)
    
    return color_palette


def parse_polygon(polygon_str):
    """
    Parse polygon string "x1,y1 x2,y2 x3,y3 ..." into numpy array of points.
    
    Args:
        polygon_str: String containing polygon coordinates
        
    Returns:
        numpy array of shape (N, 2) with x,y coordinates
    """
    if not polygon_str or not polygon_str.strip():
        return None
    
    points = []
    for point_str in polygon_str.strip().split():
        if ',' in point_str:
            x, y = point_str.split(',')
            points.append([float(x), float(y)])
    
    if len(points) < 3:  # Need at least 3 points for a polygon
        return None
    
    return np.array(points, dtype=np.int32)


def load_segmentations_from_xml(xml_path):
    """
    Load all polygon segmentations and bounding boxes from a VOC XML file.
    
    Args:
        xml_path: Path to XML annotation file
        
    Returns:
        List of tuples: (label_name, polygon_points, bbox) where:
            - label_name: string
            - polygon_points: (N, 2) numpy array or None
            - bbox: tuple (xmin, ymin, xmax, ymax) or None
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    segmentations = []
    
    # Find all object elements
    for obj in root.findall("object"):
        # Get label name
        name_elem = obj.find("name")
        label_name = name_elem.text if name_elem is not None else "unknown"
        
        # Get bounding box
        bbox = None
        bndbox_elem = obj.find("bndbox")
        if bndbox_elem is not None:
            try:
                xmin = int(float(bndbox_elem.find("xmin").text))
                ymin = int(float(bndbox_elem.find("ymin").text))
                xmax = int(float(bndbox_elem.find("xmax").text))
                ymax = int(float(bndbox_elem.find("ymax").text))
                bbox = (xmin, ymin, xmax, ymax)
            except (ValueError, AttributeError):
                bbox = None
        
        # Get segmentation polygon
        polygon_points = None
        seg_elem = obj.find("segmentation")
        if seg_elem is not None:
            polygon_elem = seg_elem.find("polygon")
            if polygon_elem is not None and polygon_elem.text:
                polygon_points = parse_polygon(polygon_elem.text)
        
        # Add to segmentations if we have either polygon or bbox
        if polygon_points is not None or bbox is not None:
            segmentations.append((label_name, polygon_points, bbox))
    
    return segmentations


def draw_segmentations_opencv(image, segmentations, original_bboxes=None, output_path=None, color_palette=None):
    """
    Draw polygon segmentations and bounding boxes on image using OpenCV.
    
    Args:
        image: Input image (BGR format)
        segmentations: List of (label_name, polygon_points, bbox) tuples
        original_bboxes: List of original bounding boxes from input XML to draw in different style
        output_path: Optional path to save the result image
        color_palette: Dict mapping label names to RGB tuples
    """
    if color_palette is None:
        color_palette = load_color_palette(LABEL_FILE)
    
    img = image.copy()
    
    # Draw original bounding boxes first (if any) in a distinctive color
    if original_bboxes:
        for orig_bbox in original_bboxes:
            if orig_bbox:
                xmin, ymin, xmax, ymax = orig_bbox
                # Draw original bounding box in white with dashed style
                # Create dashed effect
                dash_length = 15
                gap_length = 10
                # Top edge
                for x in range(xmin, xmax, dash_length + gap_length):
                    end_x = min(x + dash_length, xmax)
                    cv2.line(img, (x, ymin), (end_x, ymin), (255, 255, 255), 3)
                # Bottom edge
                for x in range(xmin, xmax, dash_length + gap_length):
                    end_x = min(x + dash_length, xmax)
                    cv2.line(img, (x, ymax), (end_x, ymax), (255, 255, 255), 3)
                # Left edge
                for y in range(ymin, ymax, dash_length + gap_length):
                    end_y = min(y + dash_length, ymax)
                    cv2.line(img, (xmin, y), (xmin, end_y), (255, 255, 255), 3)
                # Right edge
                for y in range(ymin, ymax, dash_length + gap_length):
                    end_y = min(y + dash_length, ymax)
                    cv2.line(img, (xmax, y), (xmax, end_y), (255, 255, 255), 3)
                # Add label
                cv2.putText(img, "Original BBox", (xmin, ymin - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    for label_name, polygon_points, bbox in segmentations:
        # Get color for this label (RGB format)
        color_rgb = color_palette.get(label_name.lower(), color_palette["default"])
        # Convert RGB to BGR for OpenCV
        color = (color_rgb[2], color_rgb[1], color_rgb[0])
        
        # Draw polygon if available
        if polygon_points is not None:
            # Draw filled polygon with transparency
            overlay = img.copy()
            cv2.fillPoly(overlay, [polygon_points], color)
            cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
            
            # Draw polygon outline
            cv2.polylines(img, [polygon_points], isClosed=True, color=color, thickness=2)
        
        # Draw bounding box if available
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            # Draw bounding box rectangle with thicker line to distinguish from polygon
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)
        
        # Add label text at top-left of bounding box or first polygon point
        if bbox is not None:
            x, y = bbox[0], bbox[1]
        elif polygon_points is not None and len(polygon_points) > 0:
            x, y = polygon_points[0]
        else:
            continue
        
        # Add background for text readability
        (text_width, text_height), _ = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (int(x), int(y) - text_height - 8), 
                     (int(x) + text_width + 4, int(y)), color, -1)
        cv2.putText(img, label_name, (int(x) + 2, int(y) - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display image
    cv2.imshow("Segmentation Test", img)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save if output path specified
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved result to: {output_path}")


def draw_segmentations_matplotlib(image, segmentations, original_bboxes=None, output_path=None, color_palette=None):
    """
    Draw polygon segmentations and bounding boxes on image using Matplotlib.
    
    Args:
        image: Input image (BGR format, will be converted to RGB)
        segmentations: List of (label_name, polygon_points, bbox) tuples
        original_bboxes: List of original bounding boxes from input XML to draw in different style
        output_path: Optional path to save the result image
        color_palette: Dict mapping label names to RGB tuples
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for matplotlib backend. "
            "Install it with: pip install matplotlib\n"
            "Or use --backend opencv instead."
        )
    from matplotlib.patches import Rectangle
    
    if color_palette is None:
        color_palette = load_color_palette(LABEL_FILE)
    
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_rgb)
    ax.set_title("Segmentation Visualization", fontsize=14, fontweight='bold')
    
    # Draw original bounding boxes first (if any) in white with dashed style
    if original_bboxes:
        for i, orig_bbox in enumerate(original_bboxes):
            if orig_bbox:
                xmin, ymin, xmax, ymax = orig_bbox
                width = xmax - xmin
                height = ymax - ymin
                orig_rect = Rectangle((xmin, ymin), width, height,
                                     linewidth=3, edgecolor='white', 
                                     facecolor='none', linestyle='--', alpha=0.9)
                ax.add_patch(orig_rect)
                ax.text(xmin, ymin - 8, "Original BBox", color='white', 
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', 
                                edgecolor='white', alpha=0.7, linewidth=1))
    
    for label_name, polygon_points, bbox in segmentations:
        # Get color for this label (already in RGB format)
        color_rgb_int = color_palette.get(label_name.lower(), color_palette["default"])
        color_rgb = (color_rgb_int[0] / 255.0, color_rgb_int[1] / 255.0, color_rgb_int[2] / 255.0)
        
        # Draw polygon if available
        if polygon_points is not None:
            # Create polygon patch
            polygon = Polygon(polygon_points, closed=True, 
                            facecolor=color_rgb, edgecolor=color_rgb, 
                            alpha=0.4, linewidth=2)
            ax.add_patch(polygon)
        
        # Draw bounding box if available
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            width = xmax - xmin
            height = ymax - ymin
            # Draw bounding box rectangle with dashed style
            bbox_rect = Rectangle((xmin, ymin), width, height,
                                 linewidth=2, edgecolor=color_rgb, 
                                 facecolor='none', linestyle='--', alpha=0.8)
            ax.add_patch(bbox_rect)
        
        # Add label text at top-left of bounding box or first polygon point
        if bbox is not None:
            x, y = bbox[0], bbox[1]
        elif polygon_points is not None and len(polygon_points) > 0:
            x, y = polygon_points[0]
        else:
            continue
        
        ax.text(x, y - 5, label_name, color=color_rgb, 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor=color_rgb, alpha=0.9, linewidth=1.5))
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save if output path specified
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved result to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize polygon segmentations from XML annotation files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test/segmentation_test.py 00004
  python test/segmentation_test.py 00004 --output test_result.jpg
  python test/segmentation_test.py 00004 --backend opencv
        """
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Base filename (without extension) of the image/XML pair in output folder"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Optional path to save the visualization result"
    )
    default_backend = "matplotlib" if HAS_MATPLOTLIB else "opencv"
    parser.add_argument(
        "--backend",
        type=str,
        choices=["opencv", "matplotlib"],
        default=default_backend,
        help=f"Backend to use for visualization (default: {default_backend})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory containing images and labels folders (default: output)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="input",
        help="Directory containing input images and labels folders for original bounding boxes (default: input)"
    )
    parser.add_argument(
        "--show-original-bbox",
        action="store_true",
        default=True,
        help="Show original bounding box from input XML file (default: True)"
    )
    
    args = parser.parse_args()
    
    # Build file paths
    base_name = args.filename
    image_path = os.path.join(args.output_dir, "images", f"{base_name}.jpg")
    xml_path = os.path.join(args.output_dir, "labels", f"{base_name}.xml")
    input_xml_path = os.path.join(args.input_dir, "labels", f"{base_name}.xml")
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        print(f"Available images in {args.output_dir}/images:")
        img_dir = os.path.join(args.output_dir, "images")
        if os.path.exists(img_dir):
            for f in os.listdir(img_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    print(f"  - {os.path.splitext(f)[0]}")
        sys.exit(1)
    
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found: {xml_path}")
        sys.exit(1)
    
    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from {image_path}")
        sys.exit(1)
    
    # Load segmentations
    print(f"Loading segmentations from: {xml_path}")
    try:
        segmentations = load_segmentations_from_xml(xml_path)
        print(f"Found {len(segmentations)} segmentations:")
        for i, (label, points, bbox) in enumerate(segmentations, 1):
            polygon_info = f"{len(points)} points" if points is not None else "no polygon"
            bbox_info = f"bbox: {bbox}" if bbox is not None else "no bbox"
            print(f"  {i}. {label}: {polygon_info}, {bbox_info}")
    except Exception as e:
        print(f"Error loading segmentations: {e}")
        sys.exit(1)
    
    if len(segmentations) == 0:
        print("Warning: No segmentations found in XML file")
        return
    
    # Load original bounding boxes from input XML if available
    original_bboxes = []
    if args.show_original_bbox and os.path.exists(input_xml_path):
        print(f"\nLoading original bounding boxes from: {input_xml_path}")
        try:
            input_segmentations = load_segmentations_from_xml(input_xml_path)
            for label, points, bbox in input_segmentations:
                if bbox is not None:
                    original_bboxes.append(bbox)
                    print(f"  Original bbox: {bbox} ({label})")
        except Exception as e:
            print(f"Warning: Could not load original bounding boxes: {e}")
    
    # Load color palette from label file
    color_palette = load_color_palette(LABEL_FILE)
    print(f"Loaded {len([k for k in color_palette.keys() if k != 'default'])} labels with colors")
    
    # Draw and display
    print(f"\nDisplaying visualization using {args.backend} backend...")
    if args.backend == "opencv":
        draw_segmentations_opencv(image, segmentations, original_bboxes if args.show_original_bbox else None, args.output, color_palette)
    else:
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available, falling back to opencv backend")
            draw_segmentations_opencv(image, segmentations, original_bboxes if args.show_original_bbox else None, args.output, color_palette)
        else:
            draw_segmentations_matplotlib(image, segmentations, original_bboxes if args.show_original_bbox else None, args.output, color_palette)
    
    print("Done!")


if __name__ == "__main__":
    main()

