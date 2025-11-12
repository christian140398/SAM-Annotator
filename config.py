"""
Configuration file for SAM Annotator
Users can specify export format and other settings here
"""

# Export format: "voc" or "coco"
EXPORT_FORMAT = "coco"  

# COCO export settings (only used when EXPORT_FORMAT = "coco")
COCO_CATEGORIES = None  

# Whether bounding box XML files exist in input/labels folder
# If True: system will load images and corresponding XML files with bounding boxes
# If False: system will only load images (no XML files expected)
BOUNDING_BOX_EXISTS = False

# Label name for bounding boxes when saved to output/bb_labels folder
# This label will be used in the saved annotation files (VOC XML or COCO JSON)
BB_LABEL = "drone"
