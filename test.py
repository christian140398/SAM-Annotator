import os, glob, json, cv2, numpy as np
from pycocotools import mask as mask_utils
from segment_anything import sam_model_registry, SamPredictor
import xml.etree.ElementTree as ET
import torch
from tkinter import Tk, Canvas, Frame, Label, Button
from PIL import Image, ImageTk

# ===================== CONFIG =====================
IMG_DIR = r"input"                                # folder with images
XML_DIR = r"input"                                # VOC xmls (optional; same name as image)
OUT_JSON = r"output\instances_parts.json"         # COCO-style output
CHECKPOINT = r"models\sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"                          # "vit_b" | "vit_l" | "vit_h"
CATEGORIES = ["body", "rotor", "camera", "other"] # labels 1..4
CLIP_TO_XML_BOX = True                             # clip each mask to VOC bbox if available
SHOW_HELP_OVERLAY = False                          # now shown in keybind panel
# ==================================================

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
torch.set_grad_enabled(False)  # CPU only

# ---------- utils ----------
def load_voc_box(xml_path):
    if not os.path.isfile(xml_path):
        return None
    try:
        r = ET.parse(xml_path).getroot()
        obj = r.find("object")
        if obj is None: return None
        bb = obj.find("bndbox")
        xmin = int(float(bb.find("xmin").text))
        ymin = int(float(bb.find("ymin").text))
        xmax = int(float(bb.find("xmax").text))
        ymax = int(float(bb.find("ymax").text))
        return xmin, ymin, xmax, ymax
    except Exception:
        return None

def mask_to_rle(mask_bool):
    rle = mask_utils.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle

def apply_clip_to_box(mask_bool, box, H, W):
    if box is None:
        return mask_bool
    xmin, ymin, xmax, ymax = box
    xmin = max(0, min(W-1, xmin)); xmax = max(0, min(W-1, xmax))
    ymin = max(0, min(H-1, ymin)); ymax = max(0, min(H-1, ymax))
    box_mask = np.zeros((H, W), dtype=bool)
    box_mask[ymin:ymax+1, xmin:xmax+1] = True
    return mask_bool & box_mask

def cv2_to_pil(cv_img):
    """Convert OpenCV image (BGR) to PIL Image (RGB)"""
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)

def draw_overlay(base, masks, labels, box=None, current_label_idx=None, current_mask=None, current_points=None):
    img = base.copy()
    palette = {
        "body":  (0,255,0),
        "rotor": (255,200,0),
        "camera":(0,200,255),
        "other": (200,0,255)
    }
    # Draw finalized masks
    for m, lab in zip(masks, labels):
        color = palette.get(lab, (255,0,0))
        img[m] = (0.6*img[m] + 0.4*np.array(color, np.uint8)).astype(np.uint8)
        # put label text at the mask bbox top-left
        rle = mask_to_rle(m)
        x,y,w,h = mask_utils.toBbox({"size":[img.shape[0],img.shape[1]], "counts": rle["counts"].encode()})
        cv2.putText(img, lab, (int(x), max(0, int(y)-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    # Draw current segment being built (if any)
    if current_mask is not None:
        color = palette.get(CATEGORIES[current_label_idx-1] if current_label_idx else "other", (255,0,0))
        img[current_mask] = (0.5*img[current_mask] + 0.5*np.array(color, np.uint8)).astype(np.uint8)
    
    # Draw current click points - positive (green dot) and negative (red dot)
    if current_points is not None:
        for (x, y), is_positive in current_points:
            if is_positive:
                # Positive point: small green dot
                cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
                cv2.circle(img, (int(x), int(y)), 4, (0, 200, 0), 1)
            else:
                # Negative point: small red dot
                cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
                cv2.circle(img, (int(x), int(y)), 4, (0, 0, 200), 1)

    if box is not None:
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
    
    return img

# ---------- load images ----------
img_paths = sorted([p for p in glob.glob(os.path.join(IMG_DIR, "*")) if p.lower().endswith((".jpg",".jpeg",".png"))])
assert img_paths, f"No images found in {IMG_DIR}"

# ---------- COCO containers ----------
images_json = []
annotations_json = []
categories_json = [{"id": i+1, "name": n} for i, n in enumerate(CATEGORIES)]
ann_id = 1

# ---------- main loop ----------
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=CHECKPOINT)
predictor = SamPredictor(sam)

# Create tkinter window
root = Tk()
root.title("SAM Annotator")
root.geometry("1200x800")

# Handle window close (X button)
def on_closing():
    """Handle window close event"""
    global should_quit
    
    # Set quit flag - this will trigger saving and exit in the main loop
    should_quit = True
    # root.quit() breaks out of the event loop in the while loop
    # The main loop will then save annotations and call root.destroy()
    try:
        root.quit()
    except:
        pass  # Window might already be destroyed

root.protocol("WM_DELETE_WINDOW", on_closing)

# Nav bar at top
nav_frame = Frame(root, bg="#2c3e50", height=50)
nav_frame.pack(fill="x", side="top")
nav_frame.pack_propagate(False)
Label(nav_frame, text="SAM Annotator", bg="#2c3e50", fg="white", font=("Arial", 14, "bold")).pack(side="left", padx=20, pady=10)
# Placeholder for future nav features

# Main content area
content_frame = Frame(root)
content_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Image canvas
img_canvas = Canvas(content_frame, bg="#34495e", cursor="crosshair")
img_canvas.pack(side="left", fill="both", expand=True)

# Keybind panel in bottom right
keybind_frame = Frame(root, bg="#ecf0f1", relief="sunken", borderwidth=2)
keybind_frame.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)

keybind_title = Label(keybind_frame, text="Keyboard Shortcuts", bg="#ecf0f1", font=("Arial", 10, "bold"))
keybind_title.pack(padx=10, pady=(10, 5))

keybinds = [
    ("1-4", "Select label (body/rotor/camera/other)"),
    ("Left Click", "Add point (green = include area)"),
    ("Right Click", "Add point (red = exclude area)"),
    ("N", "Next segment (finalize current)"),
    ("S", "Save & next image"),
    ("U", "Undo last point"),
    ("Q", "Quit"),
]

for key, desc in keybinds:
    frame = Frame(keybind_frame, bg="#ecf0f1")
    frame.pack(fill="x", padx=10, pady=2)
    Label(frame, text=f"{key}:", width=12, anchor="w", bg="#ecf0f1", font=("Arial", 9, "bold")).pack(side="left")
    Label(frame, text=desc, anchor="w", bg="#ecf0f1", font=("Arial", 9)).pack(side="left", fill="x")

status_label = Label(keybind_frame, text="", bg="#ecf0f1", font=("Arial", 9), fg="#7f8c8d")
status_label.pack(padx=10, pady=(10, 10))

# Global state
img_idx = 0
current_label_idx = 2  # default "rotor"
current_points = []  # List of tuples: ((x, y), is_positive) where is_positive is True/False
current_mask = None
masks = []
labels = []
base_img = None
display_scale = 1.0
img_photo = None
pass_next = False
should_quit = False
box = None  # will be set per image

def update_image_display():
    """Update the displayed image on canvas"""
    global img_photo, display_scale, base_img, masks, labels, box, current_label_idx, current_mask, current_points
    if base_img is None:
        return
    
    disp = draw_overlay(base_img, masks, labels, box, current_label_idx, current_mask, current_points)
    pil_img = cv2_to_pil(disp)
    
    # Get canvas size
    canvas_width = img_canvas.winfo_width()
    canvas_height = img_canvas.winfo_height()
    
    if canvas_width > 1 and canvas_height > 1:
        # Calculate scale to fit canvas
        img_w, img_h = pil_img.size
        scale_w = canvas_width / img_w
        scale_h = canvas_height / img_h
        display_scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        new_w = int(img_w * display_scale)
        new_h = int(img_h * display_scale)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    img_photo = ImageTk.PhotoImage(pil_img)
    img_canvas.delete("all")
    img_canvas.create_image(canvas_width//2, canvas_height//2, image=img_photo, anchor="center")
    
    # Update status
    category = CATEGORIES[current_label_idx-1] if current_label_idx else "other"
    if current_points:
        pos_count = sum(1 for _, is_pos in current_points if is_pos)
        neg_count = sum(1 for _, is_pos in current_points if not is_pos)
        points_text = f"+{pos_count} -{neg_count}"
    else:
        points_text = "0"
    segments_count = len(masks)
    status_text = f"Label: {category} | Points: {points_text} | Segments: {segments_count}"
    status_label.config(text=status_text)

def canvas_to_image_coords(canvas_x, canvas_y):
    """Convert canvas coordinates to image coordinates"""
    if base_img is None:
        return None, None
    
    canvas_width = img_canvas.winfo_width()
    canvas_height = img_canvas.winfo_height()
    img_h, img_w = base_img.shape[:2]
    
    # Calculate image position accounting for centering
    img_display_w = int(img_w * display_scale)
    img_display_h = int(img_h * display_scale)
    
    offset_x = (canvas_width - img_display_w) // 2
    offset_y = (canvas_height - img_display_h) // 2
    
    # Convert to image coordinates
    img_x = int((canvas_x - offset_x) / display_scale)
    img_y = int((canvas_y - offset_y) / display_scale)
    
    # Clamp to image bounds
    img_x = max(0, min(img_w - 1, img_x))
    img_y = max(0, min(img_h - 1, img_y))
    
    return img_x, img_y

def update_mask_from_points():
    """Update the mask prediction based on current points (positive and negative)"""
    global current_mask, box, predictor
    
    if base_img is None:
        current_mask = None
        return
    
    # Separate positive and negative points
    positive_points = [(x, y) for (x, y), is_pos in current_points if is_pos]
    negative_points = [(x, y) for (x, y), is_pos in current_points if not is_pos]
    
    # Need at least one positive point to generate a mask
    if len(positive_points) == 0:
        current_mask = None
        return
    
    img_h, img_w = base_img.shape[:2]
    
    # Combine all points (positive first, then negative)
    all_points = positive_points + negative_points
    pts = np.array(all_points, dtype=np.float32)
    
    # Create labels: 1 for positive, 0 for negative
    lbl = np.array([1] * len(positive_points) + [0] * len(negative_points), dtype=np.int32)
    
    m_arr, scores, _ = predictor.predict(point_coords=pts, point_labels=lbl, multimask_output=True)
    best = m_arr[np.argmax(scores)].astype(bool)
    if box is not None:
        best = apply_clip_to_box(best, box, img_h, img_w)
    if best.sum() > 10:
        current_mask = best
    else:
        current_mask = None

def canvas_click(event):
    """Handle left mouse click on canvas - add positive point"""
    global current_points
    
    # Get click position in canvas coordinates
    canvas_x = event.x
    canvas_y = event.y
    
    # Convert to image coordinates
    img_x, img_y = canvas_to_image_coords(canvas_x, canvas_y)
    if img_x is None:
        return
    
    # Add as positive point (is_positive = True)
    current_points.append(((img_x, img_y), True))
    update_mask_from_points()
    update_image_display()

def canvas_right_click(event):
    """Handle right mouse click on canvas - add negative point (exclude area from segmentation)"""
    global current_points
    
    # Get click position in canvas coordinates
    canvas_x = event.x
    canvas_y = event.y
    
    # Convert to image coordinates
    img_x, img_y = canvas_to_image_coords(canvas_x, canvas_y)
    if img_x is None:
        return
    
    # Add as negative point (is_positive = False) - tells SAM to exclude this area
    current_points.append(((img_x, img_y), False))
    update_mask_from_points()
    update_image_display()

img_canvas.bind("<Button-1>", canvas_click)
img_canvas.bind("<Button-3>", canvas_right_click)  # Right click on Windows/Linux
img_canvas.bind("<Button-2>", canvas_right_click)  # Right click on Mac (middle button emulates right)
img_canvas.bind("<Configure>", lambda e: update_image_display())

def handle_key(event):
    """Handle keyboard events"""
    global current_label_idx, current_points, current_mask, masks, labels, pass_next, should_quit, img_idx, base_img
    
    key = event.char.lower() if event.char else ""
    keycode = event.keycode
    
    if key == 'q':
        # Quit without saving (only 's' saves)
        should_quit = True
        root.quit()
        return
    
    if key == 's':  # save and next image
        # finalize current segment if exists
        if current_mask is not None and len(current_points) > 0:
            label = CATEGORIES[current_label_idx-1] if current_label_idx else "other"
            masks.append(current_mask)
            labels.append(label)
            current_points = []
            current_mask = None
        pass_next = True
        root.quit()
        return
    
    if key == 'n':  # next segment (finalize current and start new)
        if current_mask is not None and len(current_points) > 0:
            label = CATEGORIES[current_label_idx-1] if current_label_idx else "other"
            masks.append(current_mask)
            labels.append(label)
        # Clear for new segment
        current_points = []
        current_mask = None
        update_image_display()
        return
    
    if key == 'u':
        # Undo: remove last point, or if no points, undo last finalized segment
        if current_points:
            current_points.pop()
            update_mask_from_points()
            update_image_display()
        elif masks:
            masks.pop()
            labels.pop()
            update_image_display()
        return
    
    if key in ('1', '2', '3', '4'):
        new_label = int(key)
        # If changing label, finalize current segment first
        if current_mask is not None and len(current_points) > 0 and current_label_idx != new_label:
            label = CATEGORIES[current_label_idx-1] if current_label_idx else "other"
            masks.append(current_mask)
            labels.append(label)
            current_points = []
            current_mask = None
        current_label_idx = new_label
        update_image_display()
        return

root.bind("<Key>", handle_key)
root.focus_set()

# Process images
img_idx = 0
while img_idx < len(img_paths):
    img_path = img_paths[img_idx]
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    base_img = img.copy()

    # per-image state
    xml_path = os.path.join(XML_DIR, os.path.splitext(os.path.basename(img_path))[0] + ".xml")
    box = load_voc_box(xml_path) if CLIP_TO_XML_BOX else None
    masks = []      # list of boolean arrays
    labels = []     # list of strings

    predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # compute SAM image features

    # register to COCO images (id = img_idx+1)
    image_id = img_idx + 1
    images_json.append({"id": image_id, "file_name": os.path.relpath(img_path, start=os.path.dirname(OUT_JSON)).replace("\\","/"),
                        "width": W, "height": H})

    # Reset per-image state
    current_points = []
    current_mask = None
    
    # Update window title
    root.title(f"SAM Annotator - {os.path.basename(img_path)} ({img_idx+1}/{len(img_paths)})")
    
    # Initial display
    update_image_display()
    
    # Event loop for this image
    pass_next = False
    should_quit = False
    while not pass_next and not should_quit:
        root.update()
        root.update_idletasks()
        if should_quit:
            # Don't save - just quit (only 's' key saves)
            break

    # write annotations for this image
    for m, lab in zip(masks, labels):
        rle = mask_to_rle(m)
        bbox = mask_utils.toBbox({"size":[H,W], "counts": rle["counts"].encode()}).tolist()
        category_id = CATEGORIES.index(lab) + 1
        annotations_json.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": rle,
            "area": float(m.sum()),
            "bbox": bbox,
            "iscrowd": 0
        })
        ann_id += 1

    img_idx += 1 if pass_next else 0
    if should_quit:
        break

# ---------- save COCO ----------
coco = {
    "images": images_json,
    "annotations": annotations_json,
    "categories": categories_json
}
with open(OUT_JSON, "w") as f:
    json.dump(coco, f, indent=2)
print(f"✅ Saved COCO annotations to {OUT_JSON}")
print(f"Images: {len(images_json)}  Annotations: {len(annotations_json)}")
root.destroy()
