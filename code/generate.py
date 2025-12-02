"""
Generate synthetic training images and YOLO labels by compositing RGBA object
cutouts onto random background photos. Each output consists of:
- a composite `.jpg` in `output/images/`
- a YOLO-format `.txt` label in `output/labels/` with: class cx cy w h (normalized)

Assumptions and tips:
- Object assets are PNGs with alpha (RGBA) organized by class subfolder under `input/objects/`.
- Backgrounds are JPG/PNG images under `input/backgrounds/`.
- Paths below are set for Google Colab (`/content/...`). If running locally,
  change them to your local project paths (e.g., `./input/backgrounds/`).
"""

import cv2
import numpy as np
import os
import random

# =========================
# Config
# =========================

# Input Paths (Colab-style). Change to local relative paths if needed:
# Example local override:
# BACKGROUND_FOLDER = "./input/backgrounds/"
# OBJECTS_FOLDER = "./input/objects/"
BACKGROUND_FOLDER = "/content/YOLOTrainingForDrone/input/backgrounds/"
OBJECTS_FOLDER = "/content/YOLOTrainingForDrone/input/objects/"

# Output Paths
# Example local override:
# OUTPUT_IMAGE_FOLDER = "./output/images/"
# OUTPUT_LABEL_FOLDER = "./output/labels/"
OUTPUT_IMAGE_FOLDER = "/content/YOLOTrainingForDrone/output/images/"
OUTPUT_LABEL_FOLDER = "/content/YOLOTrainingForDrone/output/labels/"

# Generation Settings
NUM_IMAGES_TO_GENERATE = 1000  # total number of composites to create

# Augmentation Settings
SCALE_RANGE = (0.05, 0.20)  # as a fraction of the background's min dimension
ROTATION_RANGE = (0, 360)   # degrees; select uniformly at random


# =========================
# Helper Functions
# =========================

def create_output_folders():
    """Ensure the output directories for images and labels exist (idempotent)."""
    os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_FOLDER, exist_ok=True)


def get_background_files(folder):
    """
    Return a list of all valid background image file paths in the given folder.

    Valid extensions: .jpg, .jpeg, .png (case-insensitive).
    Exits the process early if no background images are found.
    """
    valid_extensions = {".jpg", ".jpeg", ".png"}
    background_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    if not background_files:
        print(f"Error: No background images found in {folder}.")
        exit()

    return background_files


def load_object_images(root_folder):
    """
    Load all RGBA object images (PNG with alpha) from subfolders of `root_folder`.

    Folder structure (class names taken from folder names):
        root_folder/
            car/*.png
            plane/*.png
            tent/*.png
            ...

    Class IDs are assigned in sorted folder-name order to ensure deterministic
    mapping and to match any separate YAML class ordering you might generate.

    Returns:
        objects   - list[np.ndarray]: all RGBA object images (for convenience)
        labels    - list[str]: class names in class_id order
        class_map - list[tuple[np.ndarray, int]]: (image, class_id) pairs
    """
    objects = []
    labels = []
    class_map = []

    # Discover class folders in sorted order so class IDs are deterministic
    for folder_name in sorted(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Collect PNG files (sorted for reproducibility)
        png_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(".png")
        ])

        if not png_files:
            # No images in this class folder, skip it
            continue

        # Assign class_id based on current length of labels
        class_id = len(labels)
        labels.append(folder_name)

        for filename in png_files:
            full_path = os.path.join(folder_path, filename)
            # IMREAD_UNCHANGED preserves alpha channel if present
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

            if img is None:
                print(f"Warning: Could not load object image {full_path}. Skipping.")
                continue

            # Require RGBA (4 channels) so we can alpha-blend cleanly
            if img.shape[2] != 4:
                print(f"Warning: Object image {full_path} has no alpha channel. Skipping.")
                continue

            objects.append(img)
            class_map.append((img, class_id))

    if not class_map:
        print(f"Error: No valid RGBA object images found in subfolders of {root_folder}.")
        exit()

    return objects, labels, class_map


def alpha_blend(foreground, background, alpha_mask):
    """
    Alpha-blend a foreground with alpha onto a background ROI.

    Args:
        foreground: HxWx4 RGBA image (uint8)
        background: HxWx3 BGR ROI from background (uint8)
        alpha_mask: HxW single-channel alpha mask (uint8, 0..255)

    Returns:
        blended: HxWx3 BGR image (uint8)
    """
    # Expand single-channel alpha to 3 channels, normalize to [0, 1]
    alpha_3 = cv2.cvtColor(alpha_mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0

    fg_rgb = foreground[:, :, :3].astype(float)
    bg = background.astype(float)

    # Standard alpha compositing: out = a*fg + (1-a)*bg
    blended = alpha_3 * fg_rgb + (1.0 - alpha_3) * bg
    return blended.astype(np.uint8)


def paste_with_transparency(background, foreground, x_offset, y_offset):
    """
    Paste a foreground RGBA image onto a BGR background at (x_offset, y_offset).

    - Clips the foreground if it would extend past background bounds.
    - Performs alpha blending with the foreground's alpha channel.

    Returns:
        output: background copy with foreground composited (np.ndarray) or None
        bbox:   (x, y, w, h) of the pasted foreground in background coordinates or None
    """
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]

    # If the foreground would go out of bounds, clip it to fit
    if y_offset + fg_h > bg_h or x_offset + fg_w > bg_w:
        y_end = min(y_offset + fg_h, bg_h)
        x_end = min(x_offset + fg_w, bg_w)

        fg_h = y_end - y_offset
        fg_w = x_end - x_offset

        if fg_h <= 0 or fg_w <= 0:
            return None, None

        foreground = foreground[:fg_h, :fg_w]

    # Define background region of interest (ROI) to receive the foreground
    roi = background[y_offset:y_offset + fg_h, x_offset:x_offset + fg_w]

    # Split foreground into RGB and alpha
    alpha_mask = foreground[:, :, 3]

    # Alpha-blend the foreground onto the ROI
    blended_roi = alpha_blend(foreground, roi, alpha_mask)

    # Write blended ROI back into a copy of the background
    output = background.copy()
    output[y_offset:y_offset + fg_h, x_offset:x_offset + fg_w] = blended_roi

    return output, (x_offset, y_offset, fg_w, fg_h)


def rotate_with_canvas(image, angle):
    """
    Rotate an RGBA image by 'angle' degrees, expanding the canvas so nothing is cut off.

    Returns:
        rotated RGBA image with transparent borders where needed.
    """
    h, w = image.shape[:2]
    cX, cY = w // 2, h // 2

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    # Compute new bounding dimensions to fit the entire rotated image
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust transform for the translation to the new center
    M[0, 2] += (new_w / 2) - cX
    M[1, 2] += (new_h / 2) - cY

    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0, 0),  # transparent padding
    )
    return rotated


# =========================
# Main Generation Logic
# =========================

def generate_training_data():
    """
    Generate synthetic training images and YOLO labels for multiple objects.

    Workflow per image:
    1) Choose a random background.
    2) Choose a random object class and asset.
    3) Scale the object to a fraction of the background's min dimension.
    4) Randomly rotate the object (with canvas expansion).
    5) Ensure object fits; rescale down if needed.
    6) Randomly place on the background and alpha-composite.
    7) Compute YOLO label (normalized cx, cy, w, h) and save files.
    """
    print("Starting data generation...")

    create_output_folders()

    # Load all object images and assign class IDs based on folder names
    objects, labels, class_map = load_object_images(OBJECTS_FOLDER)
    print(f"✅ Loaded {len(labels)} class(es) from {OBJECTS_FOLDER}:")
    for cid, name in enumerate(labels):
        print(f"   Class {cid}: {name}")

    # Load background images
    background_files = get_background_files(BACKGROUND_FOLDER)
    print(f"✅ Found {len(background_files)} background images.")

    # Generation loop
    for i in range(NUM_IMAGES_TO_GENERATE):
        try:
            # 1) Pick a random background and read it (BGR)
            bg_path = random.choice(background_files)
            bg_img = cv2.imread(bg_path)

            if bg_img is None:
                print(f"Warning: Could not read {bg_path}. Skipping.")
                continue

            bg_h, bg_w = bg_img.shape[:2]
            min_bg_dim = min(bg_h, bg_w)

            # 2) Pick a random object image and its class ID
            obj_img, obj_class = random.choice(class_map)

            # 3) Scale while preserving aspect ratio
            obj_h, obj_w = obj_img.shape[:2]
            # target size: fraction of background's min dimension
            scale = random.uniform(*SCALE_RANGE)
            target_size = min_bg_dim * scale  # desired size of the longer side

            scale_factor = target_size / max(obj_h, obj_w)
            new_w = max(1, int(obj_w * scale_factor))
            new_h = max(1, int(obj_h * scale_factor))

            obj_scaled = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 4) Random rotation (canvas expands to keep full object)
            angle = random.uniform(*ROTATION_RANGE)
            obj_rot = rotate_with_canvas(obj_scaled, angle)
            final_h, final_w = obj_rot.shape[:2]

            # 5) If rotated object is bigger than background, rescale it down (keep aspect)
            if final_h >= bg_h or final_w >= bg_w:
                scale_factor2 = min((bg_h - 1) / final_h, (bg_w - 1) / final_w)
                scale_factor2 = max(scale_factor2, 1e-3)  # avoid zero or negative
                final_w = max(1, int(final_w * scale_factor2))
                final_h = max(1, int(final_h * scale_factor2))
                obj_rot = cv2.resize(obj_rot, (final_w, final_h), interpolation=cv2.INTER_AREA)

            # 6) Random placement ensuring it fits within background bounds
            max_x = bg_w - final_w
            max_y = bg_h - final_h
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)

            # Paste object onto background using alpha compositing
            result_img, obj_box = paste_with_transparency(bg_img, obj_rot, paste_x, paste_y)
            if result_img is None or obj_box is None:
                print(f"Warning: Failed to paste object on image {i+1}. Skipping.")
                continue

            # Unpack bounding box in background coords
            x_min, y_min, box_w, box_h = obj_box

            # 7) Convert bbox to YOLO format (normalized center x, center y, width, height)
            x_center = x_min + box_w / 2.0
            y_center = y_min + box_h / 2.0

            x_center_norm = x_center / bg_w
            y_center_norm = y_center / bg_h
            w_norm = box_w / bg_w
            h_norm = box_h / bg_h

            # One object per image in this pipeline; extend to multiple by repeating paste steps
            yolo_string = f"{obj_class} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"

            # Build output filenames
            base_filename = f"synth_{i+1:05d}"
            img_path = os.path.join(OUTPUT_IMAGE_FOLDER, f"{base_filename}.jpg")
            label_path = os.path.join(OUTPUT_LABEL_FOLDER, f"{base_filename}.txt")

            # Save image & corresponding YOLO label
            cv2.imwrite(img_path, result_img)
            with open(label_path, "w") as f:
                f.write(yolo_string + "\n")

            # Occasional progress logging
            if (i + 1) % 10 == 0:
                print(f"   ... Generated {i+1}/{NUM_IMAGES_TO_GENERATE} images")

        except Exception as e:
            # Keep going even if one iteration fails (robustness)
            print(f"Error during generation of image {i+1}: {e}. Skipping.")

    print(f"\n✅ Generation complete. {NUM_IMAGES_TO_GENERATE} images and labels saved to 'output/'.")


if __name__ == "__main__":
    generate_training_data()