import cv2
import numpy as np
import os
import random

# =========================
# Config
# =========================

# Input Paths
BACKGROUND_FOLDER = "/content/YOLOTrainingForDrone/input/backgrounds/"
OBJECTS_FOLDER = "/content/YOLOTrainingForDrone/input/objects/"

# Output Paths
OUTPUT_IMAGE_FOLDER = "/content/YOLOTrainingForDrone/output/images/"
OUTPUT_LABEL_FOLDER = "/content/YOLOTrainingForDrone/output/labels/"

# Generation Settings
NUM_IMAGES_TO_GENERATE = 1000

# Augmentation Settings
SCALE_RANGE = (0.05, 0.20)  # 5%–20% of background min dimension
ROTATION_RANGE = (0, 360)   # degrees


# =========================
# Helper Functions
# =========================

def create_output_folders():
    """Ensure the output directories for images and labels exist."""
    os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_FOLDER, exist_ok=True)


def get_background_files(folder):
    """Return a list of all valid image files in the background folder."""
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
    Load all RGBA object images from subfolders of root_folder.

    Folder structure:
        root_folder/
            car/*.png
            plane/*.png
            tent/*.png
            ...

    Class IDs are assigned in sorted folder-name order, and this
    matches the YAML generator.

    Returns:
        objects   - list of all RGBA images (for convenience)
        labels    - list of class names in class_id order
        class_map - list of (image, class_id) tuples
    """
    objects = []
    labels = []
    class_map = []

    # Discover class folders in sorted order
    for folder_name in sorted(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # PNG files inside this class folder
        png_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(".png")
        ])

        if not png_files:
            # no images in this class folder, skip it
            continue

        # Assign class_id based on order in labels
        class_id = len(labels)
        labels.append(folder_name)

        for filename in png_files:
            full_path = os.path.join(folder_path, filename)
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

            if img is None:
                print(f"Warning: Could not load object image {full_path}. Skipping.")
                continue

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
    foreground: RGBA image
    background: BGR ROI from background
    alpha_mask: single-channel alpha mask (uint8)
    """
    alpha_3 = cv2.cvtColor(alpha_mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0

    fg_rgb = foreground[:, :, :3].astype(float)
    bg = background.astype(float)

    blended = alpha_3 * fg_rgb + (1.0 - alpha_3) * bg
    return blended.astype(np.uint8)


def paste_with_transparency(background, foreground, x_offset, y_offset):
    """
    Paste a foreground RGBA image onto a BGR background at (x_offset, y_offset).
    Returns:
        - modified background image
        - bounding box (x, y, w, h) of the pasted foreground in background coords
    """
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]

    # If the foreground goes out of bounds, clip it
    if y_offset + fg_h > bg_h or x_offset + fg_w > bg_w:
        y_end = min(y_offset + fg_h, bg_h)
        x_end = min(x_offset + fg_w, bg_w)

        fg_h = y_end - y_offset
        fg_w = x_end - x_offset

        if fg_h <= 0 or fg_w <= 0:
            return None, None

        foreground = foreground[:fg_h, :fg_w]

    # Region of interest on the background
    roi = background[y_offset:y_offset + fg_h, x_offset:x_offset + fg_w]

    # Split foreground into RGB and alpha
    alpha_mask = foreground[:, :, 3]

    # Blend
    blended_roi = alpha_blend(foreground, roi, alpha_mask)

    # Write blended ROI back into a copy of the background
    output = background.copy()
    output[y_offset:y_offset + fg_h, x_offset:x_offset + fg_w] = blended_roi

    return output, (x_offset, y_offset, fg_w, fg_h)


def rotate_with_canvas(image, angle):
    """
    Rotate an RGBA image by 'angle' degrees, expanding the canvas so nothing is cut off.
    """
    h, w = image.shape[:2]
    cX, cY = w // 2, h // 2

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    # New bounding dimensions
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to account for translation
    M[0, 2] += (new_w / 2) - cX
    M[1, 2] += (new_h / 2) - cY

    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0, 0),  # transparent border
    )
    return rotated


# =========================
# Main Generation Logic
# =========================

def generate_training_data():
    """Generate synthetic training images and YOLO labels for multiple objects."""
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
            # Pick a random background
            bg_path = random.choice(background_files)
            bg_img = cv2.imread(bg_path)

            if bg_img is None:
                print(f"Warning: Could not read {bg_path}. Skipping.")
                continue

            bg_h, bg_w = bg_img.shape[:2]
            min_bg_dim = min(bg_h, bg_w)

            # Pick random object + its class ID
            obj_img, obj_class = random.choice(class_map)

            # ---- SCALE WHILE PRESERVING ASPECT RATIO ----
            obj_h, obj_w = obj_img.shape[:2]

            # target size: fraction of background's min dimension
            scale = random.uniform(*SCALE_RANGE)
            target_size = min_bg_dim * scale  # desired size of the longer side

            scale_factor = target_size / max(obj_h, obj_w)
            new_w = max(1, int(obj_w * scale_factor))
            new_h = max(1, int(obj_h * scale_factor))

            obj_scaled = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # ------------------------------------------------

            # Random rotation
            angle = random.uniform(*ROTATION_RANGE)
            obj_rot = rotate_with_canvas(obj_scaled, angle)
            final_h, final_w = obj_rot.shape[:2]

            # If rotated object is bigger than background, rescale it down (keep aspect ratio)
            if final_h >= bg_h or final_w >= bg_w:
                scale_factor2 = min((bg_h - 1) / final_h, (bg_w - 1) / final_w)
                scale_factor2 = max(scale_factor2, 1e-3)  # avoid 0
                final_w = max(1, int(final_w * scale_factor2))
                final_h = max(1, int(final_h * scale_factor2))
                obj_rot = cv2.resize(obj_rot, (final_w, final_h), interpolation=cv2.INTER_AREA)

            # Random placement (ensuring it fits)
            max_x = bg_w - final_w
            max_y = bg_h - final_h
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)

            # Paste object onto background
            result_img, obj_box = paste_with_transparency(bg_img, obj_rot, paste_x, paste_y)
            if result_img is None or obj_box is None:
                print(f"Warning: Failed to paste object on image {i+1}. Skipping.")
                continue

            # Unpack bounding box
            x_min, y_min, box_w, box_h = obj_box

            # Convert to YOLO format (normalized center x, center y, width, height)
            x_center = x_min + box_w / 2.0
            y_center = y_min + box_h / 2.0

            x_center_norm = x_center / bg_w
            y_center_norm = y_center / bg_h
            w_norm = box_w / bg_w
            h_norm = box_h / bg_h

            yolo_string = f"{obj_class} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"

            # Build filenames
            base_filename = f"synth_{i+1:05d}"
            img_path = os.path.join(OUTPUT_IMAGE_FOLDER, f"{base_filename}.jpg")
            label_path = os.path.join(OUTPUT_LABEL_FOLDER, f"{base_filename}.txt")

            # Save image & label
            cv2.imwrite(img_path, result_img)
            with open(label_path, "w") as f:
                f.write(yolo_string + "\n")

            if (i + 1) % 10 == 0:
                print(f"   ... Generated {i+1}/{NUM_IMAGES_TO_GENERATE} images")

        except Exception as e:
            print(f"Error during generation of image {i+1}: {e}. Skipping.")

    print(f"\n✅ Generation complete. {NUM_IMAGES_TO_GENERATE} images and labels saved to 'output/'.")


if __name__ == "__main__":
    generate_training_data()
