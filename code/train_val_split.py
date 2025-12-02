"""
Create a train/val split by moving paired YOLO images and labels.

Assumptions:
- Images are `.jpg` files in `DATA_PATH/images`.
- Labels are `.txt` files with the same basename in `DATA_PATH/labels`.
- Both images and labels are MOVED (not copied) into `train/` and `val/` subfolders.
  This is destructive: the originals under `images/` and `labels/` are removed.

Notes:
- `DATA_PATH` currently points to a Colab-style path (`/content/...`).
  Update it to your local project path when running outside Colab, e.g.:
    DATA_PATH = Path(__file__).resolve().parents[1] / "output"
- Shuffling uses Python's global RNG; set `random.seed(...)` before calling
  `create_split()` for reproducible splits if needed.
"""

import random
import shutil
from pathlib import Path

# --- Config ---
# Root directory containing `images/` and `labels/` subfolders.
# Update this for your environment if not running in Colab.
DATA_PATH = Path("/content/YOLOTrainingForDrone/output")

# Location of input images and labels before splitting.
IMAGE_PATH = DATA_PATH / "images"
LABEL_PATH = DATA_PATH / "labels"

# Fraction of the dataset to allocate to validation.
VAL_SPLIT_PERCENT = 0.20


def move_files(images, img_dest, label_dest):
    """
    Move each image in `images` and its corresponding label to destination folders.

    Parameters:
    - images: iterable of image file paths (str or Path) to move.
    - img_dest: Path to destination directory for images (e.g., `images/train`).
    - label_dest: Path to destination directory for labels (e.g., `labels/train`).

    Behavior:
    - Ensures destination directories exist.
    - For each image, looks up a label with the same stem and `.txt` extension.
      If the label is missing, logs a warning and skips that image.
    - Moves both image and label to their respective destinations.

    Returns:
    - moved (int): count of images (and labels) successfully moved.
    """
    # Ensure destination directories exist.
    img_dest.mkdir(parents=True, exist_ok=True)
    label_dest.mkdir(parents=True, exist_ok=True)

    moved = 0
    for img_path in images:
        img_path = Path(img_path)
        # Expected YOLO label path: same stem as image, `.txt` extension.
        label_path = LABEL_PATH / (img_path.stem + ".txt")

        # Skip images without a matching label to avoid unpaired samples.
        if not label_path.exists():
            print(f"Warning: Label not found for {img_path}. Skipping.")
            continue

        # Move the image and its label. This is destructive (files are removed from source).
        shutil.move(str(img_path), img_dest / img_path.name)
        shutil.move(str(label_path), label_dest / label_path.name)
        moved += 1

    return moved


def create_split():
    """
    Shuffle dataset and move a percentage to validation, the rest to training.

    Source:
    - Images are expected directly under `IMAGE_PATH` with `.jpg` extension.

    Destinations created (if missing):
    - `images/train`, `images/val`
    - `labels/train`, `labels/val`

    Notes:
    - Only `.jpg` images are considered; adjust the glob if needed.
    - The split is randomized; call `random.seed(...)` beforehand for determinism.
    """
    print("Creating train/val split...")

    # Collect all `.jpg` images from the root images directory.
    all_images = list(IMAGE_PATH.glob("*.jpg"))
    if not all_images:
        print(f"Error: No .jpg images found in {IMAGE_PATH}. Did you run data generation?")
        return

    # Randomize order and compute split index.
    random.shuffle(all_images)
    split_index = int(len(all_images) * VAL_SPLIT_PERCENT)

    # Partition into validation and training sets by index.
    val_images = all_images[:split_index]
    train_images = all_images[split_index:]

    # Define destination directories for images and labels.
    train_img_dir = IMAGE_PATH / "train"
    val_img_dir = IMAGE_PATH / "val"
    train_label_dir = LABEL_PATH / "train"
    val_label_dir = LABEL_PATH / "val"

    # Move validation files first.
    print(f"Moving {len(val_images)} files to validation set...")
    val_moved = move_files(val_images, val_img_dir, val_label_dir)

    # Then move training files.
    print(f"Moving {len(train_images)} files to training set...")
    train_moved = move_files(train_images, train_img_dir, train_label_dir)

    # Final summary.
    print("\nSplit complete.")
    print(f"  Total images: {len(all_images)}")
    print(f"  Training set: {train_moved} images")
    print(f"  Validation set: {val_moved} images")


if __name__ == "__main__":
    create_split()