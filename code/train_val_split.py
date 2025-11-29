import random
import shutil
from pathlib import Path

# Config
DATA_PATH = Path("/content/YOLOTrainingForDrone/output")
IMAGE_PATH = DATA_PATH / "images"
LABEL_PATH = DATA_PATH / "labels"
VAL_SPLIT_PERCENT = 0.20


def move_files(images, img_dest, label_dest):
    img_dest.mkdir(parents=True, exist_ok=True)
    label_dest.mkdir(parents=True, exist_ok=True)

    moved = 0
    for img_path in images:
        img_path = Path(img_path)
        label_path = LABEL_PATH / (img_path.stem + ".txt")

        if not label_path.exists():
            print(f"Warning: Label not found for {img_path}. Skipping.")
            continue

        shutil.move(str(img_path), img_dest / img_path.name)
        shutil.move(str(label_path), label_dest / label_path.name)
        moved += 1

    return moved


def create_split():
    print("Creating train/val split...")

    all_images = list(IMAGE_PATH.glob("*.jpg"))
    if not all_images:
        print(f"Error: No .jpg images found in {IMAGE_PATH}. Did you run data generation?")
        return

    random.shuffle(all_images)
    split_index = int(len(all_images) * VAL_SPLIT_PERCENT)
    val_images = all_images[:split_index]
    train_images = all_images[split_index:]

    train_img_dir = IMAGE_PATH / "train"
    val_img_dir = IMAGE_PATH / "val"
    train_label_dir = LABEL_PATH / "train"
    val_label_dir = LABEL_PATH / "val"

    print(f"Moving {len(val_images)} files to validation set...")
    val_moved = move_files(val_images, val_img_dir, val_label_dir)

    print(f"Moving {len(train_images)} files to training set...")
    train_moved = move_files(train_images, train_img_dir, train_label_dir)

    print("\nSplit complete.")
    print(f"  Total images: {len(all_images)}")
    print(f"  Training set: {train_moved} images")
    print(f"  Validation set: {val_moved} images")


if __name__ == "__main__":
    create_split()
