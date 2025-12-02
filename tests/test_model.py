import cv2
from pathlib import Path
from ultralytics import YOLO

# Config
MODEL_PATH = Path("/content/YOLOTrainingForDrone/TrainedModel/weights/best.pt")
VAL_DIR = Path("/content/YOLOTrainingForDrone/output/images/val")
RESULTS_DIR = Path("/content/YOLOTrainingForDrone/output/validation_results")

CONF_THRESHOLD = 0.50
DEVICE = 0  # GPU index (use 'cpu' if needed)


def load_model(model_path: Path) -> YOLO | None:
    """Load a YOLO model from disk, returning None on failure."""
    if not model_path.is_file():
        print(f"Error: Model file not found at {model_path}")
        return None

    try:
        model = YOLO(str(model_path))
        print(f"Loaded model from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None


def get_validation_images(folder: Path) -> list[Path]:
    """Return a list of image paths in the validation folder."""
    extensions = ("*.jpg", "*.jpeg", "*.png")
    images = [p for pattern in extensions for p in folder.glob(pattern)]

    if not images:
        print(f"Error: No images found in {folder}")
    else:
        print(f"Found {len(images)} images to process in {folder}")

    return images


def predict_and_save(model: YOLO, image_paths: list[Path]) -> None:
    """Run predictions on images and save plotted results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {RESULTS_DIR}")

    for img_path in image_paths:
        try:
            results = model(img_path, conf=CONF_THRESHOLD, device=DEVICE)
            plotted = results[0].plot()

            save_path = RESULTS_DIR / img_path.name
            cv2.imwrite(str(save_path), plotted)

            print(f"Saved: {save_path.name}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"\nPrediction complete. Saved {len(image_paths)} image(s) to {RESULTS_DIR}.")


def run_predictions() -> None:
    model = load_model(MODEL_PATH)
    if model is None:
        return

    images = get_validation_images(VAL_DIR)
    if not images:
        return

    predict_and_save(model, images)


if __name__ == "__main__":
    run_predictions()
