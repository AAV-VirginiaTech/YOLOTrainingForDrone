# ============================================================================
# YOLO Model Testing Script
# ============================================================================
# This script loads a trained YOLO model and runs inference on validation
# images, saving the annotated results with bounding boxes and predictions.
# Designed for testing drone-based object detection models.
# ============================================================================

import cv2
from pathlib import Path
from ultralytics import YOLO

# ============================================================================
# Configuration Section
# ============================================================================
# Define paths to the trained model, validation images, and output directory

# Path to the trained YOLO model weights file (best performing checkpoint)
MODEL_PATH = Path("/content/YOLOTrainingForDrone/TrainedModel/weights/best.pt")

# Directory containing validation images to test the model on
VAL_DIR = Path("/content/YOLOTrainingForDrone/output/images/val")

# Directory where annotated prediction results will be saved
RESULTS_DIR = Path("/content/YOLOTrainingForDrone/output/validation_results")

# Confidence threshold for detections (0.0-1.0)
# Only predictions with confidence >= this value will be shown
CONF_THRESHOLD = 0.50

# Device to run inference on
# 0 = first GPU, 1 = second GPU, etc. Use 'cpu' for CPU inference
DEVICE = 0


def load_model(model_path: Path) -> YOLO | None:
    """
    Load a YOLO model from disk.
    
    Args:
        model_path: Path object pointing to the .pt model weights file
    
    Returns:
        YOLO model object if successful, None if loading fails
    
    Note:
        This function performs validation to ensure the model file exists
        before attempting to load it, preventing cryptic error messages.
    """
    # Check if the model file actually exists at the specified path
    if not model_path.is_file():
        print(f"Error: Model file not found at {model_path}")
        return None

    try:
        # Load the YOLO model using Ultralytics library
        # Convert Path to string as YOLO constructor expects string path
        model = YOLO(str(model_path))
        print(f"Loaded model from: {model_path}")
        return model
    except Exception as e:
        # Catch any errors during model loading (corrupt file, wrong format, etc.)
        print(f"Error loading model from {model_path}: {e}")
        return None


def get_validation_images(folder: Path) -> list[Path]:
    """
    Collect all image files from the validation directory.
    
    Args:
        folder: Path object pointing to the directory containing validation images
    
    Returns:
        List of Path objects for each image file found (jpg, jpeg, png)
        Returns empty list if no images are found
    
    Note:
        Searches for common image extensions (jpg, jpeg, png).
        Case-sensitive on Linux/Mac, case-insensitive on Windows.
    """
    # Define supported image file extensions
    extensions = ("*.jpg", "*.jpeg", "*.png")
    
    # Use list comprehension to find all matching files
    # Iterates through each extension pattern and collects matching files
    images = [p for pattern in extensions for p in folder.glob(pattern)]

    # Provide feedback about the number of images found
    if not images:
        print(f"Error: No images found in {folder}")
    else:
        print(f"Found {len(images)} images to process in {folder}")

    return images


def predict_and_save(model: YOLO, image_paths: list[Path]) -> None:
    """
    Run YOLO predictions on a list of images and save annotated results.
    
    Args:
        model: Loaded YOLO model object ready for inference
        image_paths: List of Path objects pointing to images to process
    
    Returns:
        None (saves results to disk as side effect)
    
    Process:
        1. Creates output directory if it doesn't exist
        2. Iterates through each image
        3. Runs model inference with specified confidence threshold
        4. Plots bounding boxes and labels on the image
        5. Saves the annotated image to the results directory
    
    Note:
        Failed individual images will be skipped with an error message,
        but processing will continue for remaining images.
    """
    # Create the results directory if it doesn't already exist
    # parents=True creates parent directories as needed
    # exist_ok=True prevents error if directory already exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {RESULTS_DIR}")

    # Process each image in the list
    for img_path in image_paths:
        try:
            # Run YOLO inference on the current image
            # conf: minimum confidence threshold for detections
            # device: GPU/CPU to use for inference
            results = model(img_path, conf=CONF_THRESHOLD, device=DEVICE)
            
            # Generate a plotted image with bounding boxes and labels
            # results[0] gets the first (and only) image result
            # plot() returns a numpy array (BGR format) with annotations drawn
            plotted = results[0].plot()

            # Construct the save path using the same filename as input
            save_path = RESULTS_DIR / img_path.name
            
            # Save the annotated image using OpenCV
            # Convert Path to string as cv2.imwrite expects string path
            cv2.imwrite(str(save_path), plotted)

            # Provide feedback on successful save
            print(f"Saved: {save_path.name}")
            
        except Exception as e:
            # Catch and report any errors during processing
            # Could be file read errors, inference errors, or write errors
            print(f"Error processing {img_path}: {e}")

    # Summary message after all images have been processed
    print(f"\nPrediction complete. Saved {len(image_paths)} image(s) to {RESULTS_DIR}.")


def run_predictions() -> None:
    """
    Main orchestration function that coordinates the prediction workflow.
    
    This function:
        1. Loads the YOLO model from disk
        2. Collects validation images from the specified directory
        3. Runs predictions and saves annotated results
    
    Returns:
        None
    
    Note:
        Early returns on failure ensure we don't proceed with invalid state.
        Each step validates its inputs before proceeding to the next.
    """
    # Step 1: Load the trained model
    model = load_model(MODEL_PATH)
    if model is None:
        # Exit early if model loading failed
        return

    # Step 2: Collect all validation images
    images = get_validation_images(VAL_DIR)
    if not images:
        # Exit early if no images were found
        return

    # Step 3: Run predictions and save results
    predict_and_save(model, images)


# ============================================================================
# Script Entry Point
# ============================================================================
if __name__ == "__main__":
    # Execute the main prediction workflow when script is run directly
    # (not imported as a module)
    run_predictions()