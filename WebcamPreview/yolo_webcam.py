# Import required libraries
import os
import cv2  # OpenCV for image processing
import base64  # For encoding/decoding base64 data
import numpy as np  # For numerical array operations
from io import BytesIO  # For in-memory binary streams
from PIL import Image  # For image format conversions
from IPython.display import Javascript, display  # For running JavaScript in Jupyter
from google.colab.output import eval_js  # For executing JS and getting return values in Colab


# Confidence threshold for YOLO detections (only show predictions above 70%)
CONF_THRESHOLD = 0.70

def load_js(js_path):
    """
    Load the webcam JS into the notebook.
    
    Args:
        js_path: Path to the JavaScript file containing webcam capture code
    
    Raises:
        FileNotFoundError: If the JS file doesn't exist at the specified path
    """
    if not os.path.exists(js_path):
        raise FileNotFoundError(f"JS file not found at: {js_path}")
    with open(js_path, "r") as f:
        js_code = f.read()
    # Display and execute the JavaScript code in the notebook
    display(Javascript(js_code))


def js_to_image(img_data_url: str) -> np.ndarray:
    """
    Convert a JS dataURL (e.g. 'data:image/jpeg;base64,...') 
    into an OpenCV BGR image.
    
    Args:
        img_data_url: Base64-encoded image data URL from JavaScript
    
    Returns:
        numpy array representing the image in BGR format (OpenCV convention),
        or None if input is empty
    """
    if not img_data_url:
        return None

    # Strip the 'data:image/jpeg;base64,' prefix to get only the base64 data
    header, encoded = img_data_url.split(",", 1)
    # Decode base64 string to bytes
    img_bytes = base64.b64decode(encoded)

    # Convert bytes to PIL Image in RGB format
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    # Convert PIL Image to numpy array
    img = np.array(pil_img)
    # Convert RGB -> BGR for OpenCV compatibility (OpenCV uses BGR by default)
    img = img[:, :, ::-1]
    return img


def bbox_to_bytes(overlay_rgba: np.ndarray) -> str:
    """
    Convert RGBA overlay (numpy array) to a PNG dataURL string
    so it can be drawn as <img> over the video in JS.
    
    Args:
        overlay_rgba: RGBA image as numpy array containing bounding boxes and labels
    
    Returns:
        Base64-encoded PNG data URL string, or empty string if input is None
    """
    if overlay_rgba is None:
        return ""

    # Convert numpy array to PIL Image with RGBA mode
    pil_img = Image.fromarray(overlay_rgba, mode="RGBA")
    # Create in-memory buffer to store PNG data
    buff = BytesIO()
    # Save image to buffer as PNG format
    pil_img.save(buff, format="PNG")
    # Encode PNG bytes as base64
    b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    # Return as data URL that can be used in HTML/JS
    return "data:image/png;base64," + b64




def start_yolo_webcam(
    model_path="/content/YOLOTrainingForDrone/TrainedModel/weights/best.pt",
    js_path="/content/YOLOTrainingForDrone/WebcamPreview/webcam_js_code.js"
):
    """
    Start real-time YOLO object detection on webcam feed in Google Colab.
    
    This function loads a YOLO model, captures frames from the webcam via JavaScript,
    runs inference on each frame, and overlays bounding boxes with class labels.
    
    Args:
        model_path: Path to the trained YOLO model weights file
        js_path: Path to the JavaScript file for webcam capture
    """
    # Import YOLO from ultralytics
    from ultralytics import YOLO
    # Load the trained YOLO model
    model = YOLO(model_path)
    # Load and execute the webcam JavaScript code
    load_js(js_path)

    # Initial status message displayed to the user
    label_html = "Running YOLO... (click video to stop)"
    # Empty bbox initially (no detections yet)
    bbox = ""

    # Get class name mapping from model (can be dict or list)
    names = model.names

    # Main loop: continuously process webcam frames
    while True:
        # Call JavaScript function to get the next frame from webcam
        # Pass current status message and bounding box overlay
        js_reply = eval_js(f'stream_frame("{label_html}", "{bbox}")')
        # If js_reply is empty/None, user stopped the webcam
        if not js_reply:
            break

        # Convert the base64 image data from JS to OpenCV format
        frame = js_to_image(js_reply["img"])
        # Run YOLO inference on the frame
        # verbose=False: suppress console output
        # conf=CONF_THRESHOLD: only return detections above confidence threshold
        results = model(frame, verbose=False, conf=CONF_THRESHOLD)[0]

        # Create a transparent overlay for drawing boxes and labels
        h, w = frame.shape[:2]  # Get frame height and width
        # RGBA overlay: 4 channels (Red, Green, Blue, Alpha)
        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        # Process detected objects if any exist
        if results.boxes:
            for box in results.boxes:
                # Extract bounding box coordinates (x1, y1) = top-left, (x2, y2) = bottom-right
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                # Extract class ID (which object class was detected)
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                # Extract confidence score (how confident the model is)
                conf = float(box.conf[0]) if box.conf is not None else 0.0

                # Clamp coordinates to frame boundaries to prevent out-of-bounds errors
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                # Skip invalid boxes (zero or negative width/height)
                if x2 <= x1 or y2 <= y1:
                    continue

                # --- DRAW BOUNDING BOX ---
                cv2.rectangle(
                    overlay,
                    (x1, y1),  # Top-left corner
                    (x2, y2),  # Bottom-right corner
                    (0, 255, 0, 255),  # Color: green (RGBA), fully opaque
                    2  # Line thickness
                )

                # --- CREATE AND DRAW LABEL TEXT ---
                # Look up class name from class ID
                if isinstance(names, dict):
                    # If names is a dictionary, use .get() with fallback
                    cls_name = names.get(cls_id, f"id:{cls_id}")
                else:  # names is a list
                    # If names is a list, index it with bounds checking
                    cls_name = names[cls_id] if 0 <= cls_id < len(names) else f"id:{cls_id}"

                # Format label with class name and confidence percentage
                label_text = f"{cls_name} {conf * 100:.1f}%"

                # Position text above the bounding box
                text_x = x1
                text_y = max(y1 - 7, 12)  # 7 pixels above box, minimum 12 pixels from top

                # Measure text size to create properly-sized background box
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text,
                    cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                    0.6,  # Font scale
                    2  # Thickness
                )

                # Draw filled background rectangle for text (makes text readable)
                cv2.rectangle(
                    overlay,
                    (text_x, text_y - text_h - baseline),  # Top-left of text background
                    (text_x + text_w, text_y + baseline),  # Bottom-right of text background
                    (0, 255, 0, 255),  # Green background, fully opaque
                    thickness=-1  # Filled rectangle
                )

                # Draw the actual text on top of the background
                cv2.putText(
                    overlay,
                    label_text,
                    (text_x, text_y),  # Bottom-left position of text
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,  # Font scale
                    (255, 255, 255, 255),  # White text, fully opaque
                    2,  # Thickness
                    cv2.LINE_AA  # Anti-aliased line (smoother text)
                )

        # Set alpha channel: make transparent where no drawing occurred, opaque where drawn
        # Check if any RGB channel is non-zero (i.e., something was drawn there)
        mask = overlay[:, :, :3].max(axis=2) > 0
        # Set alpha to 255 (opaque) where mask is True, 0 (transparent) elsewhere
        overlay[:, :, 3] = mask.astype(np.uint8) * 255

        # Convert the overlay to base64 PNG for JavaScript to display
        bbox = bbox_to_bytes(overlay)

    # Loop exited (user stopped webcam)
    print("Webcam YOLO stopped.")