import os
import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from IPython.display import Javascript, display
from google.colab.output import eval_js



CONF_THRESHOLD = 0.70

def load_js(js_path):
    """Load the webcam JS into the notebook."""
    if not os.path.exists(js_path):
        raise FileNotFoundError(f"JS file not found at: {js_path}")
    with open(js_path, "r") as f:
        js_code = f.read()
    display(Javascript(js_code))


def js_to_image(img_data_url: str) -> np.ndarray:
    """
    Convert a JS dataURL (e.g. 'data:image/jpeg;base64,...') 
    into an OpenCV BGR image.
    """
    if not img_data_url:
        return None

    # strip 'data:image/jpeg;base64,' prefix
    header, encoded = img_data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)

    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img = np.array(pil_img)
    # convert RGB -> BGR for OpenCV
    img = img[:, :, ::-1]
    return img


def bbox_to_bytes(overlay_rgba: np.ndarray) -> str:
    """
    Convert RGBA overlay (numpy array) to a PNG dataURL string
    so it can be drawn as <img> over the video in JS.
    """
    if overlay_rgba is None:
        return ""

    pil_img = Image.fromarray(overlay_rgba, mode="RGBA")
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return "data:image/png;base64," + b64




def start_yolo_webcam(
    model_path="/content/YOLOTrainingForDrone/TrainedModel/weights/best.pt",
    js_path="/content/YOLOTrainingForDrone/WebcamPreview/webcam_js_code.js"
):
    from ultralytics import YOLO
    model = YOLO(model_path)
    load_js(js_path)

    label_html = "Running YOLO... (click video to stop)"
    bbox = ""

    # class name lookup (handles dict or list)
    names = model.names

    while True:
        js_reply = eval_js(f'stream_frame("{label_html}", "{bbox}")')
        if not js_reply:
            break

        frame = js_to_image(js_reply["img"])
        results = model(frame, verbose=False, conf=CONF_THRESHOLD)[0]

        # Overlay for boxes + labels
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        if results.boxes:
            for box in results.boxes:
                # coords, class, conf
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                conf = float(box.conf[0]) if box.conf is not None else 0.0

                # clamp
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                # skip junk
                if x2 <= x1 or y2 <= y1:
                    continue

                # --- BOX ---
                cv2.rectangle(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0, 255),   # green, opaque
                    2
                )

                # --- LABEL TEXT ---
                # name lookup
                if isinstance(names, dict):
                    cls_name = names.get(cls_id, f"id:{cls_id}")
                else:  # list
                    cls_name = names[cls_id] if 0 <= cls_id < len(names) else f"id:{cls_id}"

                label_text = f"{cls_name} {conf * 100:.1f}%"

                # text position
                text_x = x1
                text_y = max(y1 - 7, 12)

                # measure text
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )

                # background box for text
                cv2.rectangle(
                    overlay,
                    (text_x, text_y - text_h - baseline),
                    (text_x + text_w, text_y + baseline),
                    (0, 255, 0, 255),   # solid green
                    thickness=-1
                )

                # text (thick, white)
                cv2.putText(
                    overlay,
                    label_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255, 255),  # white
                    2,
                    cv2.LINE_AA
                )

        # set alpha where *any* color channel is non-zero
        mask = overlay[:, :, :3].max(axis=2) > 0
        overlay[:, :, 3] = mask.astype(np.uint8) * 255

        bbox = bbox_to_bytes(overlay)

    print("Webcam YOLO stopped.")
