import os
import json
import cv2
from pathlib import Path
from datetime import datetime
from backend.trace_detector import detect_traces
from ui.theme import theme  # optional for color choices

# ------------------------- Folders -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
TRACE_SAVE_DIR = BASE_DIR / "VisionBoard-Proj/pcb_trace_results"
TRACE_LOG_DIR = os.path.join(TRACE_SAVE_DIR, "trace_data")

os.makedirs(TRACE_SAVE_DIR, exist_ok=True)
os.makedirs(TRACE_LOG_DIR, exist_ok=True)

# ------------------------- Function -------------------------
def run_trace_detection_and_save(frame, visualize: bool = True) -> tuple[str, list, list]:
    """
    Detect copper traces and save:
    - annotated image (optional)
    - trace measurement logs (JSON)

    Returns:
        annotated_path: str
        distances: list
        coord_logs: list
    """
    # 1️⃣ Detect traces
    contours, processed_img, distances, coord_logs = detect_traces(frame, visualize=visualize)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2️⃣ Save annotated image
    annotated_filename = f"pcb_traces_{timestamp}.png"
    annotated_path = os.path.join(TRACE_SAVE_DIR, annotated_filename)
    if visualize and processed_img is not None:
        cv2.imwrite(annotated_path, processed_img)
        print(f"[Trace Detection] Annotated image saved at: {annotated_path}")
    else:
        annotated_path = ""  # skip image saving if not visualizing

    # 3️⃣ Save measurement logs
    log_filename = f"trace_coords_{timestamp}.json"
    log_path = os.path.join(TRACE_LOG_DIR, log_filename)
    with open(log_path, "w") as f:
        json.dump(coord_logs, f, indent=2)
    print(f"[Trace Detection] Trace coordinates saved at: {log_path}")

    return annotated_path, distances, coord_logs
