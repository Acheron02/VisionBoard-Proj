import cv2
import numpy as np
from backend.pcb_detector import PCBDetector
from backend.measure_trace_dist import measure_parallel_trace_distances

def detect_traces(frame: np.ndarray, visualize: bool = True, angle_threshold: float = 10):
    """
    Detect and highlight copper traces on a PCB.
    Measures distances between parallel traces.
    """
    if frame is None or frame.size == 0:
        return [], frame, [], []

    output_img = frame.copy()

    # 1️⃣ Detect PCB region
    pcb_detector = PCBDetector()
    result = pcb_detector.detect(frame)
    if not result.detected or result.bbox is None:
        print("[Trace Detection] PCB not detected.")
        return [], output_img, [], []

    x, y, w, h = result.bbox
    pcb_roi = frame[y:y+h, x:x+w]

    # 2️⃣ Grayscale + CLAHE
    gray = cv2.cvtColor(pcb_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 3️⃣ Decide black/white traces
    mean_val = np.mean(gray)
    if mean_val > 127:
        morph_image = cv2.morphologyEx(
            gray, cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        )
        _, trace_mask = cv2.threshold(morph_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        morph_image = cv2.morphologyEx(
            gray, cv2.MORPH_TOPHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        )
        _, trace_mask = cv2.threshold(morph_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4️⃣ Morphological cleaning
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    clean_mask = cv2.morphologyEx(trace_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    # 5️⃣ Find contours
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if 10 < cv2.contourArea(cnt) < (w*h*0.6)]

    # 6️⃣ Measure distances between parallel traces
    distances, dist_img, coord_logs = measure_parallel_trace_distances(
        filtered_contours,
        pcb_roi.shape,
        pcb_image=output_img,
        offset=(x, y)
    )

    # Overlay visualization
    if visualize and dist_img is not None:
        output_img = cv2.addWeighted(output_img, 0.7, dist_img, 0.3, 0)

    return filtered_contours, output_img, distances, coord_logs
