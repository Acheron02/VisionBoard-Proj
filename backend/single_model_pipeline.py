import os
import cv2
import numpy as np
from ultralytics import YOLO
from ui.theme import theme
from backend.final_grading_pipeline import load_grading_config, get_color

class SingleModelPipeline:
    MIN_BOX_W = 4
    MIN_BOX_H = 4
    MIN_AREA = 50
    MIN_AREA_RATIO = 0.0001
    MAX_AREA_RATIO = 0.95

    def __init__(self, model_path: str, model_config: dict, enable_trace: bool = False):
        """
        model_path: folder or .pt file
        model_config: {conf, iou, max_det}
        enable_trace: True only if trace detection is needed
        """
        self.cfg = model_config
        self.model_paths = self._resolve_model_paths(model_path)
        self.models = [YOLO(p) for p in self.model_paths]
        self.enable_trace = enable_trace

    # ---------------- Utils ----------------
    @staticmethod
    def _resolve_model_paths(path):
        if os.path.isfile(path) and path.endswith(".pt"):
            return [path]
        if os.path.isdir(path):
            return [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith(".pt")
            ]
        return []

    @staticmethod
    def non_max_suppression(boxes, scores, iou_thresh=0.3):
        if not boxes:
            return []

        boxes = np.array(boxes)
        scores = np.array(scores)

        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            order = order[np.where(iou <= iou_thresh)[0] + 1]

        return keep

    # ---------------- Main ----------------
    def run(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            return None

        H, W = img.shape[:2]
        img_area = H * W

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not (2 < np.mean(gray) < 250):
            return None

        all_boxes, all_labels, all_scores = [], [], []

        cfg = load_grading_config()

        # -------- YOLO inference --------
        for model in self.models:
            results = model.predict(
                image_path,
                conf=self.cfg["conf"],
                iou=self.cfg["iou"],
                max_det=self.cfg["max_det"],
                save=False,
            )

            if not results or results[0].boxes is None:
                continue

            boxes = results[0].boxes
            names = model.names

            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                bw, bh = x2 - x1, y2 - y1
                area = bw * bh

                if (
                    bw < self.MIN_BOX_W
                    or bh < self.MIN_BOX_H
                    or area < self.MIN_AREA
                ):
                    continue

                ratio = area / img_area
                if not (self.MIN_AREA_RATIO <= ratio <= self.MAX_AREA_RATIO):
                    continue

                label = names.get(int(boxes.cls[i]), "unknown")
                score = float(boxes.conf[i])

                all_boxes.append([x1, y1, x2, y2])
                all_labels.append(label)
                all_scores.append(score)

        # -------- NMS per label --------
        final_boxes, final_labels = [], []
        for lbl in set(all_labels):
            idxs = [i for i, l in enumerate(all_labels) if l == lbl]
            lbl_boxes = [all_boxes[i] for i in idxs]
            lbl_scores = [all_scores[i] for i in idxs]

            keep = self.non_max_suppression(lbl_boxes, lbl_scores)
            for i in keep:
                final_boxes.append(lbl_boxes[i])
                final_labels.append(lbl)

        # -------- Build summary --------
        defect_summary = {}
        defects = []

        for box, label in zip(final_boxes, final_labels):
            defect_summary[label] = defect_summary.get(label, 0) + 1
            defects.append({"label": label, "bbox": tuple(box)})

        # -------- Trace detection (DATA ONLY) --------
        trace_data = None
        if self.enable_trace:
            from backend.run_trace_detection import run_trace_detection_and_save
            trace_result = run_trace_detection_and_save(img, visualize=False)
            self.trace_coords = trace_result[2] if trace_result and len(trace_result) > 2 else []
            trace_data = {
                "trace_distances": trace_result[1] if trace_result and len(trace_result) > 1 else None,
                "trace_coords": trace_result[2] if trace_result and len(trace_result) > 2 else [],
            }

        # -------- Build annotated image (ALWAYS) --------
        annotated_img = img.copy()
        for box, label in zip(final_boxes, final_labels):
            x1, y1, x2, y2 = box
            color = get_color(label, cfg["CUSTOM_DEFECT_COLORS"])  # consistent with grading
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

        annotated_dir = os.path.join("annotated_images", "single_model")
        os.makedirs(annotated_dir, exist_ok=True)
        basename = os.path.basename(image_path)
        annotated_path = os.path.join(annotated_dir, basename)
        cv2.imwrite(annotated_path, annotated_img)

        return {
            "defect_summary": defect_summary,
            "defects": defects,
            "trace": trace_data,
            "annotated_image_path": annotated_path,
        }

