
import os
import cv2
import numpy as np
from ultralytics import YOLO
from ui.theme import theme

def get_custom_color(label):
    """
    Returns the BGR color for a defect label from the current theme.
    Defaults to white if the label is not found.
    """
    defect_colors = theme.colors().get("defect_colors", {})
    rgb = defect_colors.get(label, (255, 255, 255))
    return tuple(reversed(rgb))

class SingleModelPipeline:
    MIN_BOX_W = 4
    MIN_BOX_H = 4
    MIN_AREA = 50
    MIN_AREA_RATIO = 0.0001
    MAX_AREA_RATIO = 0.95

    def __init__(self, model_path: str, model_config: dict, enable_trace: bool = False):
        """
        model_path: folder or single .pt path
        model_config: dict with 'conf', 'iou', 'max_det'
        enable_trace: True only for Model 1
        """
        self.cfg = model_config
        self.model_paths = self._resolve_model_paths(model_path)
        self.models = [YOLO(p) for p in self.model_paths]
        self.enable_trace = enable_trace

    def _resolve_model_paths(self, path):
        if os.path.isfile(path) and path.endswith(".pt"):
            return [path]
        if os.path.isdir(path):
            return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pt")]
        return []

    def non_max_suppression(self, boxes, scores=None, iou_threshold=0.3):
        if len(boxes) == 0:
            return []
        boxes = np.array(boxes)
        scores = np.array(scores) if scores is not None else np.ones(len(boxes))
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
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
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def run(self, image_path: str, annotated_dir: str):
        os.makedirs(annotated_dir, exist_ok=True)

        img_raw = cv2.imread(image_path)
        if img_raw is None or img_raw.size == 0:
            return None
        img_raw = np.ascontiguousarray(img_raw)
        H, W = img_raw.shape[:2]
        img_area = H * W

        gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        if mean_val < 2 or mean_val > 250:
            return None

        all_boxes, all_labels, all_scores = [], [], []

        # ---------------- Collect boxes from all models ----------------
        for model in self.models:
            results = model.predict(
                image_path,
                conf=self.cfg["conf"],
                iou=self.cfg["iou"],
                max_det=self.cfg["max_det"],
                save=False
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
                if bw < self.MIN_BOX_W or bh < self.MIN_BOX_H or area < self.MIN_AREA:
                    continue
                area_ratio = area / img_area
                if area_ratio < self.MIN_AREA_RATIO or area_ratio > self.MAX_AREA_RATIO:
                    continue
                cls_id = int(boxes.cls[i])
                label = names.get(cls_id, "unknown")
                all_boxes.append([x1, y1, x2, y2])
                all_labels.append(label)
                all_scores.append(float(boxes.conf[i]))

        # ---------------- Apply NMS per label ----------------
        final_boxes, final_labels = [], []
        for lbl in set(all_labels):
            inds = [i for i, l in enumerate(all_labels) if l == lbl]
            lbl_boxes = [all_boxes[i] for i in inds]
            lbl_scores = [all_scores[i] for i in inds]
            keep = self.non_max_suppression(lbl_boxes, lbl_scores, 0.3)
            for i in keep:
                final_boxes.append(lbl_boxes[i])
                final_labels.append(lbl)

        # ---------------- Draw boxes ----------------
        img_annotated = img_raw.copy()
        defect_summary = {}
        defects_per_model = {"final": []}

        for box, label in zip(final_boxes, final_labels):
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            area = bw * bh
            area_ratio = area / img_area

            # Filter huge boxes or tiny boxes again
            if bw < self.MIN_BOX_W or bh < self.MIN_BOX_H or area < self.MIN_AREA:
                continue
            if area_ratio < self.MIN_AREA_RATIO or area_ratio > self.MAX_AREA_RATIO:
                continue

            color = get_custom_color(label)
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 2)
            defect_summary[label] = defect_summary.get(label, 0) + 1
            defects_per_model["final"].append({"label": label, "bbox": (x1, y1, x2, y2)})


        out_path = os.path.join(annotated_dir, os.path.basename(image_path))
        cv2.imwrite(out_path, img_annotated)

        # ---------------- PCB TRACE DETECTION ----------------
        if self.enable_trace:
            from backend.run_trace_detection import run_trace_detection_and_save
            trace_image_path, trace_distances, trace_coords = run_trace_detection_and_save(img_raw, visualize=True)
        else:
            trace_image_path, trace_distances, trace_coords = None, None, None

        return {
            "defect_summary": defect_summary,
            "defects_per_model": defects_per_model,
            "annotated_image_path": out_path,
            "trace_image_path": trace_image_path,
            "trace_coords": trace_coords,
            "trace_distances": trace_distances
        }
