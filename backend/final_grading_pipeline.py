import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
from ui.theme import theme
from backend.run_trace_detection import run_trace_detection_and_save

CONFIG_PATH = "/home/jmc2/VisionBoard-Proj/config/grading_config.json"

# ---------------- Config ----------------
DEFAULT_CONFIG = {
    "DEFECT_GRADE_THRESHOLDS": {"A": 5, "B": 20, "C": 50, "F": 100},
    "MIN_BOX_WIDTH": 6,
    "MIN_BOX_HEIGHT": 6,
    "CUSTOM_DEFECT_COLORS": theme.themes["dark"]["defect_colors"],
}


def load_grading_config():
    if not os.path.exists(CONFIG_PATH):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG.copy()

    with open(CONFIG_PATH, "r") as f:
        user_cfg = json.load(f)

    cfg = DEFAULT_CONFIG.copy()
    cfg.update(user_cfg)
    cfg["CUSTOM_DEFECT_COLORS"] = {
        **DEFAULT_CONFIG["CUSTOM_DEFECT_COLORS"],
        **user_cfg.get("CUSTOM_DEFECT_COLORS", {}),
    }
    return cfg


def get_color(label, custom_colors):
    rgb = custom_colors.get(label.lower(), (255, 255, 255))
    return tuple(reversed(rgb))


# ---------------- Pipeline ----------------
class FinalGradingPipeline:
    def __init__(self, model_folders: list, model_configs: dict):
        """
        model_folders: list of folders, each containing .pt files
        model_configs: dict mapping model folder or file to config dict
        """
        self.model_configs = model_configs
        self.models = []
        # Load all .pt files in all model folders
        for folder in model_folders:
            if os.path.isdir(folder):
                pt_files = [
                    os.path.join(folder, f)
                    for f in os.listdir(folder)
                    if f.endswith(".pt")
                ]
                for pt in pt_files:
                    model = YOLO(pt)
                    model._path = pt  # store the path ourselves
                    self.models.append(model)
            elif os.path.isfile(folder) and folder.endswith(".pt"):
                model = YOLO(folder)
                model._path = folder
                self.models.append(model)

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

    @staticmethod
    def compute_grade(total, thresholds):
        for grade, limit in sorted(thresholds.items(), key=lambda x: x[1]):
            if total <= limit:
                return grade
        return list(thresholds.keys())[-1]

    # ---------------- Run ----------------
    def run(self, image_path, annotated_dir):
        cfg = load_grading_config()
        img = cv2.imread(image_path)
        if img is None:
            return None

        annotated = img.copy()
        H, W = img.shape[:2]
        img_area = H * W

        all_boxes, all_labels, all_scores = [], [], []

        # -------- YOLO inference for all models --------
        for model in self.models:
            cfg_m = self.model_configs.get(getattr(model, "_path", None), {"conf":0.25, "iou":0.5, "max_det":300})
            results = model.predict(
                image_path,
                conf=cfg_m["conf"],
                iou=cfg_m["iou"],
                max_det=cfg_m["max_det"],
                save=False,
            )
            if not results or results[0].boxes is None:
                continue

            boxes = results[0].boxes
            names = model.names
            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                bw, bh = x2 - x1, y2 - y1
                area = bw * bh

                if (
                    bw < cfg["MIN_BOX_WIDTH"]
                    or bh < cfg["MIN_BOX_HEIGHT"]
                    or area < 50
                    or area / img_area < 0.0001
                ):
                    continue

                all_boxes.append([x1, y1, x2, y2])
                all_labels.append(names.get(int(boxes.cls[i]), "unknown"))
                all_scores.append(float(boxes.conf[i]))

        # -------- NMS --------
        final_boxes, final_labels = [], []
        for lbl in set(all_labels):
            idxs = [i for i, l in enumerate(all_labels) if l == lbl]
            keep = self.non_max_suppression([all_boxes[i] for i in idxs], [all_scores[i] for i in idxs])
            for i in keep:
                final_boxes.append(all_boxes[idxs[i]])
                final_labels.append(lbl)

        # -------- Trace detection --------
        trace_img, trace_dists, trace_coords = run_trace_detection_and_save(img, visualize=True)

        # -------- Draw + Count --------
        defect_summary = {}
        for box, label in zip(final_boxes, final_labels):
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), get_color(label, cfg["CUSTOM_DEFECT_COLORS"]), 2)
            defect_summary[label] = defect_summary.get(label, 0) + 1

        os.makedirs(annotated_dir, exist_ok=True)
        out_path = os.path.join(annotated_dir, os.path.basename(image_path))
        cv2.imwrite(out_path, annotated)

        total = sum(defect_summary.values())
        grade = self.compute_grade(total, cfg["DEFECT_GRADE_THRESHOLDS"])

        return {
            "defect_summary": defect_summary,
            "trace_image_path": trace_img,
            "trace_distances": trace_dists,
            "trace_coords": trace_coords,
            "grade": grade,
            "annotated_image_path": out_path,
        }
