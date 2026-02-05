import os
import cv2
import numpy as np
import json
from ultralytics import YOLO
from ui.theme import theme
from backend.run_trace_detection import run_trace_detection_and_save

CONFIG_PATH = "/home/jmc2/VisionBoard-Proj/config/grading_config.json"

# ---------------------- Defaults ----------------------
DEFAULT_DEFECT_COLORS = theme.themes["dark"]["defect_colors"]

DEFAULT_CONFIG = {
    "DEFECT_GRADE_THRESHOLDS": {"A": 5, "B": 20, "C": 50, "F": 100},
    "MIN_BOX_WIDTH": 6,
    "MIN_BOX_HEIGHT": 6,
    "CUSTOM_DEFECT_COLORS": DEFAULT_DEFECT_COLORS,
}

# ---------------------- Helpers ----------------------
def load_grading_config():
    if not os.path.exists(CONFIG_PATH):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG.copy()

    try:
        with open(CONFIG_PATH, "r") as f:
            user_cfg = json.load(f)
    except Exception:
        return DEFAULT_CONFIG.copy()

    cfg = DEFAULT_CONFIG.copy()
    cfg.update(user_cfg)

    # Merge nested dictionaries
    if "DEFECT_GRADE_THRESHOLDS" in user_cfg:
        cfg["DEFECT_GRADE_THRESHOLDS"] = user_cfg["DEFECT_GRADE_THRESHOLDS"]

    if "CUSTOM_DEFECT_COLORS" in user_cfg:
        cfg["CUSTOM_DEFECT_COLORS"] = {**DEFAULT_DEFECT_COLORS, **user_cfg["CUSTOM_DEFECT_COLORS"]}
    return cfg

def get_custom_color(label, custom_colors=None):
    if custom_colors is None:
        custom_colors = {}

    rgb = custom_colors.get(label)
    if rgb is None:
        rgb = theme.colors().get("defect_colors", {}).get(label, (255, 255, 255))
    return tuple(reversed(rgb))  # RGB -> BGR

def resolve_model_paths(paths):
    all_paths = []
    for p in paths:
        if os.path.isfile(p) and p.endswith(".pt"):
            all_paths.append(p)
        elif os.path.isdir(p):
            all_paths.extend([os.path.join(p, f) for f in os.listdir(p) if f.endswith(".pt")])
    return all_paths

# ---------------------- FinalGradingPipeline ----------------------
class FinalGradingPipeline:
    def __init__(self, model_a_paths: list, model_b_paths: list, model_configs: dict):
        self.model_configs = model_configs
        self.model_a_paths = resolve_model_paths(model_a_paths)
        self.model_b_paths = resolve_model_paths(model_b_paths)
        self.model_a = [YOLO(p) for p in self.model_a_paths]
        self.model_b = [YOLO(p) for p in self.model_b_paths]

    def run(self, image_path: str, annotated_dir: str):
        cfg = load_grading_config()
        MIN_BOX_W = int(cfg.get("MIN_BOX_WIDTH", 6))
        MIN_BOX_H = int(cfg.get("MIN_BOX_HEIGHT", 6))
        CUSTOM_DEFECT_COLORS = cfg.get("CUSTOM_DEFECT_COLORS", DEFAULT_DEFECT_COLORS)
        DEFECT_GRADE_THRESHOLDS = cfg["DEFECT_GRADE_THRESHOLDS"]

        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            return None
        H, W = img.shape[:2]
        img_area = H * W
        annotated_img = img.copy()

        # Brightness check
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        if mean_val < 2 or mean_val > 250:
            return None

        defects_per_model = {}

        # --------------------------- Helper to run YOLO models ---------------------------
        def run_models(models_list, all_boxes_accum, all_labels_accum, all_scores_accum):
            for model in models_list:
                model_name = getattr(model, "path", str(model))
                cfg_model = self.model_configs.get(model_name, {"conf": 0.25, "iou": 0.5, "max_det": 300})
                results = model.predict(
                    image_path,
                    conf=cfg_model["conf"],
                    iou=cfg_model["iou"],
                    max_det=cfg_model["max_det"],
                    save=False
                )
                model_defects = []
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    names = model.names
                    for i, box in enumerate(boxes.xyxy):
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(W, x2), min(H, y2)
                        bw, bh = x2 - x1, y2 - y1
                        area = bw * bh
                        area_ratio = area / img_area
                        if bw < MIN_BOX_W or bh < MIN_BOX_H or area < 50 or area_ratio < 0.0001:
                            continue
                        cls_id = int(boxes.cls[i])
                        label = names.get(cls_id, "unknown")
                        score = float(boxes.conf[i])
                        model_defects.append({"label": label, "bbox": (x1, y1, x2, y2)})
                        all_boxes_accum.append([x1, y1, x2, y2])
                        all_labels_accum.append(label)
                        all_scores_accum.append(score)
                defects_per_model[model_name] = model_defects

        # --------------------------- Stage 1: model_a ---------------------------
        all_boxes_a, all_labels_a, all_scores_a = [], [], []
        run_models(self.model_a, all_boxes_a, all_labels_a, all_scores_a)

        # Apply NMS for model_a
        final_boxes, final_labels = [], []
        for lbl in set(all_labels_a):
            inds = [i for i, l in enumerate(all_labels_a) if l == lbl]
            lbl_boxes = [all_boxes_a[i] for i in inds]
            lbl_scores = [all_scores_a[i] for i in inds]
            keep = self.non_max_suppression(lbl_boxes, lbl_scores, 0.3)
            for i in keep:
                final_boxes.append(lbl_boxes[i])
                final_labels.append(lbl)

        # --------------------------- Stage 2: model_b ---------------------------
        all_boxes_b, all_labels_b, all_scores_b = [], [], []
        run_models(self.model_b, all_boxes_b, all_labels_b, all_scores_b)

        # Merge model_b results
        all_boxes_combined = final_boxes + all_boxes_b
        all_labels_combined = final_labels + all_labels_b
        all_scores_combined = [1.0]*len(final_boxes) + all_scores_b  # assume max confidence for previous

        merged_final_boxes, merged_final_labels = [], []
        for lbl in set(all_labels_combined):
            inds = [i for i, l in enumerate(all_labels_combined) if l == lbl]
            lbl_boxes = [all_boxes_combined[i] for i in inds]
            lbl_scores = [all_scores_combined[i] for i in inds]
            keep = self.non_max_suppression(lbl_boxes, lbl_scores, 0.3)
            for i in keep:
                merged_final_boxes.append(lbl_boxes[i])
                merged_final_labels.append(lbl)

        # --------------------------- Stage 3: Trace detection ---------------------------
        trace_image_path, trace_distances, trace_coords = run_trace_detection_and_save(img, visualize=True)

        # Add trace violations as defects
        if trace_coords:
            for idx, trace_meta in enumerate(trace_coords):
                merged_final_labels.append(f"trace_violation")
                x1, y1 = int(trace_meta['start'][0]), int(trace_meta['start'][1])
                x2, y2 = int(trace_meta['end'][0]), int(trace_meta['end'][1])
                merged_final_boxes.append([x1, y1, x2, y2])

        # --------------------------- Draw all boxes ---------------------------
        defect_summary = {}
        merged_defects = []

        for box, label in zip(merged_final_boxes, merged_final_labels):
            x1, y1, x2, y2 = box
            color = get_custom_color(label, CUSTOM_DEFECT_COLORS)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            defect_summary[label] = defect_summary.get(label, 0) + 1
            merged_defects.append({"label": label, "bbox": (x1, y1, x2, y2)})

        # Save annotated image
        os.makedirs(annotated_dir, exist_ok=True)
        out_path = os.path.join(annotated_dir, os.path.basename(image_path))
        cv2.imwrite(out_path, annotated_img)

        # Compute grade
        total_defects = sum(defect_summary.values())
        grade = self.compute_grade(total_defects, DEFECT_GRADE_THRESHOLDS) if total_defects > 0 else "Pass"

        return {
            "defect_summary": defect_summary,
            "defects_per_model": defects_per_model,
            "merged_defects_final": merged_defects,
            "trace_image_path": trace_image_path,
            "trace_distances": trace_distances,
            "trace_coords": trace_coords,
            "grade": grade,
            "annotated_image_path": out_path
        }

    @staticmethod
    def non_max_suppression(boxes, scores=None, iou_threshold=0.3):
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

    @staticmethod
    def compute_grade(total_defects: int, thresholds: dict) -> str:
        if not thresholds:
            return "Pass"
        sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1])
        last_grade_label, last_threshold = sorted_thresholds[-1]
        for grade_label, max_defects in sorted_thresholds[:-1]:
            if total_defects <= max_defects:
                return grade_label
        return last_grade_label
