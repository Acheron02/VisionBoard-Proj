import tkinter as tk
import cv2
from PIL import Image, ImageTk
import os
from datetime import datetime
import threading
import time
import numpy as np

from ui.roundedbutton import RoundedButton
from ui.backbtn import BackButton
from pages.resultpage import ResultsPage
from pages.errorpage import ErrorPage
from ultralytics import YOLO
from ui.actiondialog import ActionDialog
from ui.theme import theme
from backend.pcb_detector import PCBDetector
from backend.single_model_pipeline import SingleModelPipeline
from backend.final_grading_pipeline import FinalGradingPipeline

class CameraPage(tk.Frame):
    CAMERA_INDEX = 0
    VIDEO_WIDTH = 750
    VIDEO_HEIGHT = 420
    FINAL_GRADING_MODELS = ["Model 1", "Model 2"]

    def __init__(self, parent, show_page, monitor, model_name=None, grading=False, config=None):
        super().__init__(parent)
        self.show_page = show_page
        self.monitor = monitor
        self.model_name = model_name
        self.grading = grading
        self.config = config or {}

        self.monitor.pause_camera_check()
        self.running = True
        self._destroyed = False
        self.cap = None
        self.latest_frame = None
        self.latest_frame_raw = None
        self.dialog = None
        self.imgtk = None

        self.colors = theme.colors()
        self.configure(bg=self.colors["bg"])
        theme.subscribe(self.apply_theme)

        self._build_top_bar()
        self._build_main_content()

        # ---------------- PCB DETECTOR ----------------
        self.pcb_detector = PCBDetector()

        # ---------------- DIRECTORIES ----------------
        base_name = "grading" if grading else model_name
        self.captured_dir = os.path.join("captured_images", base_name)
        self.annotated_dir = os.path.join("annotated_images", base_name)
        os.makedirs(self.captured_dir, exist_ok=True)
        os.makedirs(self.annotated_dir, exist_ok=True)

        # ---------------- MODELS ----------------
        self.model_paths = {
            "Model 1": "/home/jmc2/VisionBoard-Proj/machine_learning_models/model_a",
            "Model 2": "/home/jmc2/VisionBoard-Proj/machine_learning_models/model_b",
            "Model 3": "/home/jmc2/VisionBoard-Proj/machine_learning_models/model_c",
        }
        self.model_configs = self.config.get("MODEL_DETECTION_CONFIGS", {
            "Model 1": {"conf": 0.5, "iou": 0.35, "max_det": 100},
            "Model 2": {"conf": 0.4, "iou": 0.50, "max_det": 200},
            "Model 3": {"conf": 0.3, "iou": 0.50, "max_det": 100},
        })

        # Only load YOLO if single-model detection
        self.models = {}
        if not grading and self.model_name:
            self.models[self.model_name] = YOLO(self.model_paths[self.model_name], task="detect")

        # ---------------- CAMERA ----------------
        self.init_camera()
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        self.after(33, self.display_frame)
        self.bind("<Destroy>", self._on_destroy)

    # ---------------- TOP BAR ----------------
    def _build_top_bar(self):
        self.top_frame = tk.Frame(self, bg=self.colors["bg"])
        self.top_frame.pack(fill="x", pady=(10,5))
        self.top_frame.columnconfigure(0, weight=0)
        self.top_frame.columnconfigure(1, weight=1)
        self.top_frame.columnconfigure(2, weight=0)

        self.left_frame = tk.Frame(self.top_frame, bg=self.colors["bg"], width=80)
        self.left_frame.grid(row=0, column=0, sticky="w", padx=(20,0))
        self.left_frame.grid_propagate(False)

        self.back_btn = BackButton(self.left_frame, command=self.on_back)
        self.back_btn.pack(anchor="w")
        self.back_btn.apply_theme(self.colors)

        self.center_frame = tk.Frame(self.top_frame, bg=self.colors["bg"])
        self.center_frame.grid(row=0, column=1, sticky="nsew")

        self.title_label = tk.Label(
            self.center_frame,
            text="",
            font=(theme.font_bold, theme.sizes["title"]),
            fg=self.colors["text"],
            bg=self.colors["bg"]
        )
        self.title_label.place(relx=0.5, rely=0.5, anchor="center")
        self.update_title()

        self.right_spacer = tk.Frame(self.top_frame, bg=self.colors["bg"], width=80)
        self.right_spacer.grid(row=0, column=2, sticky="e", padx=(0,20))
        self.right_spacer.grid_propagate(False)

    # ---------------- MAIN CONTENT ----------------
    def _build_main_content(self):
        self.container = tk.Frame(self, bg=self.colors["bg"])
        self.container.pack(fill="both", expand=True)

        self.video_holder = tk.Frame(
            self.container,
            width=self.VIDEO_WIDTH,
            height=self.VIDEO_HEIGHT,
            bg=self.colors["bg"]
        )
        self.video_holder.pack(pady=(5,10))
        self.video_holder.pack_propagate(False)

        self.video_frame = tk.Label(self.video_holder, bg=self.colors["bg"])
        self.video_frame.pack(fill="both", expand=True)

        self.buttons_frame = tk.Frame(self.container, bg=self.colors["bg"])
        self.buttons_frame.pack(pady=(5,15))

        self.capture_btn = RoundedButton(
            self.buttons_frame,
            text="Capture",
            width=200,
            height=80,
            radius=20,
            font=(theme.font_regular, theme.sizes["body"]),
            command=self.capture_image
        )
        self.capture_btn.pack()
        self.capture_btn.apply_theme(self.colors)

    # ---------------- TITLE ----------------
    def update_title(self):
        if self.grading:
            self.title_label.config(text="Final PCB Grading")
        elif self.model_name:
            self.title_label.config(text=f"Using {self.model_name}")
        else:
            self.title_label.config(text="Camera")

    # ================= CAMERA =================
    def init_camera(self):
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.CAMERA_INDEX, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Resolution set to: {actual_w}x{actual_h}")

    def camera_loop(self):
        while self.running and not self._destroyed:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.5)
                continue
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.5)
                continue
            self.latest_frame_raw = frame.copy()
            self.latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            time.sleep(1/30)

    def display_frame(self):
        if self.latest_frame is not None and not self._destroyed:
            img = Image.fromarray(self.latest_frame).resize(
                (self.VIDEO_WIDTH, self.VIDEO_HEIGHT)
            )
            self.imgtk = ImageTk.PhotoImage(img)
            self.video_frame.configure(image=self.imgtk)
        if self.running:
            self.after(33, self.display_frame)

    # ================= CAPTURE =================
    def capture_image(self):
        if self.latest_frame_raw is None:
            return

        detection = self.pcb_detector.detect(self.latest_frame_raw)
        if detection is None or not detection.detected:
            self.show_no_pcb_dialog()
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = os.path.join(self.captured_dir, f"{ts}.png")
        cv2.imwrite(raw_path, self.latest_frame_raw)

        if self.grading:
            self.run_final_grading(raw_path)
        else:
            self.run_single_model(raw_path)

    # ================= SINGLE MODEL =================
    def run_single_model(self, image_path):
        model_cfg = self.model_configs[self.model_name]
        pipeline = SingleModelPipeline(
            model_path=self.model_paths[self.model_name],
            model_config=model_cfg,
            enable_trace=(self.model_name.lower() == "model 1")
        )
        result = pipeline.run(
            image_path=image_path,
            annotated_dir=self.annotated_dir,
        )

        if result is None:
            self.show_no_pcb_dialog()
            return

        defect_summary = result.get("defect_summary", {})
        self.cleanup()
        self.show_page(
            ResultsPage,
            monitor=self.monitor,
            original_capture_path=result["annotated_image_path"],
            result_image_path=result["annotated_image_path"],
            model_name=self.model_name,
            defect_summary=result.get("defect_summary"),
            defects_per_model=result.get("defects_per_model"),
            grade=None
        )

    # ================= FINAL GRADING =================
    def run_final_grading(self, image_path):
        pipeline = FinalGradingPipeline(
            model_a_paths=[self.model_paths["Model 1"]],
            model_b_paths=[self.model_paths["Model 2"]],
            model_configs=self.model_configs
        )

        result = pipeline.run(image_path=image_path, annotated_dir=self.annotated_dir)
        if result is None:
            self.show_no_pcb_dialog()
            return

        defect_summary = result.get("defect_summary", {})
        self.cleanup()
        self.show_page(
            ResultsPage,
            monitor=self.monitor,
            original_capture_path=result["annotated_image_path"],
            model_name="Final PCB Grading",
            defect_summary=result.get("defect_summary"),
            defects_per_model=result.get("defects_per_model"),
            grade=result.get("grade", "Pass")
        )

    # ================= DIALOGS =================
    def show_no_pcb_dialog(self):
        if self.dialog:
            self.dialog.destroy()
        self.dialog = ActionDialog(
            self,
            title="No PCB Detected",
            message="No PCB was detected in the captured image. Please place a PCB within the camera view.",
            confirm_text="Retake",
            confirm_command=lambda: None,
            cancel_text="",
            toggle_button=getattr(self.master, "toggle", None)
        )

    # ================= NAVIGATION =================
    def on_back(self):
        self.cleanup()
        from pages.choosemodel import ChooseModel
        self.show_page(ChooseModel, monitor=self.monitor, theme=theme)

    # ================= CLEANUP =================
    def cleanup(self):
        self.running = False
        self._destroyed = True
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.dialog:
            self.dialog.destroy()
            self.dialog = None
        self.monitor.resume_camera_check()

    def _on_destroy(self, *_):
        self.cleanup()
        if self.apply_theme in theme.subscribers:
            theme.subscribers.remove(self.apply_theme)

    # ================= THEME =================
    def apply_theme(self, colors):
        try:
            self.colors = colors
            self.configure(bg=colors["bg"])
            self.top_frame.configure(bg=colors["bg"])
            self.left_frame.configure(bg=colors["bg"])
            self.center_frame.configure(bg=colors["bg"])
            self.right_spacer.configure(bg=colors["bg"])
            self.title_label.configure(bg=colors["bg"], fg=colors["text"])
            self.container.configure(bg=colors["bg"])
            self.video_holder.configure(bg=colors["bg"])
            self.buttons_frame.configure(bg=colors["bg"])
            self.capture_btn.apply_theme(colors)
            self.back_btn.apply_theme(colors)
            self.video_frame.configure(bg=colors["bg"])
        except tk.TclError:
            pass
