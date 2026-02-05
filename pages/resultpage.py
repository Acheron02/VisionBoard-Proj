
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import os
import cups
import subprocess
import qrcode

from ui.backbtn import BackButton
from ui.downloadbtn import DownloadButton
from ui.retakebtn import RetakeButton
from ui.actiondialog import ActionDialog
from ui.theme import theme
from ui.roundedbutton import RoundedButton
from ui.printbtn import PrintButton
from pages.errorpage import ErrorPage

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

from backend.generateURL import generate_download_url  # Your URL generator

# ---------------- LABELS ----------------
DEFECT_FULL_LABELS = {
    "short": "Short Circuit",
    "open": "Open Circuit",
    "90": "90° Angle Traces",
    "ps": "Poor Solder",
    "sb": "Solder Bridges",
    "mc": "Missing Components",
    "resistor": "Resistor",
    "capacitor": "Capacitor",
}

GRADE_FEEDBACK = {
    "Pass": "Good Job, your PCB is clean and has little to no detected defects! Keep up the good work!",
    "Minor Severity": (
        "Some defects were detected. Improve your soldering technique, use the correct temperature and solder, and adjust photoresist baking for cleaner PCB etching."
    ),
    "Major Severity": (
        "Critical defects were found. Review your design and fabrication process, check for shorts or missing components, and correct issues before using this PCB."
    )
}


def get_custom_color(label, custom_colors=None):
    if custom_colors is None:
        custom_colors = {}

    rgb = custom_colors.get(label)

    if rgb is None:
        rgb = theme.colors().get("defect_colors", {}).get(label, (255, 255, 255))

    return rgb 

def get_full_label(label):
    return DEFECT_FULL_LABELS.get(label, label)

# ---------------- RESULTS PAGE ----------------
class ResultsPage(tk.Frame):
    IMAGE_MAX_WIDTH = 800
    IMAGE_MAX_HEIGHT = 420
    MIN_BOX_W = 4
    MIN_BOX_H = 4
    MIN_AREA = 50
    MAX_AREA_RATIO = 0.95

    def __init__(self, parent, show_page, monitor, original_capture_path, model_name,
                 defect_summary=None, defects_per_model=None, grade=None, result_image_path=None):
        """
        defects_per_model: dict of {model_name: [defect_dict,...]} where defect_dict has keys:
            'label': defect label
            'bbox': (x1, y1, x2, y2)
        """
        super().__init__(parent)
        self.show_page = show_page
        self.monitor = monitor
        self.model_name = model_name
        self.grade = grade
        self.enable_trace = (model_name == "Model 1" or model_name == "Final PCB Grading")

        # Save original capture separately for Retake button
        self.original_capture_path = original_capture_path

        # Merge defects into one summary for legend
        self.defect_summary = defect_summary or {}
        self.defects_per_model = defects_per_model or {}

        # Generate a single merged result image
        self.result_image_path = os.path.join(
            os.path.dirname(original_capture_path),
            "merged_result.png"
        )

        self.LEGEND_COLUMN_WIDTH = 300

        # ---------------- UI state ----------------
        self.dialog = None
        self.qr_popup = None
        self.qr_content_frame = None
        self.qr_title_label = None
        self.qr_desc_label = None
        self.qr_label = None
        self.imgtk = None
        self.original_image = None

        self.colors = theme.colors()
        self.configure(bg=self.colors["bg"])

        # ---------------- Top Bar ----------------
        self._build_top_bar()

        # ---------------- Container ----------------
        self.container = tk.Frame(self, bg=self.colors["bg"])
        self.container.pack(fill="both", expand=True, padx=10, pady=10)
        self.container.grid_rowconfigure(0, weight=0)
        self.container.grid_rowconfigure(1, weight=0)
        self.container.grid_rowconfigure(2, weight=0)
        self.container.grid_columnconfigure(0, weight=1)

        # ---------------- Image + Legend ----------------
        self.top_content_frame = tk.Frame(self.container, bg=self.colors["bg"], height=self.IMAGE_MAX_HEIGHT)
        self.top_content_frame.grid(row=0, column=0, sticky="nsew", padx=30)
        self.top_content_frame.grid_propagate(False)

        # Grid config: image takes ~80%, legend ~20%
        self.top_content_frame.grid_rowconfigure(0, weight=1)
        self.top_content_frame.grid_columnconfigure(0, weight=1)  # image column
        self.top_content_frame.grid_columnconfigure(1, minsize=30) 
        self.top_content_frame.grid_columnconfigure(2, weight=0)  # legend column

        # ---------------- Image Frame ----------------
        self.img_frame = tk.Frame(
            self.top_content_frame,
            bg="black",
            width=self.IMAGE_MAX_WIDTH,
            height=self.IMAGE_MAX_HEIGHT
        )
        self.img_frame.grid(row=0, column=0, sticky="nsew")
        self.img_frame.grid_propagate(False)

        self.img_label = tk.Label(self.img_frame, bg="black")

        # IMPORTANT: bind resize to img_frame
        self.img_frame.bind("<Configure>", self.resize_image)


        # ---------------- Legend Column ----------------
        self.legend_column = tk.Frame(self.top_content_frame, bg=self.colors["bg"], width=self.LEGEND_COLUMN_WIDTH)
        self.legend_column.grid(row=0, column=2, sticky="nsew")
        self.legend_column.grid_propagate(False)  # prevents it from resizing to fit content

        # Configure rows: grade, legend, feedback
        self.legend_column.grid_rowconfigure(0, weight=0)  # grade
        self.legend_column.grid_rowconfigure(1, weight=1)  # legend_frame
        self.legend_column.grid_rowconfigure(2, weight=0)  # feedback
        self.legend_column.grid_columnconfigure(0, weight=1)

        # Grade label
        if self.grade is not None:
            self.grade_label = tk.Label(
                self.legend_column,
                text=f"PCB Grade: {self.grade}",
                font=(theme.font_bold, theme.sizes["title2"]),
                fg=self.colors["text"],
                bg=self.colors["bg"],
                justify="center",
                anchor="n"
            )
            self.grade_label.grid(row=0, column=0, sticky="n", pady=(10,0))

        # Legend frame inside legend column
        self.legend_frame = tk.Frame(self.legend_column, bg=self.colors["bg"])
        self.legend_frame.grid(row=1, column=0, sticky="n", pady=(0,10))
        self._generate_merged_image()
        self._build_legend()
        self._render_defect_summary()


        # ---------------- Buttons ----------------
        self.buttons_frame = tk.Frame(self.container, bg=self.colors["bg"])
        self.buttons_frame.grid(row=1, column=0, pady=10)
        self.download_btn = DownloadButton(self.buttons_frame, command=self.show_qr_dialog, width=180, height=70, radius=16)
        self.retake_btn = RetakeButton(self.buttons_frame, command=self.confirm_retake_image, width=180, height=70, radius=16)
        self.print_btn = PrintButton(self.buttons_frame, command=self.print_result, width=180, height=70, radius=16)
        for btn in [self.download_btn, self.retake_btn, self.print_btn]:
            btn.pack(side="left", padx=10)

        # Theme subscription
        theme.subscribe(self.apply_theme)
        theme.subscribe(self.download_btn.apply_theme)
        theme.subscribe(self.retake_btn.apply_theme)
        theme.subscribe(self.print_btn.apply_theme)

        # ---------------- Printer Detection ----------------
        self._printer_available = self._detect_printer()
        self.print_btn.set_disabled(not self._printer_available)

        # Load image
        if not os.path.exists(self.result_image_path):
            self._show_fatal_error("Result image not found.")
        else:
            self.original_image = Image.open(self.result_image_path)
            self.top_content_frame.bind("<Configure>", self.resize_image)

        # ---------------- System monitor subscription ----------------
        self.monitor.subscribe(self.on_system_update)
        self.on_system_update(self.monitor.problems)
        self.bind("<Destroy>", self._on_destroy)

    # ---------------- Merge defects into single image ----------------
    def _generate_merged_image(self):
        import cv2
        import numpy as np
        from PIL import ImageDraw, ImageFont
        from backend.run_trace_detection import run_trace_detection_and_save

        # Load original PCB image
        img_bgr = cv2.imread(self.original_capture_path)
        if img_bgr is None:
            print(f"[Error] Failed to load image: {self.original_capture_path}")
            return

        # Convert to RGB for PIL
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Font for labels
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        except IOError:
            font = ImageFont.load_default()

        img_area = pil_img.width * pil_img.height

        for model_defects in self.defects_per_model.values():
            for defect in model_defects:
                if not isinstance(defect, dict) or "bbox" not in defect:
                    continue
                x1, y1, x2, y2 = defect["bbox"]
                bw, bh = x2 - x1, y2 - y1
                area = bw * bh
                area_ratio = area / img_area
                if bw < self.MIN_BOX_W or bh < self.MIN_BOX_H or area < self.MIN_AREA or area_ratio > self.MAX_AREA_RATIO:
                    continue  # skip invalid/huge boxes
                color = get_custom_color(defect["label"])
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # ---------------- Draw PCB trace clearance ----------------
        if self.enable_trace:
            try:
                # Run detection on BGR image
                contours, processed_img, trace_coords = run_trace_detection_and_save(img_bgr, visualize=False)

                trace_color = get_custom_color("trace_violation")
                box_size = 6  # bigger rectangles for visibility

                # Store coords for later use in resize_image
                self.trace_coords = trace_coords

                for idx, coord in enumerate(trace_coords):
                    start = tuple(coord["start"])
                    end   = tuple(coord["end"])

                    # Draw rectangles at start/end
                    draw.rectangle([start[0]-box_size, start[1]-box_size,
                                    start[0]+box_size, start[1]+box_size],
                                outline=trace_color, width=2)
                    draw.rectangle([end[0]-box_size, end[1]-box_size,
                                    end[0]+box_size, end[1]+box_size],
                                outline=trace_color, width=2)

                    # Draw connecting line
                    draw.line([start, end], fill=trace_color, width=4)

                    # Draw label T1, T2, ... with small white background for readability
                    label_text = f"T{idx+1}"
                    text_offset = (8, -8)
                    x, y = start[0]+text_offset[0], start[1]+text_offset[1]

                    # Draw white rectangle behind text
                    text_w, text_h = draw.textsize(label_text, font=font)
                    draw.rectangle([x-1, y-1, x+text_w+1, y+text_h+1], fill=(255,255,255))
                    draw.text((x, y), label_text, fill=trace_color, font=font)

            except Exception as e:
                print(f"[Trace Annotation] Failed: {e}")
        else:
            self.trace_coords = []

        # Save merged image
        os.makedirs(os.path.dirname(self.result_image_path), exist_ok=True)
        pil_img.save(self.result_image_path)

        self.original_image = pil_img  # used by resize_image()

        # Merge trace violations into defect summary
        if self.enable_trace:
            self.defect_summary["trace_violation"] = len(self.trace_coords)
        self._defect_summary_initialized = True

        # Refresh legend and feedback
        self._render_defect_summary()

    # ---------------- Top Bar ----------------
    def _build_top_bar(self):
        self.top_frame = tk.Frame(self, bg=self.colors["bg"])
        self.top_frame.pack(fill="x", pady=(10,5))
        self.top_frame.columnconfigure(0, weight=0)
        self.top_frame.columnconfigure(1, weight=1)
        self.top_frame.columnconfigure(2, weight=0)

        # Back button
        self.left_frame = tk.Frame(self.top_frame, bg=self.colors["bg"], width=80)
        self.left_frame.grid(row=0, column=0, sticky="w", padx=(20,0))
        self.left_frame.grid_propagate(False)
        self.back_btn = BackButton(self.left_frame, command=self.confirm_back_to_welcome)
        self.back_btn.pack(anchor="w")
        self.back_btn.apply_theme(self.colors)

        # Title
        self.center_frame = tk.Frame(self.top_frame, bg=self.colors["bg"])
        self.center_frame.grid(row=0, column=1, sticky="nsew")
        self.title_label = tk.Label(self.center_frame, text="Analysis Results",
                                    fg=self.colors["text"], bg=self.colors["bg"],
                                    font=(theme.font_bold, theme.sizes["title"]))
        self.title_label.place(relx=0.5, rely=0.5, anchor="center")

        # Spacer
        self.right_spacer = tk.Frame(self.top_frame, bg=self.colors["bg"], width=80)
        self.right_spacer.grid(row=0, column=2, sticky="e", padx=(0,20))
        self.right_spacer.grid_propagate(False)

    # ---------------- Theme ----------------
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
            self.top_content_frame.configure(bg=colors["bg"])
            self.img_frame.configure(bg="black")
            self.legend_frame.configure(bg=colors["bg"])
            self.buttons_frame.configure(bg=colors["bg"])

            if hasattr(self, "legend_column"):
                self.legend_column.configure(bg=colors["bg"])

            if hasattr(self, "grade_label"):
                self.grade_label.configure(
                    bg=colors["bg"],
                    fg=colors["text"]
                )

            if hasattr(self, "feedback_label") and self.feedback_label:
                self.feedback_label.configure(
                    bg=colors["bg"],
                    fg=colors["text"]
                )

            if hasattr(self, "spacer"):
                self.spacer.configure(bg=colors["bg"])

            # Update legend rows and labels
            for row_frame in self.legend_frame.winfo_children():
                row_frame.configure(bg=colors["bg"])
                for widget in row_frame.winfo_children():
                    if isinstance(widget, tk.Label):
                        widget.configure(bg=colors["bg"], fg=colors["text"])
                    elif isinstance(widget, tk.Canvas):
                        widget.configure(bg=colors["bg"])
        except tk.TclError:
            pass

    # ---------------- Legend ----------------
    def _build_legend(self):
        self.legend_buttons = []
        # Include trace violations if present
        summary_items = dict(self.defect_summary)
        label_mapping = {}  # maps display name to internal key
        for key in list(summary_items.keys()):
            if key == "trace_violation":
                label_mapping["Trace Violation"] = key
            else:
                label_mapping[get_full_label(key)] = key

        for display_label, internal_key in label_mapping.items():
            count = summary_items[internal_key]
            color = get_custom_color(internal_key)  # use internal key for color

            frame = tk.Frame(self.legend_frame, bg=self.colors["bg"])
            frame.pack(anchor="w", pady=6, fill="x")

            hex_color = f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"

            canvas = tk.Canvas(
                frame,
                width=50,
                height=30,
                highlightthickness=0,
                bg=self.colors["bg"]
            )
            canvas.pack(side="left", padx=(0,10))

            # Rounded rectangle
            radius = 5
            canvas.create_round_rect = lambda x1, y1, x2, y2, r, **kw: (
                canvas.create_polygon(
                    x1+r, y1, x2-r, y1, x2, y1+r, x2, y2-r,
                    x2-r, y2, x1+r, y2, x1, y2-r, x1, y1+r,
                    smooth=True, **kw
                )
            )

            canvas.create_round_rect(
                2, 2, 48, 28, radius,
                fill=hex_color,
                outline="black" if color == (255,255,255) else ""
            )

            lbl = tk.Label(
                frame,
                text=f"{display_label} ({count})",
                font=(theme.font_bold, 20),
                bg=self.colors["bg"],
                fg=self.colors["text"],
                anchor="w",
                justify="left",
                wraplength=self.LEGEND_COLUMN_WIDTH
            )
            lbl.pack(side="left", fill="x")


    # ---------------- Defect Summary ----------------
    def _render_defect_summary(self):
        # Remove old feedback label if exists
        if hasattr(self, "feedback_label") and self.feedback_label:
            self.feedback_label.destroy()
            self.feedback_label = None

        # ---------------- Merge trace violations into defect_summary ----------------
        if hasattr(self, "trace_coords") and self.trace_coords:
            self.defect_summary["trace_violation"] = len(self.trace_coords)

        if self.grade:
            feedback_text = GRADE_FEEDBACK.get(self.grade, "")
            if feedback_text:
                # Feedback like grade label, centered
                self.feedback_label = tk.Label(
                    self.legend_column,
                    text=feedback_text,
                    font=(theme.font_regular, theme.sizes["feedback"]),
                    fg=self.colors["text"],
                    bg=self.colors["bg"],
                    justify="left",      # left-align for readability
                    anchor="n",          # anchor top
                    wraplength=self.LEGEND_COLUMN_WIDTH  # avoid overflow
                )
                # Place below grade label, with decent vertical gap
                self.feedback_label.grid(
                    row=2, column=0, sticky="n", pady=(0,10)
                )

    # ---------------- Download / QR Dialog ----------------
    def show_qr_dialog(self):
        try:
            # ---------------- Generate PDF ----------------
            # PDF filename
            pdf_filename = os.path.splitext(os.path.basename(self.original_capture_path))[0] + ".pdf"

            # Save PDF in server-expected folder: annotated_images/<model_name>
            base_annotated_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "annotated_images")
            model_folder = os.path.join(base_annotated_folder, self.model_name)
            os.makedirs(model_folder, exist_ok=True)

            pdf_path = os.path.join(model_folder, pdf_filename)

            # Generate PDF if it doesn't exist
            if not os.path.exists(pdf_path):
                if FPDF is None:
                    ActionDialog(self, title="Missing Module",
                                message="fpdf is not installed.\nRun `pip install fpdf`.",
                                confirm_text="OK")
                    return

                # Always regenerate merged image to ensure trace violations are drawn
                self._generate_merged_image()

                # Open merged image
                img_rgb_path = self.result_image_path
                with Image.open(img_rgb_path) as im:
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                        tmp_path = os.path.join(model_folder, "tmp_merged_result.jpg")
                        im.save(tmp_path, format="JPEG")
                        img_rgb_path = tmp_path

                    # Draw trace violations again to be sure
                    draw = ImageDraw.Draw(im)
                    trace_color = get_custom_color("trace_violation")
                    box_size = 6
                    for idx, coord in enumerate(getattr(self, "trace_coords", [])):
                        start = tuple(coord["start"])
                        end = tuple(coord["end"])

                        draw.rectangle([start[0]-box_size, start[1]-box_size, start[0]+box_size, start[1]+box_size],
                                    outline=trace_color, width=2)
                        draw.rectangle([end[0]-box_size, end[1]-box_size, end[0]+box_size, end[1]+box_size],
                                    outline=trace_color, width=2)
                        draw.line([start, end], fill=trace_color, width=4)
                        draw.text((start[0]+8, start[1]-8), f"T{idx+1}", fill=trace_color)

                    # Save temporary image for PDF
                    tmp_path = os.path.join(model_folder, "tmp_for_pdf.jpg")
                    im.save(tmp_path, format="JPEG")
                    img_rgb_path = tmp_path

                # ---------------- Create PDF ----------------
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 30)
                pdf.cell(0, 10, "Analysis Results", ln=True, align="C")
                pdf.ln(10)

                # Fit image to PDF width
                pdf_w = pdf.w - 20
                img = Image.open(img_rgb_path)
                iw, ih = img.size
                ratio = pdf_w / iw
                pdf.image(img_rgb_path, x=10, y=None, w=pdf_w, h=ih*ratio)
                pdf.ln(10)

                # ---------------- Draw legend ----------------
                pdf.set_font("Arial", "B", 18)
                pdf.cell(0, 8, "Legend:", ln=True)
                pdf.ln(5)

                rect_size = 5
                spacing_x = 5
                spacing_y = 3

                legend_items = dict(self.defect_summary)
                for label, count in legend_items.items():
                    full_label = "Trace Violation" if label == "trace_violation" else get_full_label(label)
                    color = get_custom_color(label)

                    x, y = pdf.get_x(), pdf.get_y()
                    pdf.set_fill_color(*color)
                    if color == (255, 255, 255):
                        pdf.set_draw_color(0, 0, 0)
                    else:
                        pdf.set_draw_color(*color)
                        
                    pdf.rect(x, y, rect_size, rect_size, style="FD")
                    pdf.set_xy(x + rect_size + spacing_x, y)
                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(0, rect_size, f"{full_label} ({count})", ln=True)
                    pdf.ln(spacing_y)

                if self.grade:
                    pdf.ln(10)
                    pdf.set_font("Arial", "B", 28)
                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(0, 10, f"Final Grade: {self.grade}", ln=True, align="C")

                pdf.output(pdf_path)

            # Generate signed download URL
            url = generate_download_url(self.model_name, pdf_filename)

        except Exception as e:
            ActionDialog(self, title="QR Error", message=f"Failed to generate QR:\n{e}", confirm_text="OK")
            return

        # ---------------- Generate QR ----------------
        qr = qrcode.QRCode(box_size=6, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        img = ImageTk.PhotoImage(qr_img)

        # Close function with refresh
        def close_qr_dialog():
            if self.qr_popup:
                self.qr_popup.close()
                self.qr_popup = None
            self.resize_image()  # refresh main result image

        # ---------------- Show QR Popup ----------------
        self.qr_popup = ActionDialog(
            self, title="", message="", confirm_text="", cancel_text="",
            toggle_button=getattr(self.master, "toggle", None)
        )

        self.qr_content_frame = tk.Frame(self.qr_popup.dialog, bg=self.colors["bg"])
        self.qr_content_frame.pack(padx=10, pady=10)

        self.qr_title_label = tk.Label(
            self.qr_content_frame,
            text="Download PDF",
            fg=self.colors["text"],
            bg=self.colors["bg"],
            font=(theme.font_bold, theme.sizes["subtitle"]),
            justify="center"
        )
        self.qr_title_label.pack(pady=(0, 5))

        self.qr_desc_label = tk.Label(
            self.qr_content_frame,
            text="Scan the QR code below to download the PDF.\nValid for 5 minutes only.",
            fg=self.colors["text"],
            bg=self.colors["bg"],
            font=("Arial", 14),
            justify="center",
            wraplength=400
        )
        self.qr_desc_label.pack(pady=(0, 10))

        self.qr_label = tk.Label(self.qr_content_frame, image=img, bg=self.colors["bg"])
        self.qr_label.image = img
        self.qr_label.pack(pady=(0, 10))

        close_btn = RoundedButton(
            self.qr_content_frame,
            text="Close",
            command=close_qr_dialog,
            width=160,
            height=50,
            radius=16,
            bg=self.colors["accent"],
            fg=self.colors["text"],
            font=(theme.font_bold, 14)
        )
        close_btn.pack(pady=(0, 5))
        theme.subscribe(close_btn.apply_theme)

    # ---------------- Image Resize with correct overlay ----------------
    def resize_image(self, event=None):
        if not self.original_image:
            return

        img_area = self.original_image.width * self.original_image.height

        fw = self.img_frame.winfo_width()
        fh = self.img_frame.winfo_height()
        if fw <= 1 or fh <= 1:
            return

        iw, ih = self.original_image.size
        scale = min(fw / iw, fh / ih)
        nw, nh = int(iw * scale), int(ih * scale)

        # Resize base image
        img = self.original_image.resize((nw, nh), Image.LANCZOS)
        draw = ImageDraw.Draw(img)

        # ---------------- Draw YOLO defects (scaled) ----------------
        for model_defects in self.defects_per_model.values():
            for defect in model_defects:
                if not isinstance(defect, dict) or "bbox" not in defect:
                    continue
                x1, y1, x2, y2 = defect["bbox"]
                bw, bh = x2 - x1, y2 - y1
                area = bw * bh
                area_ratio = area / img_area
                if bw < self.MIN_BOX_W or bh < self.MIN_BOX_H or area < self.MIN_AREA or area_ratio > self.MAX_AREA_RATIO:
                    continue
                # scale coords
                x1_s, y1_s, x2_s, y2_s = int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)
                color = get_custom_color(defect["label"])
                draw.rectangle([x1_s, y1_s, x2_s, y2_s], outline=color, width=3)

        if self.enable_trace and getattr(self, "trace_coords", None):
            # ---------------- Draw trace points/lines (scaled) ----------------
            try:
                # Run detection only once on original image and store results in coord_logs
                for model_defects in self.defects_per_model.values():
                    continue  # already handled YOLO
                # coord_logs was already generated in _generate_merged_image
                for idx, coord in enumerate(getattr(self, "trace_coords", [])):
                    start = (int(coord["start"][0] * scale), int(coord["start"][1] * scale))
                    end   = (int(coord["end"][0] * scale), int(coord["end"][1] * scale))
                    trace_color = get_custom_color("trace_violation")
                    box_size = max(1, int(4 * scale))

                    # Draw rectangles at start/end
                    draw.rectangle([start[0]-box_size, start[1]-box_size,
                                    start[0]+box_size, start[1]+box_size],
                                outline=trace_color, width=2)
                    draw.rectangle([end[0]-box_size, end[1]-box_size,
                                    end[0]+box_size, end[1]+box_size],
                                outline=trace_color, width=2)
                    # Draw connecting line
                    draw.line([start, end], fill=trace_color, width=4)

                    # Add label
                    label_text = f"T{idx+1}"
                    text_offset = (int(6*scale), int(-6*scale))
                    draw.text((start[0]+text_offset[0], start[1]+text_offset[1]),
                            label_text, fill=trace_color)
            except Exception as e:
                print(f"[Trace Annotation] Failed: {e}")

        self.imgtk = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self.imgtk)
        self.img_label.image = self.imgtk
        self.img_label.place(
            relx=0.5,
            rely=0.5,
            anchor="center",
            width=nw,
            height=nh
        )

    # ---------------- Navigation ----------------
    def confirm_back_to_welcome(self):
        from pages.welcomepage import WelcomePage
        def go_back():
            self.show_page(WelcomePage)
        ActionDialog(self, title="Back to Home",
                     message="Return to Home? Current results will be discarded.",
                     confirm_text="Confirm", confirm_command=go_back,
                     toggle_button=getattr(self.master,"toggle",None))

    def confirm_retake_image(self):
        from pages.camerapage import CameraPage

        def retake():
            # Cleanup current page resources
            self.cleanup()

            # Delete previous capture if exists
            for path in [self.original_capture_path, getattr(self, "result_image_path", None)]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f"Failed to remove file {path}: {e}")

            # Decide if this was a final grading session
            is_final_grading = self.model_name and (self.model_name.lower().startswith("final") or self.grade is not None)

            # Force a fresh CameraPage instance
            self.show_page(
                CameraPage,
                monitor=self.monitor,
                model_name=None if is_final_grading else self.model_name,
                grading=is_final_grading,
                force_reload=True  # <-- new param for CameraPage to ensure clean start
            )

        ActionDialog(
            self,
            title="Retake Image",
            message="Retake the image? Current result will be discarded.",
            confirm_text="Confirm",
            confirm_command=retake,
            toggle_button=getattr(self.master, "toggle", None)
        )


    # ---------------- Print ----------------
    def print_result(self):
        try:
            # Ensure printer is available
            conn = cups.Connection()
            printers = conn.getPrinters()
            if not printers:
                ActionDialog(self, title="Print Error", message="No printers detected.", confirm_text="OK")
                return

            # Pick default printer (or first available)
            default_printer = conn.getDefault() or list(printers.keys())[0]

            # PDF path to print
            pdf_filename = os.path.splitext(os.path.basename(self.original_capture_path))[0] + ".pdf"
            base_annotated_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "annotated_images")
            model_folder = os.path.join(base_annotated_folder, self.model_name)
            pdf_path = os.path.join(model_folder, pdf_filename)

            if not os.path.exists(pdf_path):
                ActionDialog(self, title="Print Error", message="PDF not found. Generate it first.", confirm_text="OK")
                return

            # Send the PDF to printer
            conn.printFile(default_printer, pdf_path, "PCB Analysis Result", {})

            ActionDialog(self, title="Print", message=f"Sent to printer: {default_printer}", confirm_text="OK")

        except Exception as e:
            ActionDialog(self, title="Print Error", message=f"Failed to print:\n{e}", confirm_text="OK")


    # ---------------- Fatal Error ----------------
    def _show_fatal_error(self, msg):
        tk.Label(self, text=msg, fg="red", bg=self.colors["bg"], font=(theme.font_bold,24)).pack(expand=True)

    # ---------------- System Monitor ----------------
    def on_system_update(self, problems):
        if problems and not self.dialog:
            def go_to_errorpage():
                # Cleanup current page resources first
                self.cleanup()
                # Show ErrorPage with a "Continue back to ResultsPage" option
                self.show_page(
                    ErrorPage,
                    monitor=self.monitor,
                    problems=problems,
                    next_page=ResultsPage,
                    next_page_kwargs={
                        "monitor": self.monitor,
                        "original_capture_path": self.original_capture_path,
                        "model_name": self.model_name,
                        "defect_summary": self.defect_summary,
                        "defects_per_model": self.defects_per_model,
                        "grade": self.grade,
                        "result_image_path": self.result_image_path
                    }
                )

            # Show ActionDialog first
            self.dialog = ActionDialog(
                self,
                title="System Problem Detected",
                message="\n".join(f"- {p}" for p in problems),
                confirm_text="Exit",
                confirm_command=go_to_errorpage,
                cancel_text="",  # optional
                toggle_button=self.master.toggle
            )

    # ---------------- Printer Detection ----------------
    def _detect_printer(self):
        try:
            conn = cups.Connection()
            printers = conn.getPrinters()
            if not printers:
                return False
            lsusb_output = subprocess.check_output("lsusb", shell=True, text=True)
            for name, info in printers.items():
                uri = info.get("device-uri","")
                if uri.startswith("usb://") and uri.split("/")[-1] in lsusb_output:
                    return True
            return False
        except Exception:
            return False

    # ---------------- Cleanup ----------------
    def cleanup(self):
        if self.dialog:
            self.dialog.destroy()
            self.dialog = None
        if self.qr_popup:
            self.qr_popup.destroy()
            self.qr_popup = None
        self.monitor.unsubscribe(self.on_system_update)

    # ---------------- Destroy ----------------
    def _on_destroy(self, *_):
        self.cleanup()
        if self.apply_theme in theme.subscribers:
            theme.subscribers.remove(self.apply_theme)
