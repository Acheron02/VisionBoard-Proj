import tkinter as tk
import json
from ui.roundedbutton import RoundedButton
from pages.camerapage import CameraPage
from pages.errorpage import ErrorPage
from ui.theme import theme
from ui.actiondialog import ActionDialog

button_height = 100
button_width = 300
corner_radius = 24
grid_padx = 30
grid_pady = 20
desc_wrap = 280

with open("config/grading_config.json", "r") as f:
    CONFIG = json.load(f)


class ChooseModel(tk.Frame):
    def __init__(self, parent, show_page, monitor, theme):
        super().__init__(parent)

        self.show_page = show_page
        self.monitor = monitor
        self.theme = theme
        self.dialog = None

        # === Main container ===
        self.container = tk.Frame(self)
        self.container.pack(expand=True)

        self.title_label = tk.Label(
            self.container,
            text="Choose Your Vision",
            font=(theme.font_bold, theme.sizes["title"])
        )
        self.title_label.pack(pady=(0, 10))

        self.info_label = tk.Label(
            self.container,
            text="Select a model to get started.",
            font=(theme.font_regular, theme.sizes["subtitle"])
        )
        self.info_label.pack(pady=(0, 50))

        # === Grid frame ===
        self.grid_frame = tk.Frame(self.container)
        self.grid_frame.pack()

        colors = theme.colors()
        self.buttons = []

        # === Helper to create grid items ===
        def create_item(row, col, title, desc, command, bold=False):
            frame = tk.Frame(self.grid_frame)
            frame.grid(row=row, column=col, padx=grid_padx, pady=grid_pady, sticky="n")

            btn = RoundedButton(
                frame,
                text=title,
                width=button_width,
                height=button_height,
                radius=corner_radius,
                font=(theme.font_bold if bold else theme.font_regular, theme.sizes["body"]),
                bg=colors["accent"],
                fg=colors["text"],
                command=command
            )
            btn.pack()

            label = tk.Label(
                frame,
                text=desc,
                font=(theme.font_regular, theme.sizes["subtitle"]),
                wraplength=desc_wrap,
                justify="center"
            )
            label.pack(pady=(10, 0))

            return btn, label, frame

        # === Model buttons ===
        btn1, lbl1, frm1 = create_item(
            0, 0,
            "Model 1",
            "Short and Open Circuit, 90° Angle Traces, Trace Clearance Violation",
            lambda: self.select_model(1)
        )

        btn2, lbl2, frm2 = create_item(
            0, 1,
            "Model 2",
            "Poor Solder, Solder Bridges, and Missing Components",
            lambda: self.select_model(2)
        )

        btn3, lbl3, frm3 = create_item(
            1, 0,
            "Model 3",
            "Passive Components: Resistors and Capacitors",
            lambda: self.select_model(3)
        )

        self.final_btn, self.final_lbl, self.final_frame = create_item(
            1, 1,
            "Final PCB Grading",
            "Ready to grade your PCB?",
            self.run_final_grading,
            bold=False
        )

        self.buttons.extend([btn1, btn2, btn3])
        self.desc_labels = [lbl1, lbl2, lbl3, self.final_lbl]
        self.frames = [frm1, frm2, frm3, self.final_frame]

        # Apply initial theme
        self.apply_theme(colors)
        theme.subscribe(self.apply_theme)

        # Subscribe to monitor
        self.monitor.subscribe(self.on_system_update)
        self.on_system_update(self.monitor.problems)

        self.bind("<Destroy>", self._on_destroy)

    # -----------------------------
    # Theme Handling
    def apply_theme(self, c):
        try:
            self.configure(bg=c["bg"])
            self.container.configure(bg=c["bg"])
            self.grid_frame.configure(bg=c["bg"])

            self.title_label.configure(bg=c["bg"], fg=c["text"])
            self.info_label.configure(bg=c["bg"], fg=c["text2"])

            for frame in self.frames:
                frame.configure(bg=c["bg"])

            for label in self.desc_labels:
                label.configure(bg=c["bg"], fg=c["text2"])

            for btn in self.buttons:
                btn.apply_theme(c)

            self.final_btn.apply_theme(c)

        except tk.TclError:
            pass

    # -----------------------------
    # Cleanup
    def _on_destroy(self, *_):
        if self.apply_theme in theme.subscribers:
            theme.subscribers.remove(self.apply_theme)

        self.monitor.unsubscribe(self.on_system_update)

        if self.dialog:
            self.dialog.destroy()
            self.dialog = None

    # -----------------------------
    # Navigation
    def select_model(self, model_number):
        if self.monitor.problems:
            return
        self.show_page(
            CameraPage,
            monitor=self.monitor,
            model_name=f"Model {model_number}",
            grading=False,
            config=CONFIG
        )

    def run_final_grading(self):
        if self.monitor.problems:
            return
        self.show_page(
            CameraPage,
            monitor=self.monitor,
            model_name=None,
            grading=True,
            config=CONFIG
        )

    # -----------------------------
    # SystemMonitor
    def on_system_update(self, problems):
        for btn in self.buttons:
            btn.set_disabled(bool(problems))

        self.final_btn.set_disabled(bool(problems))

        if problems:
            if not self.dialog:
                def on_exit():
                    self.show_page(
                        ErrorPage,
                        monitor=self.monitor,
                        problems=problems,
                        next_page=ChooseModel,
                        theme=self.theme
                    )

                self.dialog = ActionDialog(
                    self,
                    title="System Error",
                    message="\n".join(problems),
                    confirm_text="Exit",
                    confirm_command=on_exit,
                    cancel_text="",
                    toggle_button=self.master.toggle
                )

                self.dialog.bind(
                    "<Destroy>",
                    lambda e: setattr(self, "dialog", None)
                )
            else:
                self.dialog.update_message("\n".join(problems))
        else:
            if self.dialog:
                self.dialog.close()
                self.dialog = None
