import sys
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLineEdit, QLabel, QFileDialog, QTabWidget, QTextEdit,
        QComboBox, QGraphicsDropShadowEffect, QCheckBox
    )
    from PyQt6.QtCore import QThread, pyqtSignal, Qt, QPropertyAnimation, QEasingCurve
    from PyQt6.QtGui import QColor, QLinearGradient, QBrush, QPalette, QPen
except ImportError:
    print("PyQt6 is not installed. Please run 'pip install PyQt6' in your terminal.")
    sys.exit(1)
import subprocess
from pathlib import Path

# --- Constants ---
UPSCALE_SCRIPT = "manga_upscale.py"
MODEL_DIR = Path("backend") / "models"

# --- Worker Thread for running background processes ---
class ProcessWorker(QThread):
    """Runs a subprocess in a separate thread to keep the UI responsive."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, command):
        super().__init__()
        self.command = command

    def run(self):
        try:
            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
                encoding='utf-8',
                errors='ignore'
            )
            
            for line in iter(process.stdout.readline, ''):
                self.progress.emit(line.strip())
            
            process.stdout.close()
            return_code = process.wait()

            if return_code != 0:
                stderr_output = process.stderr.read()
                self.error.emit(stderr_output)
            else:
                self.finished.emit("Process completed successfully!")
        
        except Exception as e:
            self.error.emit(f"Failed to start process: {str(e)}")

# --- Main Application Window ---
class MangaUpscalerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manga Upscaler")
        self.setGeometry(100, 100, 900, 700)
        self.worker = None

        self.init_ui()
        self.apply_stylesheet()
        self.populate_models()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # --- Upscale Tab ---
        upscale_tab = QWidget()
        upscale_layout = QVBoxLayout(upscale_tab)
        upscale_layout.setSpacing(20)
        
        upscale_layout.addWidget(self.create_path_selector("B&W Input Dir", "bw_path_edit", is_folder=True))
        upscale_layout.addWidget(self.create_path_selector("Color Input Dir", "color_path_edit", is_folder=True))
        upscale_layout.addWidget(self.create_path_selector("Output Directory", "output_path_edit", is_folder=True))

        # Model Selection
        upscale_layout.addWidget(self.create_model_selector("B&W Model", "bw_model_combo"))
        upscale_layout.addWidget(self.create_model_selector("Color Model", "color_model_combo"))
        
        self.run_button = QPushButton("🚀 Run Upscale")
        self.run_button.clicked.connect(self.run_upscale)
        self.add_shadow(self.run_button)
        upscale_layout.addWidget(self.run_button, alignment=Qt.AlignmentFlag.AlignCenter)
        
        tabs.addTab(upscale_tab, "✨ Upscale")

        # --- Extract Tab ---
        extract_tab = QWidget()
        extract_layout = QVBoxLayout(extract_tab)
        extract_layout.setSpacing(20)
        
        extract_layout.addWidget(self.create_path_selector("Archive Input Dir", "extract_path_edit", is_folder=True))
        self.overwrite_checkbox = QCheckBox("Overwrite existing extracted folders")
        extract_layout.addWidget(self.overwrite_checkbox)

        self.extract_button = QPushButton("📦 Run Extraction")
        self.extract_button.clicked.connect(self.run_extraction)
        self.add_shadow(self.extract_button)
        extract_layout.addWidget(self.extract_button, alignment=Qt.AlignmentFlag.AlignCenter)
        tabs.addTab(extract_tab, "📦 Extract")

        # --- Setup Tab ---
        setup_tab = QWidget()
        setup_layout = QVBoxLayout(setup_tab)
        setup_layout.setSpacing(15)
        setup_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        setup_layout.addWidget(QLabel("Download Model Packs:"))

        self.download_best_button = QPushButton("🚀 Download Best Models (Both)")
        self.download_best_button.clicked.connect(lambda: self.run_download("best"))
        self.add_shadow(self.download_best_button)
        setup_layout.addWidget(self.download_best_button)

        self.download_bw_button = QPushButton("B&W Download B&W Models")
        self.download_bw_button.clicked.connect(lambda: self.run_download("bw"))
        self.add_shadow(self.download_bw_button)
        setup_layout.addWidget(self.download_bw_button)

        self.download_color_button = QPushButton("🎨 Download Color Models")
        self.download_color_button.clicked.connect(lambda: self.run_download("color"))
        self.add_shadow(self.download_color_button)
        setup_layout.addWidget(self.download_color_button)
        
        setup_layout.addStretch() # Pushes buttons to the top

        tabs.addTab(setup_tab, "🔧 Setup & Models")

        # --- Log Console ---
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        main_layout.addWidget(self.log_console)

    def create_path_selector(self, label_text, line_edit_name, is_folder=True):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        label = QLabel(label_text)
        label.setFixedWidth(120)
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(f"Path to {label_text}...")
        setattr(self, line_edit_name, line_edit)
        
        button = QPushButton("📂 Browse")
        button.clicked.connect(lambda: self.select_path(line_edit, is_folder))
        
        layout.addWidget(label)
        layout.addWidget(line_edit)
        layout.addWidget(button)
        return widget

    def select_path(self, line_edit, is_folder):
        if is_folder:
            path = QFileDialog.getExistingDirectory(self, "Select Directory")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if path:
            line_edit.setText(path)

    def create_model_selector(self, label_text, combo_box_name):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        label = QLabel(label_text)
        label.setFixedWidth(120)
        combo_box = QComboBox()
        setattr(self, combo_box_name, combo_box)
        
        button = QPushButton("📂 Browse")
        button.clicked.connect(lambda: self.select_model_path(combo_box))
        
        layout.addWidget(label)
        layout.addWidget(combo_box)
        layout.addWidget(button)
        return widget

    def select_model_path(self, combo_box):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model File", str(MODEL_DIR), "Model Files (*.pth *.safetensors)")
        if path:
            # Add the selected model to the list if it's not already there
            p = Path(path)
            if combo_box.findText(p.name) == -1:
                combo_box.addItem(p.name)
            combo_box.setCurrentText(p.name)

    def populate_models(self):
        if not MODEL_DIR.exists():
            self.log_console.append(f"Warning: Model directory not found at '{MODEL_DIR}'. Please run setup.")
            return

        models = [f.name for f in MODEL_DIR.glob("*.pth")] + [f.name for f in MODEL_DIR.glob("*.safetensors")]
        
        # General-purpose models that should appear in both lists
        general_models = [m for m in models if '4x-anime' in m.lower() or 'ultrasharp' in m.lower()]
        
        # B&W specific models
        bw_models = [m for m in models if 'manga' in m.lower() or 'bw' in m.lower()]
        
        # Color specific models are those not in B&W or general lists
        color_models = [m for m in models if m not in bw_models and m not in general_models]

        # Combine general models with specific lists
        final_bw_models = sorted(list(set(bw_models + general_models)))
        final_color_models = sorted(list(set(color_models + general_models)))

        self.bw_model_combo.addItems(final_bw_models if final_bw_models else ["No B&W models found"])
        self.color_model_combo.addItems(final_color_models if final_color_models else ["No color models found"])


    def run_download(self, model_type):
        self.log_console.clear()
        self.log_console.append(f"Starting download for {model_type} models...")
        command = [sys.executable, UPSCALE_SCRIPT, "download", model_type]
        self.start_worker(command)

    def run_extraction(self):
        input_dir = self.extract_path_edit.text()
        if not input_dir:
            self.log_console.append("❌ Error: Please select an input directory for extraction.")
            return

        self.log_console.clear()
        self.log_console.append("📦 Starting extraction process...")
        
        command = [sys.executable, UPSCALE_SCRIPT, "extract", "--input", input_dir]
        if self.overwrite_checkbox.isChecked():
            command.append("--overwrite")
            
        self.start_worker(command)

    def run_upscale(self):
        bw_path = self.bw_path_edit.text()
        color_path = self.color_path_edit.text()
        output_path = self.output_path_edit.text()

        if not (bw_path or color_path):
            self.log_console.append("❌ Error: Please select at least one input directory.")
            return
        if not output_path:
            self.log_console.append("❌ Error: Please select an output directory.")
            return

        self.log_console.clear()
        self.log_console.append("🚀 Starting upscale process...")
        
        command = [
            sys.executable, UPSCALE_SCRIPT, "upscale",
            "--output", output_path,
            "--model-bw", self.bw_model_combo.currentText(),
            "--model-color", self.color_model_combo.currentText()
        ]
        if bw_path:
            command.extend(["--bw", bw_path])
        if color_path:
            command.extend(["--color", color_path])
            
        self.start_worker(command)

    def start_worker(self, command):
        if self.worker and self.worker.isRunning():
            self.log_console.append("⏳ A process is already running. Please wait.")
            return
        
        self.run_button.setEnabled(False)
        self.extract_button.setEnabled(False)
        self.download_best_button.setEnabled(False)
        self.download_bw_button.setEnabled(False)
        self.download_color_button.setEnabled(False)
        
        self.worker = ProcessWorker(command)
        self.worker.progress.connect(self.log_console.append)
        self.worker.error.connect(self.handle_error)
        self.worker.finished.connect(self.handle_finish)
        self.worker.start()

    def handle_error(self, message):
        self.log_console.append(f"\n--- ❌ ERROR ---\n{message}\n-----------------")
        self.reset_buttons()

    def handle_finish(self, message):
        self.log_console.append(f"\n--- ✅ SUCCESS ---\n{message}\n-----------------")
        self.reset_buttons()

    def reset_buttons(self):
        self.run_button.setEnabled(True)
        self.extract_button.setEnabled(True)
        self.download_best_button.setEnabled(True)
        self.download_bw_button.setEnabled(True)
        self.download_color_button.setEnabled(True)

    def add_shadow(self, widget):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 5)
        widget.setGraphicsEffect(shadow)

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1a1d2e;
                color: #e0e0e0;
                font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
                font-size: 11pt;
            }
            QTabWidget::pane {
                border: none;
                background-color: #23273a;
                border-radius: 12px;
            }
            QTabBar::tab {
                background: #23273a;
                color: #a0a0c0;
                padding: 12px 25px;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                font-weight: bold;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4a00e0, stop:1 #8e2de2);
                color: white;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4a00e0, stop:1 #8e2de2);
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 16px;
                font-weight: bold;
                font-size: 12pt;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5a10f0, stop:1 #9f3ef2);
            }
            QPushButton:disabled {
                background-color: #3b3f5a;
                color: #808080;
            }
            QLineEdit, QTextEdit, QComboBox {
                background-color: #2f344f;
                border: 1px solid #40456a;
                padding: 8px;
                border-radius: 12px;
                color: #f0f0f0;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #8e2de2;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png); /* You might need to create a small arrow icon */
            }
            QLabel {
                padding: 5px;
                font-weight: bold;
                color: #c0c0e0;
            }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MangaUpscalerGUI()
    window.show()
    sys.exit(app.exec())