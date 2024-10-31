import sys
import time
import os
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox
from ImageGenerator import ImageGenerator
from ContouringImg import Contouring
from ultralytics import YOLO

class YoloTrainerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize UI
        self.setWindowTitle("YOLO Model Trainer")
        self.setGeometry(100, 100, 400, 300)
        
        # Main layout
        layout = QVBoxLayout()
        
        # Input fields and labels
        self.epoch_label = QLabel("Epochs:")
        layout.addWidget(self.epoch_label)
        self.epoch_input = QLineEdit(self)
        self.epoch_input.setPlaceholderText("e.g., 50")
        layout.addWidget(self.epoch_input)
        
        self.size_label = QLabel("Image Size:")
        layout.addWidget(self.size_label)
        self.size_input = QLineEdit(self)
        self.size_input.setPlaceholderText("e.g., 640")
        layout.addWidget(self.size_input)
        
        self.name_label = QLabel("Training Run Name:")
        layout.addWidget(self.name_label)
        self.name_input = QLineEdit(self)
        self.name_input.setText(time.strftime("%m-%d-%Y-%H:%M:%S", time.localtime()))
        layout.addWidget(self.name_input)
        
        self.clean_checkbox = QCheckBox("Clean previous files")
        layout.addWidget(self.clean_checkbox)
        
        # File selection button for the mesh path
        self.mesh_path = ""
        self.mesh_button = QPushButton("Select Mesh File")
        self.mesh_button.clicked.connect(self.select_mesh_file)
        layout.addWidget(self.mesh_button)
        
        self.mesh_label = QLabel("Selected Mesh File: None")
        layout.addWidget(self.mesh_label)
        
        # Start button
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        layout.addWidget(self.start_button)
        
        # Set layout
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
    
    def select_mesh_file(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Select Mesh File", "", "STL Files (*.stl);;All Files (*)", options=options)
        if file:
            self.mesh_path = file
            self.mesh_label.setText(f"Selected Mesh File: {os.path.basename(file)}")
    
    def start_training(self):
        try:
            epochs = int(self.epoch_input.text()) if self.epoch_input.text() else 50
            imgsz = int(self.size_input.text()) if self.size_input.text() else 640
            name = self.name_input.text() if self.name_input.text() else time.strftime("%m-%d-%Y-%H:%M:%S", time.localtime())
            clean = self.clean_checkbox.isChecked()
            
            # Cleaning files if requested
            if clean:
                print("Cleaning all files")
                import glob
                main_path = "C:/IT/YoloM/"
                paths = [
                    "dataset/images/train", "dataset/images/val", "dataset/labels/train",
                    "dataset/labels/val", "images/original", "contoured_images"
                ]
                for path in paths:
                    full_path = os.path.join(main_path, path)
                    files = glob.glob(full_path + "/*")
                    for file in files:
                        os.remove(file)
                print("Cleaned all files")
            else:
                print("Not cleaning")
            
            # Run the training sequence
            generator = ImageGenerator(self.mesh_path)
            contouring = Contouring()
            generator.generate_random_images(num_images=20)
            contouring.Contour()
            
            model = YOLO("yolov8n.pt")
            project_dir = "C:/IT/YoloM/yoloenv/runs"
            model.train(data="C:/IT/YoloM/data.yaml", epochs=epochs, imgsz=imgsz, name=name, project=project_dir)
            print("Training started successfully")
        
        except Exception as e:
            error_dialog = QtWidgets.QErrorMessage(self)
            error_dialog.showMessage(f"Error during training: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YoloTrainerApp()
    window.show()
    sys.exit(app.exec_())
