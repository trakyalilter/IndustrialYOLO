import sys
import os
import pyvista as pv
import numpy as np
import random
import json
import cv2 as cv
import time
class ImageGenerator():
    def __init__(self,mesh_path='C:/IT/CameraCalibration/CamCalibrator/env/samet.STL'):
        self.meshes = [
            pv.read(mesh_path)
            # Add additional meshes as needed
        ]
        self.output_folder_original = 'C:/IT/yoloM/images/original'
        self.output_folder_train = 'C:/IT/yoloM/dataset/images/train'
        self.output_folder_val = 'C:/IT/yoloM/dataset/images/val'
        os.makedirs(self.output_folder_original, exist_ok=True)
        os.makedirs(self.output_folder_train, exist_ok=True)
        os.makedirs(self.output_folder_val, exist_ok=True)

    def generate_image(self, i=0):
        plotter = pv.Plotter(window_size=(640, 640), off_screen=True)
        plotter.background_color = [54, 54, 54]

        mesh = self.meshes[0]
        print(mesh.center)
        mesh.translate((random.uniform(-10, 10), random.uniform(-10, 10), 0), inplace=True)
        print(mesh.center)
        plotter.add_mesh(mesh, color=[0, 219, 33], style='surface')
    
        _n = random.randint(0,1)
        print(_n)
        
        if _n == 0:
            camera_distance = random.uniform(400, 700)

            plotter.camera_position = [(0, 0, camera_distance), (0, 0, 0), (0, 1, 0)]
        else:
            camera_distance = random.uniform(-400, -700)
            plotter.camera_position = [(0, 0, camera_distance), (0, 0, 0), (0, -1, 0)]
        
        plotter.render()
        output_path = os.path.join(self.output_folder_original, f'image_{_n}_{i}.png')
        plotter.screenshot(output_path)
        # Edge detection processing
        image = cv.imread(output_path)
        time.sleep(1)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 0, 20)
        kernel = np.ones((3, 3), np.uint8)
        closing = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
        _, th3 = cv.threshold(closing, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        transformed_image = cv.cvtColor(th3, cv.COLOR_GRAY2RGB)
        transformed_output_path_train = os.path.join(self.output_folder_train, f'transformed_image_{_n}_{i}.png')
        transformed_output_path_val = os.path.join(self.output_folder_val, f'transformed_image_{_n}_{i}.png')
        cv.imwrite(transformed_output_path_train, transformed_image)
        cv.imwrite(transformed_output_path_val, transformed_image)
        time.sleep(1)

    def generate_random_images(self, num_images):
            for i in range(num_images):
                self.generate_image(i=i)

