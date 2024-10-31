import cv2 as cv
import numpy as np
import os

# folder = 'C:/IT/YoloTest/env/pos_Images'
# output_folder = 'C:/IT/YoloTest/env/pos_Images'

class Contouring:
    @staticmethod
    def Contour():
        folder = 'C:/IT/YoloM/dataset/images/train'
        contoured_images_folder = 'C:/IT/YoloM/contoured_images'
        train_output_folder = 'C:/IT/YoloM/dataset/labels/train'
        val_output_folder =  'C:/IT/YoloM/dataset/labels/val'
        os.makedirs(contoured_images_folder, exist_ok=True)
        os.makedirs(train_output_folder, exist_ok=True)
        os.makedirs(val_output_folder, exist_ok=True)

        images = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('png', 'jpg', 'jpeg'))]
        print(images)
        class_number = 0
        for img_path in images:
            # Load the image
            img = cv.imread(img_path)
            img_name = os.path.basename(img_path)
            img_base_name = os.path.splitext(img_name)[0]

            # Convert to grayscale
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Apply Canny edge detection
            edges = cv.Canny(gray, 0, 20)
            
            kernel = np.ones((3, 3), np.uint8)
            closing = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
            
            ret3, th3 = cv.threshold(closing, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            # cv.imwrite(img_path,th3)
            # Find contours
            contours, _ = cv.findContours(th3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # Find the contour with the largest area
            if contours:
                largest_contour = max(contours, key=cv.contourArea)
                x, y, w, h = cv.boundingRect(largest_contour)
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Get the minimum area bounding box for the largest contour
                rect = cv.minAreaRect(largest_contour)
                box = cv.boxPoints(rect)
                box = np.int32(box)

                # Calculate YOLO format annotation
                img_height, img_width = img.shape[:2]
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height
                
                if(img_base_name.split('_')[2] == '1'):
                    class_number = 1
                elif(img_base_name.split('_')[2] == '0'):
                    class_number = 0
                # Save YOLO annotation
                yolo_annotation = f"{class_number} {x_center} {y_center} {width} {height}\n"
                annotation_path = os.path.join(train_output_folder, f'{img_base_name}.txt')
                with open(annotation_path, 'w') as f:
                    f.write(yolo_annotation)
                annotation_path = os.path.join(val_output_folder, f'{img_base_name}.txt')
                with open(annotation_path, 'w') as f:
                    f.write(yolo_annotation)
            # Save the result to a file
            output_image_path = os.path.join(contoured_images_folder, f'contoured-{img_name}')
            cv.imwrite(output_image_path, img)


        print("Annotation and images saved.")
