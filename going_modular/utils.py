import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from torchvision.transforms import ToPILImage
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import ToPILImage
import torchvision.transforms as T
from torchvision.transforms import Resize, CenterCrop, ToTensor, Compose
from PIL import ImageOps

class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, class_names_mapping=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.class_map = class_names_mapping if class_names_mapping is not None else {}

        # Filter out images without bounding boxes
        self.img_names = [img_name for img_name in os.listdir(img_dir) if self.has_boxes(label_dir, img_name)]

        # Gather unique classes
        self.classes = self.get_unique_labels()

    def has_boxes(self, label_dir, img_name):
        label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))
        try:
            with open(label_path, 'r') as file:
                lines = file.readlines()
                return any(len(line.split()) == 5 for line in lines)  # Check if any line has 5 elements
        except FileNotFoundError:
            return False  # If label file is not found, consider it as no boxes

    def __len__(self):
        return len(self.img_names)
    
    def convert_yolo_to_rcnn(self, annotations, img_width, img_height):
        rcnn_boxes = []
        for class_label, bbox in annotations:
            x_center, y_center, width, height = bbox
            xmin = max((x_center - width / 2) * img_width, 0)
            ymin = max((y_center - height / 2) * img_height, 0)
            xmax = min((x_center + width / 2) * img_width, img_width)
            ymax = min((y_center + height / 2) * img_height, img_height)
            rcnn_boxes.append([xmin, ymin, xmax, ymax])
        return rcnn_boxes


    def scale_boxes(self, boxes, original_width, original_height, transformed_width, transformed_height):
        scaled_boxes = []
        for xmin, ymin, xmax, ymax in boxes:
            scaled_xmin = xmin / original_width * transformed_width
            scaled_ymin = ymin / original_height * transformed_height
            scaled_xmax = xmax / original_width * transformed_width
            scaled_ymax = ymax / original_height * transformed_height
            scaled_boxes.append([scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax])
        return scaled_boxes

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        label_path = os.path.join(self.label_dir, self.img_names[idx].replace('.jpg', '.txt'))

        # Load image using OpenCV and convert to RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        #print(f"Original image size: {original_width} x {original_height}")  # Debug print

        # Load labels
        annotations = self.load_labels(label_path)

        # Convert YOLO annotations to [xmin, ymin, xmax, ymax] format
        boxes = self.convert_yolo_to_rcnn(annotations, original_width, original_height)

        labels = [annotation[0] for annotation in annotations]

        if self.transform:
            image_pil = Image.fromarray(image)
            transformed_image = self.transform(image_pil)
            transformed_height, transformed_width = transformed_image.size(1), transformed_image.size(2)
            #print(f"Transformed image size: {transformed_width} x {transformed_height}")  # Debug print

            # Scale bounding boxes
            scaled_boxes = self.scale_boxes(boxes, original_width, original_height, transformed_width, transformed_height)

            image = transformed_image
            boxes = scaled_boxes
        else:
            image = T.ToTensor()(image)

        # Convert boxes and labels to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}
        
        image_tensor_size = image.shape
        #print(f"Image tensor size in __getitem__: {image_tensor_size}")  # Debug print

        return image, target

    def visualize_transformation(self, idx):
        # Get the original image and its annotations
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        label_path = os.path.join(self.label_dir, self.img_names[idx].replace('.jpg', '.txt'))

        # Load and display original image
        original_image = cv2.imread(img_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_annotations = self.load_labels(label_path)
        original_height, original_width = original_image.shape[:2]

        # Convert YOLO annotations to [xmin, ymin, xmax, ymax] format for the original image
        converted_original_boxes = self.convert_yolo_to_rcnn(original_annotations, original_width, original_height)

        print("Original Image:")
        print("Original Annotations:", original_annotations)
        self.draw_bounding_boxes(original_image, [(ann[0], box) for ann, box in zip(original_annotations, converted_original_boxes)])

        # Apply transformations
        if self.transform:
            image_pil = Image.fromarray(original_image)
            transformed_image = self.transform(image_pil)
            transformed_height, transformed_width = transformed_image.size(1), transformed_image.size(2)

            # Convert YOLO annotations to [xmin, ymin, xmax, ymax] format
            boxes = self.convert_yolo_to_rcnn(original_annotations, original_width, original_height)

            # Scale bounding boxes
            scaled_boxes = self.scale_boxes(boxes, original_width, original_height, transformed_width, transformed_height)
            transformed_annotations = [(label, box) for label, box in zip([ann[0] for ann in original_annotations], scaled_boxes)]

            print("Transformed Image:")
            print("Transformed Annotations:", transformed_annotations)
            self.draw_bounding_boxes(transformed_image, transformed_annotations)
        else:
            transformed_image = original_image
            transformed_annotations = original_annotations

            print("No Transformation Applied:")
            print("Annotations:", transformed_annotations)
            self.draw_bounding_boxes(transformed_image, transformed_annotations)

    def load_labels(self, label_path):
        with open(label_path, 'r') as file:
            lines = file.readlines()

        #print(f"Label file {label_path}:")
        annotations = []
        for line in lines:
            elements = line.split()
            class_label = int(elements[0])
            bbox = list(map(float, elements[1:]))

            # Check if bbox has 4 elements (x_center, y_center, width, height)
            if len(bbox) != 4:
                #print(f"Warning: Incorrect bounding box format in {label_path} for line: {line.strip()}")
                continue  # Skip this line and move to the next one

            # Check if width and height are positive
            if bbox[2] <= 0 or bbox[3] <= 0:
                #print(f"Warning: Non-positive bbox dimensions in {label_path} for line: {line.strip()}")
                continue  # Skip this line and move to the next one

            annotations.append((class_label, bbox))

        #print(f"Parsed annotations for {label_path}: {annotations}")
        return annotations

    #(class_id x_center y_center width height) format
    def draw_bounding_boxes(self, image, annotations):
        # Convert image to numpy array if necessary
        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).numpy()
        elif isinstance(image, Image.Image):
            image = np.array(image)

        img_height, img_width, _ = image.shape
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(image)

        for annotation in annotations:
            class_label, bbox = annotation
            if len(bbox) != 4:
                raise ValueError(f"Expected bbox to have 4 values, got {len(bbox)}: {bbox}")

            xmin, ymin, xmax, ymax = bbox
            box_width = xmax - xmin
            box_height = ymax - ymin

            # Debug print statements
            print(f"Drawing box for class {class_label}: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
            print(f"Box dimensions: width={box_width}, height={box_height}")

            class_name = self.class_map.get(class_label, str(class_label))
            rect = patches.Rectangle((xmin, ymin), box_width, box_height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(xmin, ymin, class_name, color='white', fontsize=12, weight='bold', bbox=dict(facecolor='red', alpha=0.5, pad=0, edgecolor='none'))

        plt.axis('off')
        plt.show()

    def draw_batch_boxes(self, images, annotations_batch):
        """
        Draw bounding boxes on a batch of images.
        
        Args:
        images (list of PIL Images or tensors): The images in a batch.
        annotations_batch (list of dicts): Each dict contains 'boxes' and 'labels' for an image.
        """
        batch_size = len(images)
        fig, axs = plt.subplots(1, batch_size, figsize=(12 * batch_size, 12))

        if batch_size == 1:  # If there's only one image, axs is not a list
            axs = [axs]

        for i, (image, annotations) in enumerate(zip(images, annotations_batch)):
            if torch.is_tensor(image):
                image = denormalize(image).permute(1, 2, 0).numpy()
            axs[i].imshow(image)
            axs[i].axis('off')

            boxes = annotations['boxes']
            labels = annotations['labels']

            for box, label in zip(boxes, labels):
                xmin, ymin, xmax, ymax = box
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
                axs[i].add_patch(rect)
                class_name = self.class_map.get(label.item(), str(label.item()))
                axs[i].text(xmin, ymin, class_name, color='white', fontsize=12, weight='bold', bbox=dict(facecolor='red', alpha=0.5, pad=0, edgecolor='none'))

        plt.show()
        
    def get_unique_labels(self):
        """
        Gathers a list of unique class labels from the label files.
        """
        unique_labels = set()
        for img_name in self.img_names:
            label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
            with open(label_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    class_label = int(line.split()[0])
                    unique_labels.add(class_label)
        return sorted(unique_labels)
    
    def get_images_for_each_class(self):
        """
        Returns one image for each unique class label in the dataset.
        """
        images_for_classes = {}
        for img_name in self.img_names:
            label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
            annotations = self.load_labels(label_path)
            
            for annotation in annotations:
                class_label = annotation[0]

                if class_label not in images_for_classes:
                    img_path = os.path.join(self.img_dir, img_name)
                    image = Image.open(img_path).convert('RGB')
                    images_for_classes[class_label] = (image, annotations)
                    break

        return images_for_classes
    
    def transform_and_scale_single_image(self, img_input, label_path=None, debug=False):
        if isinstance(img_input, str):
            # If img_input is a file path, read the image from the file
            image = cv2.imread(img_input)
            if image is None:
                raise FileNotFoundError(f"Image file not found: {img_input}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(img_input, np.ndarray):
            # If img_input is an image array (frame), use it directly
            image = img_input
        else:
            raise TypeError("img_input must be a file path or an image array")

        original_height, original_width = image.shape[:2]

        # Load labels if label path is provided
        annotations = self.load_labels(label_path) if label_path else []

        # Convert YOLO annotations to [xmin, ymin, xmax, ymax] format
        boxes = self.convert_yolo_to_rcnn(annotations, original_width, original_height) if label_path else []

        if self.transform:
            image_pil = Image.fromarray(image)
            transformed_image = self.transform(image_pil)
            transformed_height, transformed_width = transformed_image.size(1), transformed_image.size(2)

            # Scale bounding boxes
            scaled_boxes = self.scale_boxes(boxes, original_width, original_height, transformed_width, transformed_height)

            image = transformed_image
            boxes = scaled_boxes
        else:
            image = T.ToTensor()(image)

        if debug:
            print(f"Transformed image size: {transformed_width} x {transformed_height}")
            print(f"Scaled boxes: {scaled_boxes}")

        return image, boxes

    def transform_and_scale_frame(self, frame, debug=False):
        # Directly use the frame (a NumPy array)
        image = frame
        original_height, original_width = image.shape[:2]

            # Print original size
        if debug:
            print(f"Original frame size: {original_width} x {original_height}")

        # Assuming no labels are provided for video frames
        annotations = []

        if self.transform:
            image_pil = Image.fromarray(image)
            transformed_image = self.transform(image_pil)
            transformed_height, transformed_width = transformed_image.size(1), transformed_image.size(2)

            # Scale bounding boxes (if any) - here we assume no boxes for video frames
            scaled_boxes = []

            image = transformed_image
        else:
            image = T.ToTensor()(image)

        if debug:
            print(f"Transformed image size: {transformed_width} x {transformed_height}")
            print(f"Scaled boxes: {scaled_boxes}")
        return image


