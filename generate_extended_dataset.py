import cv2
import numpy as np
import json
import os
from torchvision import transforms
from PIL import Image
import random
from tqdm import tqdm

IMAGES_FOLDER = "alphapilot_imgs"


def random_transform(image, points):
    """
    Applies a random rotation of 90, 180, or 270 degrees to the image and transforms the points.

    Args:
        image (numpy.ndarray): The input image.
        points (list of lists): Corner coordinates [x1, y1, x2, y2, x3, y3, x4, y4].

    Returns:
        tuple: The rotated image and the rotated points.
    """

    # Get image dimensions
    height, width = image.shape[:2]
    points = [points[i:i + 2] for i in range(0, len(points), 2)]

    # Choose a random rotation angle
    angles = [90, 180, 270]
    angle = random.choice(angles)

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # Apply the rotation to the points
    rotated_points = []
    for corner in points:
        x, y = corner
        rotated_point = np.round(np.dot(rotation_matrix, np.array([x, y, 1]))).astype(int)
        rotated_points.extend(rotated_point)

    # Apply color jitter
    transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    # Convert numpy array to PIL Image
    rotated_image = Image.fromarray(rotated_image)
    rotated_image = transform(rotated_image)
    rotated_image = np.array(rotated_image)

    return rotated_image, rotated_points


def main():
    # Opening original labels
    with open('training_GT_labels_v2.json', 'r') as f:
        data = json.load(f)

    img_path = os.path.join(os.getcwd(), "alphapilot_imgs")  # path to image directory

    transformed_save_path = os.path.join(os.getcwd(), "alphapilot_extended_imgs")
    if not os.path.exists(transformed_save_path):
        os.makedirs(transformed_save_path)

    transformed_labels = {}
    for image_name, points in tqdm(data.items(), desc="Transforming images"):
        image = cv2.imread(os.path.join(img_path, image_name))

        # Apply random rotation
        transformed_image, rotated_points = random_transform(image, points[0])

        # Add original image and labels to the new dataset
        transformed_labels[image_name] = points
        cv2.imwrite(os.path.join(transformed_save_path, image_name), image)

        transformed_img_name = "transformed_" + image_name
        # Add rotated image and labels to the new dataset
        transformed_labels[transformed_img_name] = np.array([rotated_points]).astype(int).tolist()
        cv2.imwrite(os.path.join(transformed_save_path, transformed_img_name), transformed_image)

    # Save the updated dataset
    with open('transformed_training_GT_labels_v2.json', 'w') as f:
        json.dump(transformed_labels, f)


if __name__ == '__main__':
    main()
