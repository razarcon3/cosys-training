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
        # convert to python int to be json serializable
        rotated_points.extend([int(rotated_point[0]), int(rotated_point[1])])

    # Apply color jitter
    transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    # Convert numpy array to PIL Image
    rotated_image = Image.fromarray(rotated_image)
    rotated_image = transform(rotated_image)
    rotated_image = np.array(rotated_image)

    return rotated_image, rotated_points


def create_train_test_split(data, test_size=0.2, random_state=None):
    """Splits the data into training and testing sets.

    Args:
        data: A dictionary where keys are image names and values are labels.
        test_size: The proportion of the data to include in the test split (0.0 to 1.0).
        random_state:  An integer for reproducible results.  Pass an integer for consistent splits.

    Returns:
        Two dictionaries: training_data, testing_data
    """
    if random_state is not None:
        random.seed(random_state)

    image_names = list(data.keys())
    random.shuffle(image_names)  # Shuffle image names *in place*

    split_index = int(len(image_names) * (1 - test_size))
    training_names = image_names[:split_index]
    testing_names = image_names[split_index:]

    training_data = {name: data[name] for name in training_names}
    testing_data = {name: data[name] for name in testing_names}

    return training_data, testing_data


def main():
    # Opening original labels
    with open('training_GT_labels_v2.json', 'r') as f:
        data = json.load(f)

    # removes images w/o labels
    valid_data = {}
    for img_name in data.keys():
        corners_data = data[img_name]
        if corners_data and len(corners_data[0]) >= 2:  # Check for valid data
            valid_data[img_name] = corners_data

    original_img_path = os.path.join(os.getcwd(), "alphapilot_imgs")  # path to image directory

    # Create output directories
    training_save_path = os.path.join(os.getcwd(), "alphapilot_extended_training_imgs")
    testing_save_path = os.path.join(os.getcwd(), "alphapilot_extended_testing_imgs")
    os.makedirs(training_save_path, exist_ok=True)  # exist_ok=True prevents errors if they already exist
    os.makedirs(testing_save_path, exist_ok=True)

    # Split the data *before* transformations
    training_data, testing_data = create_train_test_split(data, test_size=0.2, random_state=42)  # Use a random state

    transformed_training_labels = {}
    transformed_testing_labels = {}

    # Process training images
    for orig_image_name, points in tqdm(training_data.items(), desc="Creating Training Images"):
        orig_image = cv2.imread(os.path.join(original_img_path, orig_image_name))

        # Save original training image
        transformed_training_labels[orig_image_name] = points
        cv2.imwrite(os.path.join(training_save_path, orig_image_name), orig_image)

        # Apply and save transformed training image
        transformed_image, rotated_points = random_transform(orig_image, points[0])
        transformed_img_name = "transformed_" + orig_image_name
        transformed_training_labels[transformed_img_name] = [rotated_points]  # Keep as a list of lists
        cv2.imwrite(os.path.join(training_save_path, transformed_img_name), transformed_image)

    # Process testing images
    for orig_image_name, points in tqdm(testing_data.items(), desc="Creating Testing Images"):
        orig_image = cv2.imread(os.path.join(original_img_path, orig_image_name))

        # Save original test image
        transformed_testing_labels[orig_image_name] = points
        cv2.imwrite(os.path.join(testing_save_path, orig_image_name), orig_image)

        # Apply and save transformed testing image
        transformed_image, rotated_points = random_transform(orig_image, points[0])
        transformed_img_name = "transformed_" + orig_image_name
        transformed_testing_labels[transformed_img_name] = [rotated_points]  # Keep as a list of lists
        cv2.imwrite(os.path.join(testing_save_path, transformed_img_name), transformed_image)

    # Save the updated datasets
    with open('transformed_training_GT_labels_v2.json', 'w') as f:
        json.dump(transformed_training_labels, f)

    with open('transformed_testing_GT_labels_v2.json', 'w') as f:
        json.dump(transformed_testing_labels, f)


if __name__ == '__main__':
    main()
