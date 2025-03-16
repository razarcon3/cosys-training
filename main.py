import json
import os
import random
import time

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import math
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from typing import Dict
from tqdm import tqdm

IMAGES_FOLDER = "alphapilot_extended_imgs"
LABELS_FILE = "transformed_training_GT_labels_v2.json"
VISUALIZE_LABELS = False


# Custom Dataset Class
class DroneGateDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        self.img_dir = img_dir
        self.labels = json.load(open(labels_file))
        self.img_names = list(self.labels.keys())
        self.transform = transform

        # filter out images w/o corners
        self.valid_img_names = []
        for img_name in self.img_names:
            corners_data = self.labels[img_name]
            if corners_data and len(corners_data[0]) >= 2:  # Check for valid data
                self.valid_img_names.append(img_name)
        self.img_names = self.valid_img_names

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        gate_corners = self.labels[img_name]

        # Convert corners to bounding boxes
        bboxes = []
        labels = []
        for gate in gate_corners:
            # Get corner coordinates
            x1 = [gate[i] for i in range(0, len(gate), 2)]
            y1 = [gate[i] for i in range(1, len(gate), 2)]

            # Calculate side lengths
            side_lengths = [
                math.dist((x1[0], y1[0]), (x1[1], y1[1])),  # Side 1
                math.dist((x1[1], y1[1]), (x1[2], y1[2])),  # Side 2
                math.dist((x1[2], y1[2]), (x1[3], y1[3])),  # Side 3
                math.dist((x1[3], y1[3]), (x1[0], y1[0]))  # Side 4
            ]
            avg_side_length = sum(side_lengths) / len(side_lengths)

            bbox_size = int(avg_side_length * 0.55)  # Bounding box size as a fraction of the average side length

            for x, y in zip(x1, y1):
                x_min = int(x - bbox_size // 2)
                y_min = int(y - bbox_size // 2)
                x_max = int(x + bbox_size // 2)
                y_max = int(y + bbox_size // 2)
                bboxes.append([x_min, y_min, x_max, y_max])
                labels.append(1)

        # Create target dictionary
        target = {"boxes": torch.tensor(bboxes, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.int64), "corners": gate_corners}

        image = transforms.ToTensor()(image)

        return image, target


# --- Visualization Function ---
def visualize_labels(image: np.ndarray, target: Dict):
    """
    Draws bounding boxes on the image.

    Args:
        :param target:
        :param image:
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    bboxes = np.array(target["boxes"].tolist()).astype(int).tolist()
    corners = target["corners"]

    for corner_bbox in bboxes:
        x_min, y_min, x_max, y_max = map(int, corner_bbox)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    for gate in corners:
        # Draw each corner as a circle
        for i in range(0, len(gate), 2):
            x, y = int(gate[i]), int(gate[i + 1])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Green circles

        # Optionally, connect the corners with lines to visualize the gate outline
        for i in range(0, len(gate), 2):
            cv2.line(image, (gate[i], gate[i + 1]),
                     (gate[(i + 2) % len(gate)], gate[(i + 3) % len(gate)]), (255, 0, 0), 2)  # Blue lines

    cv2.imshow("Image with Bounding Boxes and Corners", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def train(dataset):
    # --- Splitting the Dataset ---
    indices = torch.randperm(len(dataset)).tolist()
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)  # 80% for training, 20% for validation
    valid_size = dataset_size - train_size  # Explicitly calculate valid_size

    train_dataset = torch.utils.data.Subset(dataset, indices[:train_size])
    valid_dataset = torch.utils.data.Subset(dataset, indices[train_size:])

    # --- Data Loaders ---
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True,
                              collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(valid_dataset, batch_size=5, shuffle=False,  # No need to shuffle validation data
                              collate_fn=lambda x: tuple(zip(*x)))

    # --- Model Setup ---
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    num_classes = 2  # 1 class (gate_corner) + background.  MUST be 2.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # --- Device (GPU/CPU) ---
    device = torch.device('cuda')
    model.to(device)

    # --- Optimizer and Scheduler ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.01, weight_decay=0.0005)  # Added weight decay
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # --- Training Loop ---
    num_epochs = 10
    for epoch in range(num_epochs):
        start_time = time.time()  # Time the epoch
        model.train()
        train_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            images = list(image.to(device) for image in images)
            # --- Correct Target Handling ---
            targets = [{k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()
            train_loss += losses.item() * len(images)  # Scale by batch size

        train_loss /= len(train_dataset)  # Average loss per sample
        epoch_time = time.time() - start_time

        # # --- Validation Loop ---
        # model.eval()  # Set the model to evaluation mode
        # valid_loss = 0.0
        # with torch.no_grad():  # Disable gradient calculations during validation
        #     for images, targets in valid_loader:
        #         images = list(image.to(device) for image in images)
        #         targets = [{k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets]
        #
        #         loss_dict = model(images, targets)
        #         losses = sum(loss for loss in loss_dict.values())
        #         valid_loss += losses.item() * len(images)
        #
        # valid_loss /= len(valid_dataset)
        lr_scheduler.step() # Step the scheduler after each epoch.

        # print(f'Epoch: {epoch + 1}/{num_epochs}, '
        #       f'Train Loss: {train_loss:.4f}, '
        #       f'Valid Loss: {valid_loss:.4f}, '
        #       f'Time: {epoch_time:.2f}s, '
        #       f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        print(f'Epoch: {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Time: {epoch_time:.2f}s, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

    torch.save(model.state_dict(), "model.pth")
    print("Training complete!")
def main():
    # Define dataset and dataloader
    img_dir = os.path.join(os.getcwd(), IMAGES_FOLDER)

    dataset = DroneGateDataset(img_dir, LABELS_FILE)

    if VISUALIZE_LABELS:
        # Visualize some random images
        num_visualize = 100
        for _ in range(num_visualize):
            random_index = random.randint(0, len(dataset) - 1)
            image, target = dataset[random_index]

            img_np = image.permute(1, 2, 0).numpy() * 255
            img_np = img_np.astype(np.uint8)
            visualize_labels(img_np, target)

    train(dataset)
# --- Main Execution ---
if __name__ == '__main__':
    main()
