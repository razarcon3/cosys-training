import os
import random
import time

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from typing import Dict
from tqdm import tqdm
from DroneGateDataset import DroneGateDataset

IMAGES_FOLDER = "alphapilot_extended_training_imgs"
LABELS_FILE = "transformed_training_GT_labels_v2.json"
VISUALIZE_LABELS = False

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
    # --- Data Loaders ---
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True,
                              collate_fn=lambda x: tuple(zip(*x)))

    # --- Model Setup ---
    model = fasterrcnn_resnet50_fpn_v2()
    num_classes = 2  # 1 class (gate_corner) + background.  MUST be 2.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # --- Device (GPU/CPU) ---
    device = torch.device('cuda')
    model.to(device)

    # --- Optimizer and Scheduler ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9,
                                weight_decay=0.0005)
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
            train_loss += losses.item()

        train_loss /= len(train_loader)  # Average loss per sample
        epoch_time = time.time() - start_time

        lr_scheduler.step()  # Step the scheduler after each epoch.

        print(f'Epoch: {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Time: {epoch_time:.2f}s, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

    torch.save(model, "full_model.pth")
    torch.save(model.state_dict(), "state_dict_model.pth")
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
