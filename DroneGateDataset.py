import json
import os
from PIL import Image
import math

import torch
from torch.utils.data import Dataset
from torchvision import transforms


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
        target = {"boxes": torch.tensor(bboxes, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.int64),
                  "corners": gate_corners}

        image = transforms.ToTensor()(image)

        return image, target
