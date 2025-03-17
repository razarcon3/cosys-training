import torch
import cv2
from DroneGateDataset import DroneGateDataset
import numpy as np

MODEL_PATH = "full_model.pth"
TEST_DATASET_IMGS_PATH = "alphapilot_extended_testing_imgs"
TEST_DATASET_LABELS_PATH = "transformed_testing_GT_labels_v2.json"

def visualize_predictions_cv2(image, gt_boxes, pred_boxes, pred_scores=None):
    """Visualizes ground truth and predicted bounding boxes on an image using OpenCV.

    Args:
        image: The input image (as a NumPy array).
        gt_boxes: A list of ground truth bounding boxes.
        pred_boxes: A list of predicted bounding boxes.
        pred_scores: (Optional) A list of prediction scores.
    """
    image = image.copy()  # Create a copy to avoid modifying the original

    # Ensure the image is in uint8 format (OpenCV expects this)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)  # Scale and convert if needed

    # Convert to BGR if it's RGB (OpenCV uses BGR)
    if image.shape[0] == 3: # Check for tensor format
        image = image.transpose(1, 2, 0) # Transpose it if it is a Tensor
    if image.shape[2] == 3:  # if channels last
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



    # Draw ground truth boxes (green)
    for box in gt_boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]  # Ensure integer coordinates
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green

    # Draw predicted boxes (red)
    if pred_boxes is not None:
        for i, box in enumerate(pred_boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red
            if pred_scores is not None:
                cv2.putText(image, f"{pred_scores[i]:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red text

    cv2.imshow("Predictions", image)
    cv2.waitKey(0)  # Wait for any key press
    cv2.destroyAllWindows()

def main():
    # Load the model
    model = torch.load(MODEL_PATH, weights_only=False)
    model.eval()

    # Load the test dataset
    dataset = DroneGateDataset(TEST_DATASET_IMGS_PATH, TEST_DATASET_LABELS_PATH)

    # Set device
    device = torch.device("cuda")
    model.to(device)  # Move the model to the GPU if available

    with torch.no_grad():  # Disable gradient calculations during inference
        for i in range(len(dataset)):
            image, targets = dataset[i]  # targets will be a dictionary that contains "boxes" and "labels"

            # Prepare the image (move to device, add batch dimension)
            image = image.to(device)
            image = image.unsqueeze(0)  # Add batch dimension [C,H,W] -> [1,C,H,W]

            # Make predictions
            predictions = model(image)  # predict

            # Move results to CPU and convert to lists for visualization
            predictions = [{k: v.cpu().numpy() for k, v in p.items()} for p in predictions]
            image = image.squeeze(0)  # Remove batch dimension for visualization

            # Extract bounding boxes and scores (filter by score threshold if needed)
            pred_boxes = predictions[0]['boxes'].tolist()
            pred_scores = predictions[0]['scores'].tolist()
            gt_boxes = targets['boxes'].tolist()

            # apply non-maximum suppression to avoid several boxes for one gate
            selected_indices = cv2.dnn.NMSBoxes(pred_boxes, pred_scores, score_threshold=0.5, nms_threshold=0.8)

            filtered_pred_boxes = []
            filtered_pred_scores = []

            if len(selected_indices) > 0:  # check that selected indices is not empty
                for idx in selected_indices:
                    filtered_pred_boxes.append(pred_boxes[idx])
                    filtered_pred_scores.append(pred_scores[idx])

            # Convert image back to NumPy array if it's a tensor.
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()  # .permute(1, 2, 0)

            visualize_predictions_cv2(image, gt_boxes, filtered_pred_boxes, filtered_pred_scores)


if __name__ == "__main__":
    main()
