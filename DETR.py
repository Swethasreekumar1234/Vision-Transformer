# ============================================================
# DETR - Detection Transformer for Sunspot Detection
# ============================================================
# DETR (Detection Transformer) was the first model to apply
# transformers directly to object detection end-to-end.
# Unlike YOLO which divides the image into a grid,
# DETR uses a CNN backbone to extract features, then passes
# them through a Transformer encoder-decoder.
# The decoder outputs a fixed set of predictions (100 boxes)
# and uses Hungarian matching to assign predictions to targets.
# This means NO anchor boxes and NO non-maximum suppression.
# ============================================================

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import numpy as np
from transformers import AutoImageProcessor, DetrForObjectDetection
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

# path to your dataset folder
DATA_DIR = '/content'

# number of full passes through training data
EPOCHS = 10

# number of images processed together in one step
BATCH_SIZE = 2

# image size — matches your Roboflow export size
IMG_SIZE = 1024

# learning rate: how large each weight update step is
LEARNING_RATE = 1e-4

# class names in your dataset
CLASSES = ['Sunspots']

# number of classes
NUM_CLASSES = len(CLASSES)  # 1

# folder to save metrics and model weights
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# use GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# STEP 1 — DATASET
# Custom dataset class that loads images and YOLO labels
# ============================================================

class SunspotDataset(Dataset):
    # split: 'train', 'valid', or 'test'
    # processor: DETR's image preprocessor
    def __init__(self, split, processor):
        self.split     = split
        self.processor = processor
        self.img_dir   = os.path.join(DATA_DIR, split,'images')
        self.label_dir = os.path.join(DATA_DIR,split,'labels')

        # check the folder exists
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Directory not found: {self.img_dir}")

        # collect all jpg filenames in this folder
        self.images = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        print(f"{split}: found {len(self.images)} images")

    # total number of images in this split
    def __len__(self):
        return len(self.images)

    # called every time PyTorch needs one image + label pair
    # idx: index of which image to load
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # open image as RGB (3 color channels)
        image = Image.open(img_path).convert('RGB')

        # find the matching YOLO label file (.txt)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')

        boxes  = []
        labels = []

        # read YOLO format labels line by line
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    # skip empty lines (background images)
                    if not line.strip():
                        continue

                    # YOLO format: class_id cx cy w h (normalized 0-1)
                    parts    = line.strip().split()
                    class_id = int(parts[0])
                    cx       = float(parts[1])  # center x
                    cy       = float(parts[2])  # center y
                    w        = float(parts[3])  # width
                    h        = float(parts[4])  # height

                    # convert normalized YOLO to absolute pixel coordinates
                    # multiply by IMG_SIZE to get actual pixel values
                    xmin = (cx - w / 2) * IMG_SIZE
                    ymin = (cy - h / 2) * IMG_SIZE
                    xmax = (cx + w / 2) * IMG_SIZE
                    ymax = (cy + h / 2) * IMG_SIZE

                    # only add valid boxes (non-zero area)
                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(class_id)

        # if no annotations, use dummy box (background image)
        if len(boxes) == 0:
            boxes  = [[0.0, 0.0, 1.0, 1.0]]
            labels = [0]

        # build COCO-style annotation dictionary
        # this is the exact format DETR's processor expects
        annotation_dict = {
            # image_id must be an integer
            "image_id": int(idx),
            "annotations": [
                {
                    # category_id: which class (0 = Sunspots)
                    "category_id": int(l),
                    # COCO bbox format: [xmin, ymin, width, height]
                    # NOT xmax ymax — COCO uses width/height
                    "bbox": [
                        float(b[0]),
                        float(b[1]),
                        float(b[2] - b[0]),   # width  = xmax - xmin
                        float(b[3] - b[1])    # height = ymax - ymin
                    ],
                    # area of the box in pixels
                    "area": float((b[2] - b[0]) * (b[3] - b[1])),
                    # iscrowd=0: individual object, not a crowd
                    "iscrowd": 0
                }
                for b, l in zip(boxes, labels)
            ]
        }

        # processor converts image to normalized tensor
        # wrap annotation_dict in a list — processor expects a list
        encoding     = self.processor(images=image, annotations=[annotation_dict], return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)  # remove batch dim: [1,3,H,W] → [3,H,W]
        target       = encoding["labels"][0]                # get label dict for this image

        return pixel_values, target


# ============================================================
# STEP 2 — COLLATE FUNCTION
# Combines individual samples into batches for the DataLoader
# ============================================================

def collate_fn(batch):
    # batch is a list of (pixel_values, target) tuples
    # stack all images into one tensor: [batch_size, 3, H, W]
    pixel_values = torch.stack([item[0] for item in batch], dim=0)
    # targets remain as a list of dicts (variable number of boxes per image)
    targets      = [item[1] for item in batch]
    return pixel_values, targets


# ============================================================
# STEP 3 — LOAD DETR MODEL
# ============================================================

print("Loading DETR model...")

# AutoImageProcessor handles all preprocessing for DETR:
# resizing, normalization, converting to tensor
processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

# DETR with ResNet-50 backbone, pretrained on COCO dataset
# num_labels=1 because we only have one class (Sunspots)
# ignore_mismatched_sizes=True: replaces the COCO head (80 classes)
# with a new head for our single class
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)

# move model to GPU or CPU
model.to(device)
for name, param in model.named_parameters():
    if 'backbone' in name:
        param.requires_grad = False

# ============================================================
# STEP 4 — CREATE DATASETS AND DATALOADERS
# ============================================================

print("Loading datasets...")
train_dataset = SunspotDataset('train', processor)
val_dataset   = SunspotDataset('valid', processor)
test_dataset  = SunspotDataset('test',  processor)

# DataLoader batches the data
# shuffle=True randomizes order each epoch (important for training)
# shuffle=False keeps order for validation and testing
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ============================================================
# STEP 5 — OPTIMIZER AND SCHEDULER
# ============================================================

# AdamW: standard optimizer for transformers
# combines Adam with weight decay to prevent overfitting
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# ReduceLROnPlateau: if val loss stops improving for 3 epochs,
# reduce learning rate by half
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5,)

# ============================================================
# STEP 6 — TRAINING LOOP WITH VALIDATION
# ============================================================

print(f"Starting DETR training for {EPOCHS} epochs...")

train_losses = []   # track training loss per epoch
val_losses   = []   # track validation loss per epoch
best_loss    = float('inf')  # track best model

for epoch in range(EPOCHS):

    # ---- TRAINING PHASE ----
    # model.train() enables:
    # - dropout (randomly zeros neurons to prevent overfitting)
    # - batch normalization updates
    model.train()
    epoch_train_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

    for pixel_values, targets in loop:
        # move data to same device as model (GPU or CPU)
        pixel_values = pixel_values.to(device)
        # move each target dict's tensors to device
        targets      = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # zero gradients — must clear before each backward pass
        # otherwise gradients from previous batch accumulate
        optimizer.zero_grad()

        # forward pass: feed images and labels through model
        # DETR computes loss internally using Hungarian matching
        outputs = model(pixel_values=pixel_values, labels=targets)

        # DETR loss = classification loss + bounding box L1 loss + GIoU loss
        loss = outputs.loss

        # backward pass: compute gradients for all parameters
        loss.backward()

        # gradient clipping: caps gradient magnitude to prevent
        # large unstable weight updates
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # update weights using computed gradients
        optimizer.step()

        epoch_train_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ---- VALIDATION PHASE ----
    # model.eval() disables dropout and freezes batch norm
    model.eval()
    epoch_val_loss = 0

    # torch.no_grad() disables gradient computation
    # saves memory and speeds up validation
    with torch.no_grad():
        for pixel_values, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            pixel_values   = pixel_values.to(device)
            targets        = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs        = model(pixel_values=pixel_values, labels=targets)
            epoch_val_loss += outputs.loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

    # update learning rate based on validation loss
    scheduler.step(avg_val_loss)

    # save model if this is the best validation loss so far
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), f"{RESULTS_DIR}/best_detr_sunspots.pt")
        print(f"  --> New best model saved (val_loss={best_loss:.4f})")


# ============================================================
# STEP 7 — TESTING
# Run on held-out test set after all training is complete
# ============================================================

print("\nRunning on test set...")

# load the best saved model weights for testing
model.load_state_dict(torch.load(f"{RESULTS_DIR}/best_detr_sunspots.pt", map_location=device))
model.eval()

all_predictions   = []
all_targets_boxes = []

with torch.no_grad():
    for pixel_values, targets in tqdm(test_loader, desc="Testing"):
        pixel_values = pixel_values.to(device)

        # forward pass without labels = inference only (no loss)
        outputs = model(pixel_values=pixel_values)

        # post_process_object_detection converts raw model outputs
        # to human-readable boxes + scores + labels
        # threshold=0.5: only keep detections with >50% confidence
        results = processor.post_process_object_detection(
            outputs,
            threshold=0.5,
            target_sizes=[(IMG_SIZE, IMG_SIZE)] * pixel_values.shape[0]
        )

        all_predictions.extend(results)
        all_targets_boxes.extend(targets)


# ============================================================
# STEP 8 — EVALUATION METRICS
# ============================================================

def compute_iou(box1, box2):
    # compute Intersection over Union between two boxes
    # box format: [xmin, ymin, xmax, ymax]

    # find the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # intersection area (0 if no overlap)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # area of each box individually
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # union area = sum of both areas minus the overlap
    union = area1 + area2 - intersection

    # avoid division by zero
    return intersection / union if union > 0 else 0.0


def compute_metrics(predictions, targets, iou_threshold=0.5):
    # compute precision, recall, F1 across all test images
    total_tp = 0  # true positives:  predicted box matches a real sunspot
    total_fp = 0  # false positives: predicted box has no matching real sunspot
    total_fn = 0  # false negatives: real sunspot was not detected

    for pred, tgt in zip(predictions, targets):
        # get predicted boxes (move to CPU for numpy operations)
        pred_boxes = pred['boxes'].cpu().numpy() if len(pred['boxes']) > 0 else []

        # get ground truth boxes
        gt_boxes   = tgt['boxes'].cpu().numpy() if 'boxes' in tgt else []

        # track which ground truth boxes have already been matched
        matched_gt = set()

        for pb in pred_boxes:
            matched = False
            for j, gb in enumerate(gt_boxes):
                # skip already matched ground truth boxes
                if j in matched_gt:
                    continue
                # check if IoU exceeds threshold
                if compute_iou(pb, gb) >= iou_threshold:
                    total_tp  += 1      # correct detection
                    matched_gt.add(j)   # mark this GT as matched
                    matched    = True
                    break
            if not matched:
                total_fp += 1  # prediction had no matching GT

        # any unmatched GT boxes = missed sunspots
        total_fn += len(gt_boxes) - len(matched_gt)

    # precision: of all detections made, how many were correct?
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0

    # recall: of all real sunspots, how many did we find?
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    # F1: harmonic mean of precision and recall
    # balances both — useful single number to compare models
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': round(precision, 4),
        'recall':    round(recall,    4),
        'f1':        round(f1,        4),
        'tp':        total_tp,
        'fp':        total_fp,
        'fn':        total_fn
    }


# compute metrics on test set predictions
metrics = compute_metrics(all_predictions, all_targets_boxes)

print("\n========== DETR RESULTS ==========")
print(f"Precision : {metrics['precision']:.4f}  (of all detections, how many were correct)")
print(f"Recall    : {metrics['recall']:.4f}  (of all sunspots, how many were found)")
print(f"F1 Score  : {metrics['f1']:.4f}  (balance of precision and recall)")
print(f"True Positives  : {metrics['tp']}  (correctly detected sunspots)")
print(f"False Positives : {metrics['fp']}  (wrong detections)")
print(f"False Negatives : {metrics['fn']}  (missed sunspots)")
print(f"Best Val Loss   : {best_loss:.4f}")
print("===================================\n")


# ============================================================
# STEP 9 — SAVE ALL RESULTS
# ============================================================

final_results = {
    'model':            'DETR',
    'epochs':           EPOCHS,
    'best_val_loss':    best_loss,
    'final_train_loss': train_losses[-1],
    'final_val_loss':   val_losses[-1],
    'precision':        metrics['precision'],
    'recall':           metrics['recall'],
    'f1':               metrics['f1'],
    'tp':               metrics['tp'],
    'fp':               metrics['fp'],
    'fn':               metrics['fn'],
    'train_losses':     train_losses,
    'val_losses':       val_losses
}

# save to JSON so compare_models.py can read it
with open(f'{RESULTS_DIR}/DETR_results.json', 'w') as f:
    json.dump(final_results, f, indent=4)

print("Results saved to results/DETR_results.json")
print("DETR training complete!")
