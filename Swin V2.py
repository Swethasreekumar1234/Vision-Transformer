# ============================================================
# Swin V2 - Shifted Window Transformer for Sunspot Detection
# ============================================================
# Swin V2 is a hierarchical Vision Transformer that uses
# shifted windows instead of global attention.
# Unlike DETR which uses a CNN backbone + transformer decoder,
# Swin V2 IS the backbone — it processes the image in patches
# and produces feature maps at multiple scales.
# For object detection we attach a detection head on top:
# we use Faster R-CNN style ROI-based detection via
# torchvision's FasterRCNN with Swin V2 as the backbone.
# This means: region proposals → ROI pooling → classify + refine.
# ============================================================

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import numpy as np
from tqdm import tqdm

# transformers gives us the Swin V2 backbone
from transformers import AutoImageProcessor, Swinv2Model

# torchvision gives us the detection head (Faster R-CNN components)
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR      = '/content'         # root folder with train/ valid/ test/
EPOCHS        = 30                 # number of full passes through training data
BATCH_SIZE    = 2                  # images per batch — keep low for 1024px images
IMG_SIZE      = 1024               # must match your Roboflow export size
LEARNING_RATE = 1e-4               # standard fine-tuning LR for transformers
CLASSES       = ['Sunspots']       # your object categories
NUM_CLASSES   = len(CLASSES) + 1   # +1 for background class (Faster R-CNN requires this)
RESULTS_DIR   = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================
# STEP 1 — DATASET
# Same YOLO label reading as DETR, but targets formatted
# for Faster R-CNN instead of DETR's processor
# ============================================================

class SunspotDataset(Dataset):
    def __init__(self, split):
        self.split   = split
        self.img_dir = os.path.join(DATA_DIR, split,'images')
        self.label_dir=os.path.join(DATA_DIR,split,'labels')

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Directory not found: {self.img_dir}")

        self.images = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        print(f"{split}: found {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # open image and convert to tensor manually
        # torchvision detection models expect float tensors in [0, 1]
        image = Image.open(img_path).convert('RGB')

        # convert PIL image to tensor: [C, H, W] with values in [0.0, 1.0]
        image_tensor = torchvision.transforms.functional.to_tensor(image)

        # find matching YOLO label file
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')

        boxes  = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue

                    parts    = line.strip().split()
                    class_id = int(parts[0])
                    cx       = float(parts[1])
                    cy       = float(parts[2])
                    w        = float(parts[3])
                    h        = float(parts[4])

                    # convert normalized YOLO → absolute pixel coordinates
                    xmin = (cx - w / 2) * IMG_SIZE
                    ymin = (cy - h / 2) * IMG_SIZE
                    xmax = (cx + w / 2) * IMG_SIZE
                    ymax = (cy + h / 2) * IMG_SIZE

                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        # Faster R-CNN labels: 0 = background, 1+ = objects
                        # so sunspot class_id 0 becomes label 1
                        labels.append(class_id + 1)

        # Faster R-CNN requires at least one box — use dummy for background images
        if len(boxes) == 0:
            boxes  = [[0.0, 0.0, 1.0, 1.0]]
            labels = [0]  # label 0 = background

        # convert to tensors — Faster R-CNN expects float boxes, long labels
        boxes_tensor  = torch.tensor(boxes,  dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        # build target dict in the format torchvision detection expects
        target = {
            'boxes':  boxes_tensor,   # [N, 4] — xmin ymin xmax ymax
            'labels': labels_tensor,  # [N]    — integer class ids
        }

        return image_tensor, target


# ============================================================
# STEP 2 — COLLATE FUNCTION
# Same reason as DETR: variable number of boxes per image
# means we can't stack targets into a single tensor
# ============================================================

def collate_fn(batch):
    images  = [item[0] for item in batch]   # list of [C, H, W] tensors
    targets = [item[1] for item in batch]   # list of dicts
    return images, targets


# ============================================================
# STEP 3 — BUILD SWIN V2 + FASTER R-CNN MODEL
# Swin V2 is the backbone; Faster R-CNN is the detection head
# ============================================================

print("Loading Swin V2 model...")

# ---- 3a. Load Swin V2 backbone from Hugging Face ----
# microsoft/swinv2-tiny-patch4-window8-256:
#   tiny  = smallest variant, fits in Colab GPU memory
#   patch4 = each patch is 4×4 pixels
#   window8 = attention window size is 8×8 patches
#   256 = pretrained at 256×256 resolution
swin_backbone = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

# ---- 3b. Wrap backbone for torchvision compatibility ----
# torchvision's FasterRCNN expects a backbone with:
#   - a forward() method that returns a dict of feature maps
#   - an out_channels attribute telling it the feature dimension
class SwinV2Backbone(nn.Module):
    def __init__(self, swin_model):
        super().__init__()
        self.swin  = swin_model
        # Swin V2 tiny outputs 768-dim features at the final stage
        self.out_channels = 768

    def forward(self, x):
        # x: [B, 3, H, W] — batch of images
        # Swin V2 forward returns an object with last_hidden_state
        outputs = self.swin(pixel_values=x)

        # last_hidden_state: [B, num_patches, hidden_dim]
        # e.g. for 1024×1024 input with patch4: num_patches = (1024/32)^2 = 1024
        features = outputs.last_hidden_state

        B, N, C = features.shape

        # reshape from sequence → spatial grid for ROI pooling
        # Swin V2 downsamples by 32× total, so spatial size = IMG_SIZE // 32
        H = W = int(N ** 0.5)
        # [B, N, C] → [B, C, H, W]
        features = features.permute(0, 2, 1).reshape(B, C, H, W)

        # FasterRCNN expects a dict of feature maps (like FPN levels)
        # we only have one level here — key '0' is conventional
        return {'0': features}

backbone = SwinV2Backbone(swin_backbone)

# ---- 3c. Freeze Swin V2 weights ----
# same reason as DETR: small dataset, freezing backbone
# prevents overfitting and speeds up training
for param in backbone.swin.parameters():
    param.requires_grad = False

# ---- 3d. Anchor generator ----
# RPN (Region Proposal Network) proposes candidate boxes
# These anchor sizes and ratios cover small sunspots well
# Sizes in pixels — sunspots are typically small
anchor_generator = AnchorGenerator(
    sizes=((16, 32, 64, 128, 256),),   # 5 anchor sizes
    aspect_ratios=((0.5, 1.0, 2.0),)   # 3 aspect ratios → 15 anchors per location
)

# ---- 3e. ROI Align ----
# After RPN proposes regions, ROI Align crops + resizes feature map
# regions to a fixed size (7×7) for the classification head
roi_pooler = MultiScaleRoIAlign(
    featmap_names=['0'],   # which feature map levels to use
    output_size=7,         # output spatial size after pooling
    sampling_ratio=2       # number of sampling points per bin
)

# ---- 3f. Assemble full Faster R-CNN model ----
model = FasterRCNN(
    backbone=backbone,
    num_classes=NUM_CLASSES,          # 2: background + sunspot
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler,
    # reduce proposals for memory efficiency with large images
    rpn_pre_nms_top_n_train=1000,
    rpn_pre_nms_top_n_test=500,
    rpn_post_nms_top_n_train=500,
    rpn_post_nms_top_n_test=300,
    rpn_score_thresh=0.0,
)

model.to(device)
print("Model loaded successfully.")


# ============================================================
# STEP 4 — DATASETS AND DATALOADERS
# ============================================================

print("Loading datasets...")
train_dataset = SunspotDataset('train')
val_dataset   = SunspotDataset('valid')
test_dataset  = SunspotDataset('test')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


# ============================================================
# STEP 5 — OPTIMIZER AND SCHEDULER
# Only optimize parameters that require gradients
# (backbone is frozen so its params are excluded)
# ============================================================

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5
)


# ============================================================
# STEP 6 — TRAINING LOOP
# Faster R-CNN computes loss differently from DETR:
# it returns a dict of losses (rpn + roi) that you sum manually
# ============================================================

print(f"Starting Swin V2 training for {EPOCHS} epochs...")

train_losses = []
val_losses   = []
best_loss    = float('inf')

for epoch in range(EPOCHS):

    # ---- TRAINING ----
    model.train()
    epoch_train_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

    for images, targets in loop:
        # move images to device — images is a list of tensors
        images  = [img.to(device) for img in images]
        # move each target dict's tensors to device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # Faster R-CNN forward with targets returns a dict of losses:
        # loss_classifier   — classification loss for ROI head
        # loss_box_reg      — bounding box regression loss for ROI head
        # loss_objectness   — RPN objectness loss (is there an object here?)
        # loss_rpn_box_reg  — RPN box regression loss
        loss_dict = model(images, targets)

        # sum all losses into one number
        loss = sum(loss for loss in loss_dict.values())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_train_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ---- VALIDATION ----
    # Faster R-CNN is unusual: model.eval() switches to inference mode
    # and stops returning losses — it returns predictions instead.
    # To get validation loss we keep model.train() but use torch.no_grad()
    model.train()
    epoch_val_loss = 0

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict      = model(images, targets)
            loss           = sum(loss for loss in loss_dict.values())
            epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), f"{RESULTS_DIR}/best_swinv2_sunspots.pt")
        print(f"  --> New best model saved (val_loss={best_loss:.4f})")


# ============================================================
# STEP 7 — TESTING
# ============================================================

print("\nRunning on test set...")

model.load_state_dict(torch.load(f"{RESULTS_DIR}/best_swinv2_sunspots.pt", map_location=device))

# model.eval() for inference — returns predictions not losses
model.eval()

all_predictions   = []
all_targets_boxes = []

with torch.no_grad():
    for images, targets in tqdm(test_loader, desc="Testing"):
        images = [img.to(device) for img in images]

        # inference: returns list of dicts with 'boxes', 'labels', 'scores'
        predictions = model(images)

        # filter by confidence threshold
        filtered = []
        for pred in predictions:
            keep = pred['scores'] >= 0.5
            filtered.append({
                'boxes':  pred['boxes'][keep],
                'labels': pred['labels'][keep],
                'scores': pred['scores'][keep],
            })

        all_predictions.extend(filtered)
        all_targets_boxes.extend(targets)


# ============================================================
# STEP 8 — EVALUATION METRICS
# Identical to DETR — IoU-based precision/recall/F1
# ============================================================

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def compute_metrics(predictions, targets, iou_threshold=0.5):
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, tgt in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu().numpy() if len(pred['boxes']) > 0 else []
        gt_boxes   = tgt['boxes'].cpu().numpy()  if len(tgt['boxes']) > 0 else []

        # skip dummy background boxes (the [0,0,1,1] placeholders)
        gt_boxes = [b for b in gt_boxes if not (b[0] == 0 and b[1] == 0 and b[2] == 1 and b[3] == 1)]

        matched_gt = set()

        for pb in pred_boxes:
            matched = False
            for j, gb in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                if compute_iou(pb, gb) >= iou_threshold:
                    total_tp  += 1
                    matched_gt.add(j)
                    matched    = True
                    break
            if not matched:
                total_fp += 1

        total_fn += len(gt_boxes) - len(matched_gt)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': round(precision, 4),
        'recall':    round(recall,    4),
        'f1':        round(f1,        4),
        'tp':        total_tp,
        'fp':        total_fp,
        'fn':        total_fn
    }


metrics = compute_metrics(all_predictions, all_targets_boxes)

print("\n========== SWIN V2 RESULTS ==========")
print(f"Precision : {metrics['precision']:.4f}  (of all detections, how many were correct)")
print(f"Recall    : {metrics['recall']:.4f}  (of all sunspots, how many were found)")
print(f"F1 Score  : {metrics['f1']:.4f}  (balance of precision and recall)")
print(f"True Positives  : {metrics['tp']}  (correctly detected sunspots)")
print(f"False Positives : {metrics['fp']}  (wrong detections)")
print(f"False Negatives : {metrics['fn']}  (missed sunspots)")
print(f"Best Val Loss   : {best_loss:.4f}")
print("=====================================\n")


# ============================================================
# STEP 9 — SAVE RESULTS
# Same format as DETR_results.json for compare_models.py
# ============================================================

final_results = {
    'model':            'SwinV2',
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

with open(f'{RESULTS_DIR}/SwinV2_results.json', 'w') as f:
    json.dump(final_results, f, indent=4)

print("Results saved to results/SwinV2_results.json")
print("Swin V2 training complete!")
