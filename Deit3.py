# ============================================================
# DeiT III - Data-efficient Image Transformers V3
# ============================================================
# DeiT III is a refined training recipe for Vision Transformers.
# The original DeiT used knowledge distillation from a CNN teacher
# to make ViT train well on smaller datasets.
# DeiT III (2022) drops the distillation and instead uses:
#   - 3-Augment: a strong but simple augmentation strategy
#   - LayerScale: per-layer learnable scaling for stability
#   - Better regularization (stochastic depth, weight decay tuning)
# The result: a plain ViT trained with DeiT III recipe outperforms
# heavily engineered architectures.
# For detection: DeiT III backbone → Faster R-CNN head → boxes
# ============================================================

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import numpy as np
from tqdm import tqdm

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

# timm (PyTorch Image Models) has the best DeiT III implementation
# install if needed: pip install timm
import timm

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR      = '/content'
EPOCHS        = 30
BATCH_SIZE    = 2
IMG_SIZE      = 1024
LEARNING_RATE = 1e-4
CLASSES       = ['Sunspots']
NUM_CLASSES   = len(CLASSES) + 1   # +1 for background
RESULTS_DIR   = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================
# STEP 1 — DATASET
# ============================================================

class SunspotDataset(Dataset):
    def __init__(self, split):
        self.split   = split
        self.img_dir = os.path.join(DATA_DIR, split)

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Directory not found: {self.img_dir}")

        self.images = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        print(f"{split}: found {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image        = Image.open(img_path).convert('RGB')
        image_tensor = torchvision.transforms.functional.to_tensor(image)

        label_path = os.path.join(self.img_dir, os.path.splitext(img_name)[0] + '.txt')

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

                    xmin = (cx - w / 2) * IMG_SIZE
                    ymin = (cy - h / 2) * IMG_SIZE
                    xmax = (cx + w / 2) * IMG_SIZE
                    ymax = (cy + h / 2) * IMG_SIZE

                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(class_id + 1)

        if len(boxes) == 0:
            boxes  = [[0.0, 0.0, 1.0, 1.0]]
            labels = [0]

        boxes_tensor  = torch.tensor(boxes,  dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes':  boxes_tensor,
            'labels': labels_tensor,
        }

        return image_tensor, target


# ============================================================
# STEP 2 — COLLATE FUNCTION
# ============================================================

def collate_fn(batch):
    images  = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ============================================================
# STEP 3 — BUILD DeiT III + FASTER R-CNN MODEL
# ============================================================

print("Loading DeiT III model...")

# ---- 3a. Load DeiT III backbone from timm ----
# deit3_small_patch16_224:
#   small   = small variant (~22M params) — fits Colab memory
#   patch16 = each patch is 16×16 pixels
#   224     = pretrained at 224×224 resolution
# pretrained=True downloads ImageNet-pretrained weights
# features_only=True tells timm to return intermediate feature maps
# instead of the final classification logit
deit3 = timm.create_model(
    'deit3_small_patch16_224',
    pretrained=True,
    features_only=False   # we handle feature extraction manually
)

# remove the classification head — we only want the feature extractor
# timm models have a 'head' attribute for the final linear layer
deit3.head = nn.Identity()  # replace with identity (pass-through)

# ---- 3b. Wrap DeiT III for torchvision compatibility ----
class DeiT3Backbone(nn.Module):
    def __init__(self, deit_model):
        super().__init__()
        self.deit = deit_model

        # DeiT III small hidden dimension is 384
        self.out_channels = 384

        self.patch_size = 16
        self.img_size   = IMG_SIZE

    def forward(self, x):
        # x: [B, 3, H, W]
        B = x.shape[0]

        # DeiT III forward returns the full sequence including CLS token
        # shape: [B, num_patches + 1, 384]
        # for 1024×1024 with patch16: num_patches = (1024/16)^2 = 4096
        tokens = self.deit.forward_features(x)

        # remove CLS token at position 0 — keep patch tokens only
        patch_tokens = tokens[:, 1:, :]   # [B, num_patches, 384]

        num_patches = patch_tokens.shape[1]
        H = W = int(num_patches ** 0.5)

        # [B, N, C] → [B, C, H, W]
        features = patch_tokens.permute(0, 2, 1).reshape(B, 384, H, W)

        return {'0': features}

# ---- 3c. Add ImageNet normalization ----
# DeiT III expects ImageNet-normalized inputs
normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

class NormalizedDeiT3Backbone(nn.Module):
    def __init__(self, backbone, normalize):
        super().__init__()
        self.backbone     = backbone
        self.normalize    = normalize
        self.out_channels = backbone.out_channels

    def forward(self, x):
        x = torch.stack([self.normalize(img) for img in x])
        return self.backbone(x)

raw_backbone        = DeiT3Backbone(deit3)
normalized_backbone = NormalizedDeiT3Backbone(raw_backbone, normalize)

# ---- 3d. Freeze DeiT III weights ----
for param in normalized_backbone.backbone.deit.parameters():
    param.requires_grad = False

# ---- 3e. Anchor generator ----
anchor_generator = AnchorGenerator(
    sizes=((16, 32, 64, 128, 256),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)

# ---- 3f. ROI Align ----
roi_pooler = MultiScaleRoIAlign(
    featmap_names=['0'],
    output_size=7,
    sampling_ratio=2
)

# ---- 3g. Assemble Faster R-CNN ----
model = FasterRCNN(
    backbone=normalized_backbone,
    num_classes=NUM_CLASSES,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler,
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
# ============================================================

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5
)


# ============================================================
# STEP 6 — TRAINING LOOP
# ============================================================

print(f"Starting DeiT III training for {EPOCHS} epochs...")

train_losses = []
val_losses   = []
best_loss    = float('inf')

for epoch in range(EPOCHS):

    # ---- TRAINING ----
    model.train()
    epoch_train_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

    for images, targets in loop:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        loss_dict = model(images, targets)
        loss      = sum(l for l in loss_dict.values())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_train_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ---- VALIDATION ----
    model.train()
    epoch_val_loss = 0

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict      = model(images, targets)
            loss           = sum(l for l in loss_dict.values())
            epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), f"{RESULTS_DIR}/best_deit3_sunspots.pt")
        print(f"  --> New best model saved (val_loss={best_loss:.4f})")


# ============================================================
# STEP 7 — TESTING
# ============================================================

print("\nRunning on test set...")

model.load_state_dict(torch.load(f"{RESULTS_DIR}/best_deit3_sunspots.pt", map_location=device))
model.eval()

all_predictions   = []
all_targets_boxes = []

with torch.no_grad():
    for images, targets in tqdm(test_loader, desc="Testing"):
        images = [img.to(device) for img in images]

        predictions = model(images)

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

        gt_boxes = [b for b in gt_boxes if not (b[0]==0 and b[1]==0 and b[2]==1 and b[3]==1)]

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

print("\n========== DeiT III RESULTS ==========")
print(f"Precision : {metrics['precision']:.4f}  (of all detections, how many were correct)")
print(f"Recall    : {metrics['recall']:.4f}  (of all sunspots, how many were found)")
print(f"F1 Score  : {metrics['f1']:.4f}  (balance of precision and recall)")
print(f"True Positives  : {metrics['tp']}  (correctly detected sunspots)")
print(f"False Positives : {metrics['fp']}  (wrong detections)")
print(f"False Negatives : {metrics['fn']}  (missed sunspots)")
print(f"Best Val Loss   : {best_loss:.4f}")
print("=======================================\n")


# ============================================================
# STEP 9 — SAVE RESULTS
# ============================================================

final_results = {
    'model':            'DeiT3',
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

with open(f'{RESULTS_DIR}/DeiT3_results.json', 'w') as f:
    json.dump(final_results, f, indent=4)

print("Results saved to results/DeiT3_results.json")
print("DeiT III training complete!")
