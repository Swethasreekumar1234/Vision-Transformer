# ============================================================
# Vanilla ViT - Vision Transformer (Original)
# ============================================================
# ViT (Vision Transformer) was the first paper to apply a
# pure transformer directly to images with no CNN at all.
# The idea: split the image into fixed patches, treat each
# patch like a word token, feed the sequence into a standard
# BERT-style transformer encoder.
# "An Image is Worth 16x16 Words" — Dosovitskiy et al. 2020
#
# Unlike Swin V2 (local windowed attention) or BEiT V2
# (masked image modeling pretraining) or DeiT III (better
# training recipe), Vanilla ViT is the bare original design:
#   - Global self-attention across all patches
#   - No shifted windows, no hierarchical feature maps
#   - No special pretraining strategy
#   - No distillation
#
# This is your baseline transformer — the simplest ViT.
# For detection: ViT backbone → Faster R-CNN head → boxes
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

from transformers import ViTModel

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
# Identical to SwinV2, BEiTV2, DeiT3 — YOLO labels,
# plain tensor output, Faster R-CNN target format
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

        # open image and convert to float tensor in [0, 1]
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

                    # normalized YOLO → absolute pixel coordinates
                    xmin = (cx - w / 2) * IMG_SIZE
                    ymin = (cy - h / 2) * IMG_SIZE
                    xmax = (cx + w / 2) * IMG_SIZE
                    ymax = (cy + h / 2) * IMG_SIZE

                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(class_id + 1)  # +1 for background offset

        # dummy box for background images (no annotations)
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
# STEP 3 — BUILD VANILLA ViT + FASTER R-CNN MODEL
# ============================================================

print("Loading Vanilla ViT model...")

# ---- 3a. Load ViT backbone from Hugging Face ----
# google/vit-base-patch16-224:
#   base    = base-size model (86M parameters)
#   patch16 = each patch is 16×16 pixels
#   224     = pretrained at 224×224 resolution on ImageNet-21k
#
# This is the original ViT-B/16 from the "Image is Worth 16x16 Words" paper.
# No special pretraining trick — just supervised classification on ImageNet.
# This is what makes it "vanilla" compared to the other four models.
#
# add_pooling_layer=False: keep all patch token outputs, not just CLS
# We need the patch tokens to build a spatial feature map for detection
vit_model = ViTModel.from_pretrained(
    "google/vit-base-patch16-224",
    add_pooling_layer=False
)

# ---- 3b. Wrap ViT for torchvision compatibility ----
class ViTBackbone(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

        # ViT base hidden dimension is 768
        # same as BEiT V2 base — both are ViT-Base architecture
        self.out_channels = 768

        # ViT uses 16×16 patches
        # total downsampling factor = 16
        # so for 1024×1024 input: feature map = 1024/16 = 64×64
        self.patch_size = 16

    def forward(self, x):
        # x: [B, 3, H, W]
        B = x.shape[0]

        # ViT forward returns an object with last_hidden_state
        # shape: [B, num_patches + 1, 768]
        # the +1 is the CLS token at position 0
        # for 1024×1024 with patch16: num_patches = (1024/16)^2 = 4096
        outputs = self.vit(pixel_values=x)
        hidden  = outputs.last_hidden_state

        # drop the CLS token — index 0 is a global summary token,
        # not tied to any spatial location, so useless for detection
        patch_tokens = hidden[:, 1:, :]   # [B, 4096, 768]

        # reshape flat patch sequence back into 2D spatial grid
        # 4096 patches → 64×64 grid (since sqrt(4096) = 64)
        num_patches = patch_tokens.shape[1]
        H = W = int(num_patches ** 0.5)

        # [B, N, C] → [B, C, H, W]
        features = patch_tokens.permute(0, 2, 1).reshape(B, 768, H, W)

        # return as dict — Faster R-CNN expects this format
        return {'0': features}

# ---- 3c. Add ImageNet normalization ----
# ViT was pretrained with ImageNet normalization
# inputs must be normalized the same way for features to make sense
normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

class NormalizedViTBackbone(nn.Module):
    def __init__(self, backbone, normalize):
        super().__init__()
        self.backbone     = backbone
        self.normalize    = normalize
        self.out_channels = backbone.out_channels

    def forward(self, x):
        # normalize each image before passing to ViT
        x = torch.stack([self.normalize(img) for img in x])
        return self.backbone(x)

raw_backbone        = ViTBackbone(vit_model)
normalized_backbone = NormalizedViTBackbone(raw_backbone, normalize)

# ---- 3d. Freeze ViT weights ----
# same reasoning as all other models —
# pretrained features are valuable, protect them from the small dataset
for param in normalized_backbone.backbone.vit.parameters():
    param.requires_grad = False

# ---- 3e. Anchor generator ----
# same anchors as all other models for fair comparison
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
# only pass trainable parameters — ViT backbone is frozen
# ============================================================

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5
)


# ============================================================
# STEP 6 — TRAINING LOOP
# ============================================================

print(f"Starting Vanilla ViT training for {EPOCHS} epochs...")

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

        # Faster R-CNN returns dict of 4 losses when targets are provided
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
    # model.train() kept intentionally — Faster R-CNN only returns
    # losses in train mode. torch.no_grad() prevents weight updates.
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
        torch.save(model.state_dict(), f"{RESULTS_DIR}/best_vit_sunspots.pt")
        print(f"  --> New best model saved (val_loss={best_loss:.4f})")


# ============================================================
# STEP 7 — TESTING
# ============================================================

print("\nRunning on test set...")

model.load_state_dict(torch.load(f"{RESULTS_DIR}/best_vit_sunspots.pt", map_location=device))
model.eval()

all_predictions   = []
all_targets_boxes = []

with torch.no_grad():
    for images, targets in tqdm(test_loader, desc="Testing"):
        images = [img.to(device) for img in images]

        # model.eval() → returns predictions not losses
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
# Identical across all five models for fair comparison
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

        # filter out dummy background boxes inserted for empty images
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

print("\n========== VANILLA ViT RESULTS ==========")
print(f"Precision : {metrics['precision']:.4f}  (of all detections, how many were correct)")
print(f"Recall    : {metrics['recall']:.4f}  (of all sunspots, how many were found)")
print(f"F1 Score  : {metrics['f1']:.4f}  (balance of precision and recall)")
print(f"True Positives  : {metrics['tp']}  (correctly detected sunspots)")
print(f"False Positives : {metrics['fp']}  (wrong detections)")
print(f"False Negatives : {metrics['fn']}  (missed sunspots)")
print(f"Best Val Loss   : {best_loss:.4f}")
print("==========================================\n")


# ============================================================
# STEP 9 — SAVE RESULTS
# Same JSON format as all other models for compare_models.py
# ============================================================

final_results = {
    'model':            'VanillaViT',
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

with open(f'{RESULTS_DIR}/VanillaViT_results.json', 'w') as f:
    json.dump(final_results, f, indent=4)

print("Results saved to results/VanillaViT_results.json")
print("Vanilla ViT training complete!")
