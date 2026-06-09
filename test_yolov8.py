# ============================================================
# YOLOv8 - Sunspot Detection Baseline
# ============================================================
# YOLOv8 is the baseline model for this project.
# All five ViT-based models are compared against this.
# YOLO (You Only Look Once) is a one-stage detector:
# it divides the image into a grid and predicts boxes
# and class scores for each grid cell simultaneously.
# No region proposals, no two-stage pipeline — just one
# forward pass from image to detections.
# ============================================================

from ultralytics import YOLO
import json
import os

# ============================================================
# CONFIGURATION
# ============================================================

DATA_YAML   = '/content/Vision-Transformer/data.yaml'
RESULTS_DIR = 'results'
EPOCHS      = 20
PATIENCE    = 5
IMG_SIZE    = 1024
BATCH_SIZE  = 2

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# STEP 1 — TRAIN
# ============================================================

print("Loading YOLOv8 model...")
model = YOLO('yolov8s.pt')

print(f"Starting YOLOv8 training for {EPOCHS} epochs...")
results = model.train(
    data      = DATA_YAML,
    epochs    = EPOCHS,
    imgsz     = IMG_SIZE,
    batch     = BATCH_SIZE,
    patience  = PATIENCE,       # early stopping if no improvement for 5 epochs
    optimizer = 'AdamW',
    device    = 0,              # GPU
    project   = RESULTS_DIR,
    name      = 'yolov8_sunspots',
    exist_ok  = True,
    verbose   = True,
)

print("Training complete.")

# ============================================================
# STEP 2 — EVALUATE ON TEST SET
# ============================================================

print("\nEvaluating on test set...")

# load the best saved weights
best_model = YOLO(f'{RESULTS_DIR}/yolov8_sunspots/weights/best.pt')

# run evaluation on test split
metrics = best_model.val(
    data  = DATA_YAML,
    split = 'test',
    imgsz = IMG_SIZE,
    batch = BATCH_SIZE,
)

# ============================================================
# STEP 3 — EXTRACT AND PRINT METRICS
# ============================================================

precision = float(metrics.box.p.mean())   # mean precision
recall    = float(metrics.box.r.mean())   # mean recall
f1        = float(metrics.box.f1.mean())  # mean F1
map50     = float(metrics.box.map50)      # mAP at IoU 0.5

print("\n========== YOLOv8 RESULTS ==========")
print(f"Precision : {precision:.4f}  (of all detections, how many were correct)")
print(f"Recall    : {recall:.4f}  (of all sunspots, how many were found)")
print(f"F1 Score  : {f1:.4f}  (balance of precision and recall)")
print(f"mAP@0.5   : {map50:.4f}  (mean average precision at IoU 0.5)")
print("=====================================\n")

# ============================================================
# STEP 4 — SAVE RESULTS
# Same JSON format as other models for compare_models.py
# ============================================================

final_results = {
    'model':            'YOLOv8',
    'epochs':           EPOCHS,
    'best_val_loss':    0.0,       # YOLO doesn't report val loss the same way
    'final_train_loss': 0.0,
    'final_val_loss':   0.0,
    'precision':        round(precision, 4),
    'recall':           round(recall,    4),
    'f1':               round(f1,        4),
    'map50':            round(map50,     4),
    'tp':               0,         # YOLO doesn't expose raw TP/FP/FN easily
    'fp':               0,
    'fn':               0,
    'train_losses':     [],
    'val_losses':       [],
}

with open(f'{RESULTS_DIR}/YOLOv8_results.json', 'w') as f:
    json.dump(final_results, f, indent=4)

print("Results saved to results/YOLOv8_results.json")
print("YOLOv8 complete!")
