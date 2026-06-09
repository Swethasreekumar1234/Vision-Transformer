# ============================================================
# compare_models.py
# Loads all five model result JSONs and produces a comparison
# table with precision, recall, F1, and val loss.
# Run after all five models have been trained.
# ============================================================

import json
import os

RESULTS_DIR = 'results'

# ============================================================
# STEP 1 — LOAD ALL RESULTS
# ============================================================

model_files = {
    'DETR':        'DETR_results.json',
    'Swin V2':     'SwinV2_results.json',
    'BEiT V2':     'BEiTV2_results.json',
    'DeiT III':    'DeiT3_results.json',
    'Vanilla ViT': 'VanillaViT_results.json',
    'YOLOv8':      'YOLOv8_results.json',
}

results = {}

for model_name, filename in model_files.items():
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            results[model_name] = json.load(f)
        print(f"Loaded: {filename}")
    else:
        print(f"Missing: {filename} — skipping {model_name}")

if not results:
    print("No results found. Train at least one model first.")
    exit()

# ============================================================
# STEP 2 — PRINT COMPARISON TABLE
# ============================================================

print("\n")
print("=" * 75)
print(f"{'MODEL':<16} {'PRECISION':>10} {'RECALL':>10} {'F1 SCORE':>10} {'VAL LOSS':>10}")
print("=" * 75)

# sort by F1 score descending so best model is at the top
sorted_results = sorted(results.items(), key=lambda x: x[1].get('f1', 0), reverse=True)

for model_name, data in sorted_results:
    precision = data.get('precision', 0)
    recall    = data.get('recall',    0)
    f1        = data.get('f1',        0)
    val_loss  = data.get('best_val_loss', data.get('final_val_loss', 0))

    print(f"{model_name:<16} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {val_loss:>10.4f}")

print("=" * 75)

# ============================================================
# STEP 3 — HIGHLIGHT BEST MODEL PER METRIC
# ============================================================

print("\n--- Best per metric ---")

best_precision = max(results.items(), key=lambda x: x[1].get('precision', 0))
best_recall    = max(results.items(), key=lambda x: x[1].get('recall',    0))
best_f1        = max(results.items(), key=lambda x: x[1].get('f1',        0))
best_val_loss  = min(results.items(), key=lambda x: x[1].get('best_val_loss', x[1].get('final_val_loss', float('inf'))))

print(f"Best Precision : {best_precision[0]}  ({best_precision[1].get('precision', 0):.4f})")
print(f"Best Recall    : {best_recall[0]}  ({best_recall[1].get('recall', 0):.4f})")
print(f"Best F1        : {best_f1[0]}  ({best_f1[1].get('f1', 0):.4f})")
print(f"Best Val Loss  : {best_val_loss[0]}  ({best_val_loss[1].get('best_val_loss', best_val_loss[1].get('final_val_loss', 0)):.4f})")

# ============================================================
# STEP 4 — DETAILED BREAKDOWN PER MODEL
# ============================================================

print("\n--- Detailed breakdown ---")

for model_name, data in sorted_results:
    print(f"\n{model_name}")
    print(f"  Precision : {data.get('precision', 0):.4f}")
    print(f"  Recall    : {data.get('recall',    0):.4f}")
    print(f"  F1 Score  : {data.get('f1',        0):.4f}")
    print(f"  TP        : {data.get('tp', 0)}")
    print(f"  FP        : {data.get('fp', 0)}")
    print(f"  FN        : {data.get('fn', 0)}")
    print(f"  Val Loss  : {data.get('best_val_loss', data.get('final_val_loss', 0)):.4f}")
    print(f"  Epochs    : {data.get('epochs', 'N/A')}")

# ============================================================
# STEP 5 — SAVE COMPARISON TO JSON
# ============================================================

comparison = {
    model_name: {
        'precision': data.get('precision', 0),
        'recall':    data.get('recall',    0),
        'f1':        data.get('f1',        0),
        'val_loss':  data.get('best_val_loss', data.get('final_val_loss', 0)),
        'tp':        data.get('tp', 0),
        'fp':        data.get('fp', 0),
        'fn':        data.get('fn', 0),
    }
    for model_name, data in results.items()
}

with open(os.path.join(RESULTS_DIR, 'comparison.json'), 'w') as f:
    json.dump(comparison, f, indent=4)

print(f"\nComparison saved to {RESULTS_DIR}/comparison.json")
