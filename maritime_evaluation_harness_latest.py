# ============================================================
# SeaDronesSee EVALUATION HARNESS
# - Model evaluation on validation/test set
# - Comprehensive metrics (mAP, Precision, Recall, F1)
# - Per-class performance analysis
# - Confusion matrix and visualization
# - Prediction samples with ground truth comparison
# ============================================================

import os
import cv2
import glob
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import json

# ============================================================
# CONFIGURATION - MUST MATCH TRAINING HARNESS
# ============================================================
TILED_DATASET = "dataset_tiled"      # Same as training
DATA_YAML = "maritime_dataset.yaml"  # Same as training
RESULTS_DIR = "evaluation_results"   # Output directory

# Class names - MUST MATCH TRAINING
CLASS_NAMES = ["swimmer", "boat", "jetski", "lifesaving_appliance", "buoy"]

# ============================================================
# EVALUATION CONFIGURATION
# ============================================================
EVAL_CONFIG = {
    "model_path": "outputs/seadronessee_final/weights/best.pt",  # Trained model
    "conf_threshold": 0.25,      # Confidence threshold
    "iou_threshold": 0.45,       # NMS IoU threshold
    "imgsz": 640,                # Image size (must match training)
    "device": 0,                 # GPU device (0 for cuda:0, 'cpu' for CPU)
    "save_predictions": True,    # Save prediction visualizations
    "num_samples": 20,           # Number of sample visualizations to save
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def setup_directories():
    """Create output directories for evaluation results"""
    dirs = [
        RESULTS_DIR,
        f"{RESULTS_DIR}/predictions",
        f"{RESULTS_DIR}/metrics",
        f"{RESULTS_DIR}/confusion_matrix",
        f"{RESULTS_DIR}/samples"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"[INFO] Created output directories in: {RESULTS_DIR}")

def load_model(model_path):
    """Load trained YOLO model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    print(f"[INFO] Loading model from: {model_path}")
    model = YOLO(model_path)
    print(f"[INFO] Model loaded successfully")
    return model

def parse_ground_truth(label_path, img_width, img_height):
    """Parse YOLO format ground truth labels"""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, xc, yc, w, h = map(float, parts)
                    
                    # Convert to pixel coordinates
                    x1 = int((xc - w/2) * img_width)
                    y1 = int((yc - h/2) * img_height)
                    x2 = int((xc + w/2) * img_width)
                    y2 = int((yc + h/2) * img_height)
                    
                    boxes.append({
                        'class': int(cls),
                        'bbox': [x1, y1, x2, y2],
                        'xywhn': [xc, yc, w, h]  # Normalized format
                    })
    return boxes

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# ============================================================
# EVALUATION FUNCTIONS
# ============================================================
def evaluate_model(model, val_data):
    """
    Comprehensive model evaluation
    Returns detailed metrics for analysis
    """
    print("\n" + "=" * 70)
    print("RUNNING MODEL EVALUATION")
    print("=" * 70)
    
    # Run ultralytics built-in validation
    results = model.val(
        data=DATA_YAML,
        imgsz=EVAL_CONFIG['imgsz'],
        conf=EVAL_CONFIG['conf_threshold'],
        iou=EVAL_CONFIG['iou_threshold'],
        device=EVAL_CONFIG['device'],
        plots=True,
        save_json=True,
        project=RESULTS_DIR,
        name="validation"
    )
    
    return results

def detailed_per_class_evaluation(model, val_images_dir, val_labels_dir):
    """
    Detailed per-class evaluation with custom metrics
    """
    print("\n" + "=" * 70)
    print("DETAILED PER-CLASS EVALUATION")
    print("=" * 70)
    
    # Initialize metrics storage
    class_metrics = {cls_name: {
        'tp': 0, 'fp': 0, 'fn': 0,
        'predictions': [], 'ground_truths': []
    } for cls_name in CLASS_NAMES}
    
    image_files = sorted(glob.glob(f"{val_images_dir}/*.jpg"))
    
    for img_path in tqdm(image_files, desc="Evaluating images"):
        img_name = os.path.basename(img_path)
        label_path = os.path.join(val_labels_dir, img_name.replace('.jpg', '.txt'))
        
        # Load image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # Get ground truth
        gt_boxes = parse_ground_truth(label_path, w, h)
        
        # Get predictions
        results = model.predict(
            img_path,
            conf=EVAL_CONFIG['conf_threshold'],
            iou=EVAL_CONFIG['iou_threshold'],
            imgsz=EVAL_CONFIG['imgsz'],
            device=EVAL_CONFIG['device'],
            verbose=False
        )[0]
        
        # Extract predictions
        pred_boxes = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                pred_boxes.append({
                    'class': int(box.cls[0]),
                    'bbox': box.xyxy[0].cpu().numpy().astype(int).tolist(),
                    'conf': float(box.conf[0])
                })
        
        # Match predictions with ground truth
        matched_gt = set()
        
        for pred in pred_boxes:
            pred_cls = pred['class']
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes):
                if gt['class'] == pred_cls and gt_idx not in matched_gt:
                    iou = calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            cls_name = CLASS_NAMES[pred_cls]
            
            # True positive if IoU > 0.5
            if best_iou >= 0.5:
                class_metrics[cls_name]['tp'] += 1
                matched_gt.add(best_gt_idx)
            else:
                class_metrics[cls_name]['fp'] += 1
            
            class_metrics[cls_name]['predictions'].append(pred['conf'])
        
        # Count false negatives (unmatched ground truths)
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx not in matched_gt:
                cls_name = CLASS_NAMES[gt['class']]
                class_metrics[cls_name]['fn'] += 1
            
            cls_name = CLASS_NAMES[gt['class']]
            class_metrics[cls_name]['ground_truths'].append(1)
    
    # Calculate metrics per class
    print("\n" + "-" * 70)
    print("PER-CLASS METRICS")
    print("-" * 70)
    print(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    print("-" * 70)
    
    overall_metrics = {'precision': [], 'recall': [], 'f1': []}
    has_valid_classes = False
    
    for cls_name in CLASS_NAMES:
        metrics = class_metrics[cls_name]
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
        
        # EDGE CASE HANDLING: Avoid division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn
        
        # Only include classes with ground truth for overall metrics
        if support > 0:
            overall_metrics['precision'].append(precision)
            overall_metrics['recall'].append(recall)
            overall_metrics['f1'].append(f1)
            has_valid_classes = True
        
        print(f"{cls_name:<25} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support}")
    
    # Overall metrics with edge case handling
    print("-" * 70)
    if has_valid_classes and len(overall_metrics['precision']) > 0:
        avg_precision = np.mean(overall_metrics['precision'])
        avg_recall = np.mean(overall_metrics['recall'])
        avg_f1 = np.mean(overall_metrics['f1'])
    else:
        print("[WARN] No valid classes found in validation set!")
        avg_precision = avg_recall = avg_f1 = 0.0
    
    print(f"{'AVERAGE':<25} {avg_precision:<12.4f} {avg_recall:<12.4f} {avg_f1:<12.4f}")
    print("-" * 70)
    
    return class_metrics, overall_metrics

def visualize_predictions(model, val_images_dir, val_labels_dir, num_samples=20):
    """
    Create visualization comparing predictions with ground truth
    """
    print("\n" + "=" * 70)
    print(f"CREATING {num_samples} SAMPLE VISUALIZATIONS")
    print("=" * 70)
    
    image_files = sorted(glob.glob(f"{val_images_dir}/*.jpg"))
    
    # Randomly select samples
    import random
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    for idx, img_path in enumerate(tqdm(sample_files, desc="Generating visualizations")):
        img_name = os.path.basename(img_path)
        label_path = os.path.join(val_labels_dir, img_name.replace('.jpg', '.txt'))
        
        # Load image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # Get ground truth
        gt_boxes = parse_ground_truth(label_path, w, h)
        
        # Get predictions
        results = model.predict(
            img_path,
            conf=EVAL_CONFIG['conf_threshold'],
            iou=EVAL_CONFIG['iou_threshold'],
            imgsz=EVAL_CONFIG['imgsz'],
            device=EVAL_CONFIG['device'],
            verbose=False
        )[0]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Ground Truth
        img_gt = img.copy()
        for gt in gt_boxes:
            x1, y1, x2, y2 = gt['bbox']
            cls_name = CLASS_NAMES[gt['class']]
            cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_gt, cls_name, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        ax1.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"Ground Truth ({len(gt_boxes)} objects)", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Predictions
        img_pred = img.copy()
        num_preds = 0
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_idx = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = CLASS_NAMES[cls_idx]
                
                cv2.rectangle(img_pred, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(img_pred, label, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                num_preds += 1
        
        ax2.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
        ax2.set_title(f"Predictions ({num_preds} objects)", fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.suptitle(f"Sample {idx+1}: {img_name}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = f"{RESULTS_DIR}/samples/sample_{idx+1:03d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"[INFO] Saved {len(sample_files)} visualizations to {RESULTS_DIR}/samples/")

def generate_confusion_matrix(model, val_images_dir, val_labels_dir):
    """Generate and save confusion matrix"""
    print("\n" + "=" * 70)
    print("GENERATING CONFUSION MATRIX")
    print("=" * 70)
    
    num_classes = len(CLASS_NAMES)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    image_files = sorted(glob.glob(f"{val_images_dir}/*.jpg"))
    
    for img_path in tqdm(image_files, desc="Computing confusion matrix"):
        img_name = os.path.basename(img_path)
        label_path = os.path.join(val_labels_dir, img_name.replace('.jpg', '.txt'))
        
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        gt_boxes = parse_ground_truth(label_path, w, h)
        
        results = model.predict(
            img_path,
            conf=EVAL_CONFIG['conf_threshold'],
            iou=EVAL_CONFIG['iou_threshold'],
            imgsz=EVAL_CONFIG['imgsz'],
            device=EVAL_CONFIG['device'],
            verbose=False
        )[0]
        
        pred_boxes = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                pred_boxes.append({
                    'class': int(box.cls[0]),
                    'bbox': box.xyxy[0].cpu().numpy().astype(int).tolist(),
                    'conf': float(box.conf[0])
                })
        
        # Match predictions with ground truth
        matched_gt = set()
        
        for pred in pred_boxes:
            pred_cls = pred['class']
            best_iou = 0
            best_gt_cls = None
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx not in matched_gt:
                    iou = calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_cls = gt['class']
                        best_gt_idx = gt_idx
            
            if best_iou >= 0.5 and best_gt_cls is not None:
                confusion_matrix[best_gt_cls][pred_cls] += 1
                matched_gt.add(best_gt_idx)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    plt.ylabel('True Class', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_path = f"{RESULTS_DIR}/confusion_matrix/confusion_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved confusion matrix to {output_path}")
    
    return confusion_matrix

def analyze_errors(class_metrics):
    """
    Identify patterns in model failures
    Helps guide improvements and data collection
    """
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)
    
    issues_found = False
    
    for cls_name, metrics in class_metrics.items():
        tp = metrics['tp']
        fp = metrics['fp']
        fn = metrics['fn']
        
        # High false positives (over-prediction)
        if fp > tp and tp > 0:
            ratio = fp / tp
            print(f"⚠️  {cls_name}: High FALSE POSITIVES")
            print(f"    FP:{fp} > TP:{tp} (ratio: {ratio:.2f}x)")
            print(f"    → Model over-predicts this class")
            issues_found = True
        
        # High false negatives (missing detections)
        if fn > tp and tp > 0:
            ratio = fn / tp
            print(f"⚠️  {cls_name}: High FALSE NEGATIVES")
            print(f"    FN:{fn} > TP:{tp} (ratio: {ratio:.2f}x)")
            print(f"    → Model misses many instances")
            issues_found = True
        
        # Very low recall (critical issue)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if recall < 0.3 and (tp + fn) > 10:  # At least 10 instances
            print(f"❌ {cls_name}: CRITICAL - Very low recall ({recall:.2%})")
            print(f"    → Model fails to detect most instances")
            issues_found = True
        
        # No ground truth instances
        if tp + fn == 0:
            print(f"ℹ️  {cls_name}: No ground truth instances in validation set")
            issues_found = True
    
    if not issues_found:
        print("✓ No significant error patterns detected")
        print("  All classes have balanced performance")
    
    print("-" * 70)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    for cls_name, metrics in class_metrics.items():
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
        
        if fp > tp and tp > 0:
            print(f"  • {cls_name}: Add more negative examples or increase confidence threshold")
        if fn > tp and tp > 0:
            print(f"  • {cls_name}: Collect more training data or use data augmentation")
        if tp + fn == 0:
            print(f"  • {cls_name}: Add validation samples for this class")

def save_evaluation_report(results, class_metrics, overall_metrics, confusion_matrix):
    """Save comprehensive evaluation report"""
    print("\n" + "=" * 70)
    print("SAVING EVALUATION REPORT")
    print("=" * 70)
    
    report = {
        "model_path": EVAL_CONFIG['model_path'],
        "evaluation_config": EVAL_CONFIG,
        "class_names": CLASS_NAMES,
        "overall_metrics": {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "average_precision": float(np.mean(overall_metrics['precision'])),
            "average_recall": float(np.mean(overall_metrics['recall'])),
            "average_f1": float(np.mean(overall_metrics['f1']))
        },
        "per_class_metrics": {}
    }
    
    for idx, cls_name in enumerate(CLASS_NAMES):
        metrics = class_metrics[cls_name]
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        report["per_class_metrics"][cls_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "support": int(tp + fn)
        }
    
    # Save JSON report
    report_path = f"{RESULTS_DIR}/evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"[INFO] Saved JSON report to {report_path}")
    
    # Save text summary
    summary_path = f"{RESULTS_DIR}/evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SEADRONESSEE MODEL EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Model: {EVAL_CONFIG['model_path']}\n")
        f.write(f"Confidence Threshold: {EVAL_CONFIG['conf_threshold']}\n")
        f.write(f"IoU Threshold: {EVAL_CONFIG['iou_threshold']}\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"mAP@0.5: {report['overall_metrics']['mAP50']:.4f}\n")
        f.write(f"mAP@0.5:0.95: {report['overall_metrics']['mAP50-95']:.4f}\n")
        f.write(f"Average Precision: {report['overall_metrics']['average_precision']:.4f}\n")
        f.write(f"Average Recall: {report['overall_metrics']['average_recall']:.4f}\n")
        f.write(f"Average F1-Score: {report['overall_metrics']['average_f1']:.4f}\n\n")
        
        f.write("PER-CLASS METRICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}\n")
        f.write("-" * 70 + "\n")
        
        for cls_name, metrics in report["per_class_metrics"].items():
            f.write(f"{cls_name:<25} {metrics['precision']:<12.4f} "
                   f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f} "
                   f"{metrics['support']}\n")
    
    print(f"[INFO] Saved text summary to {summary_path}")

# ============================================================
# MAIN EVALUATION PIPELINE
# ============================================================
def main():
    print("=" * 70)
    print("SEADRONESSEE EVALUATION HARNESS")
    print("=" * 70)
    
    # Setup
    setup_directories()
    
    # Check if model exists
    if not os.path.exists(EVAL_CONFIG['model_path']):
        print(f"\n[ERROR] Model not found at: {EVAL_CONFIG['model_path']}")
        print("[ERROR] Please train the model first using the training harness")
        return
    
    # Load model
    model = load_model(EVAL_CONFIG['model_path'])
    
    # Define validation paths
    val_images_dir = f"{TILED_DATASET}/images/val"
    val_labels_dir = f"{TILED_DATASET}/labels/val"
    
    # Check if validation data exists
    if not os.path.exists(val_images_dir):
        print(f"\n[ERROR] Validation images not found at: {val_images_dir}")
        return
    
    val_images = glob.glob(f"{val_images_dir}/*.jpg")
    print(f"\n[INFO] Found {len(val_images)} validation images")
    
    # 1. Run built-in YOLO validation
    results = evaluate_model(model, DATA_YAML)
    
    # 2. Detailed per-class evaluation
    class_metrics, overall_metrics = detailed_per_class_evaluation(
        model, val_images_dir, val_labels_dir
    )
    
    # 3. Generate confusion matrix
    confusion_matrix = generate_confusion_matrix(
        model, val_images_dir, val_labels_dir
    )
    
    # 4. Error analysis (NEW!)
    analyze_errors(class_metrics)
    
    # 5. Visualize predictions
    if EVAL_CONFIG['save_predictions']:
        visualize_predictions(
            model, val_images_dir, val_labels_dir, 
            num_samples=EVAL_CONFIG['num_samples']
        )
    
    # 6. Save comprehensive report
    save_evaluation_report(results, class_metrics, overall_metrics, confusion_matrix)
    
    # Final summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"  - Validation results: {RESULTS_DIR}/validation/")
    print(f"  - Confusion matrix: {RESULTS_DIR}/confusion_matrix/")
    print(f"  - Sample predictions: {RESULTS_DIR}/samples/")
    print(f"  - JSON report: {RESULTS_DIR}/evaluation_report.json")
    print(f"  - Text summary: {RESULTS_DIR}/evaluation_summary.txt")
    print("=" * 70)

if __name__ == "__main__":
    main()