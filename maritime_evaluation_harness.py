"""
🎯 Maritime Object Detection - Enhanced Evaluation Harness (YOLOv8)
===============================================================================
Person 3: Comprehensive Model Evaluation Pipeline

Compatible with Person 2's YOLOv8l training pipeline

Features:
- ✅ YOLOv8 model support (not Faster R-CNN)
- ✅ 5 class detection (swimmer, boat, jetski, lifesaving_appliance, buoy)
- ✅ YOLO format annotation support
- ✅ Fixed annotation paths
- ✅ mAP@0.5, @0.75, @0.5:0.95 (COCO-style)
- ✅ Confidence threshold optimization
- ✅ Speed benchmarking with FPS
- ✅ Confusion matrix visualization
- ✅ Per-image detailed logging
- ✅ Comprehensive JSON reports

Usage:
    python maritime_evaluation_harness.py

Configuration:
    Adjust paths in the CONFIGURATION section below before running
===============================================================================
"""

import json
import os
import time
from collections import defaultdict
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
from ultralytics import YOLO

plt.style.use('seaborn-v0_8-darkgrid')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Base directory (current working directory)
BASE_DIR = os.getcwd()

# Dataset paths - Person 2 uses tiled dataset
VAL_ANNOTATIONS = os.path.join(BASE_DIR, 'dataset', 'annotations', 'instances_val.json')
VAL_IMAGES = os.path.join(BASE_DIR, 'dataset', 'images', 'val')

# YOLO tiled dataset (if using tiled images from Person 2)
TILED_VAL_IMAGES = os.path.join(BASE_DIR, 'dataset_tiled', 'images', 'val')
TILED_VAL_LABELS = os.path.join(BASE_DIR, 'dataset_tiled', 'labels', 'val')

# Model checkpoint from Person 2
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'outputs', 'train', 'weights', 'best.pt')

# Evaluation parameters
CONFIDENCE_THRESHOLD = 0.5  # Will be optimized via sweep
IOU_THRESHOLD = 0.5  # For mAP@0.5
IMG_SIZE = 640  # YOLOv8 training size

# Class configuration - UPDATED for 5 classes
NUM_CLASSES = 5
CLASS_NAMES = ['swimmer', 'boat', 'jetski', 'lifesaving_appliance', 'buoy']

# Output directory
OUTPUT_DIR = os.path.join(BASE_DIR, 'evaluation_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
# MODEL LOADING (YOLOv8)
# ==============================================================================

def load_yolo_model(checkpoint_path):
    """
    Load YOLOv8 model from checkpoint
    Compatible with Person 2's training output
    """
    print(f"📥 Loading YOLOv8 model from {checkpoint_path}...")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = YOLO(checkpoint_path)
    print("✅ YOLOv8 model loaded successfully!")
    print(f"   Model: {model.model.__class__.__name__}")
    print(f"   Classes: {len(CLASS_NAMES)}")

    return model


# ==============================================================================
# INFERENCE FUNCTION (YOLOv8)
# ==============================================================================

def run_inference_yolo(model, image_dir, confidence_threshold=0.5, log_per_image=False):
    """
    🆕 Run inference on validation images using YOLOv8

    Args:
        model: YOLOv8 model
        image_dir: Directory containing validation images
        confidence_threshold: Confidence score threshold
        log_per_image: If True, return detailed per-image results

    Returns:
        predictions: List of prediction dicts
        inference_times: List of inference times per image
        per_image_logs: (Optional) Detailed logs per image
    """
    predictions = []
    inference_times = []
    per_image_logs = [] if log_per_image else None

    # Get all images
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    print(f"🔮 Running YOLOv8 inference on {len(image_files)} images...")

    for img_file in tqdm(image_files, desc="Inference"):
        img_path = os.path.join(image_dir, img_file)

        # Time inference
        start_time = time.time()
        results = model(img_path, conf=confidence_threshold, verbose=False)
        inference_time = (time.time() - start_time) * 1000  # ms
        inference_times.append(inference_time)

        # Extract predictions
        result = results[0]
        boxes = result.boxes

        pred = {
            'filename': img_file,
            'boxes': boxes.xyxy.cpu().numpy() if len(boxes) > 0 else np.array([]),
            'labels': boxes.cls.cpu().numpy().astype(int) if len(boxes) > 0 else np.array([]),
            'scores': boxes.conf.cpu().numpy() if len(boxes) > 0 else np.array([])
        }
        predictions.append(pred)

        # Per-image logging
        if log_per_image:
            log = {
                'filename': img_file,
                'num_predictions': len(pred['boxes']),
                'prediction_scores': pred['scores'].tolist(),
                'prediction_labels': pred['labels'].tolist(),
                'class_distribution': {
                    CLASS_NAMES[i]: int(np.sum(pred['labels'] == i)) 
                    for i in range(NUM_CLASSES)
                }
            }
            per_image_logs.append(log)

    print(f"✅ Inference complete: {len(predictions)} images processed")
    print(f"⚡ Average inference time: {np.mean(inference_times):.2f} ms/image")

    if log_per_image:
        return predictions, inference_times, per_image_logs
    return predictions, inference_times


def load_ground_truth_yolo(label_dir, image_dir, img_size=640):
    """
    Load ground truth from YOLO format labels

    Args:
        label_dir: Directory with .txt label files
        image_dir: Directory with images (to get dimensions)
        img_size: Image size used during training

    Returns:
        ground_truths: List of ground truth dicts
    """
    ground_truths = []

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    print(f"📥 Loading {len(label_files)} ground truth labels...")

    for label_file in label_files:
        img_file = label_file.replace('.txt', '.jpg')
        img_path = os.path.join(image_dir, img_file)

        if not os.path.exists(img_path):
            continue

        # Read image to get dimensions
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Read YOLO labels
        label_path = os.path.join(label_dir, label_file)
        boxes = []
        labels = []

        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls, xc, yc, bw, bh = map(float, parts[:5])

                # Convert YOLO format (normalized) to xyxy
                x1 = (xc - bw/2) * w
                y1 = (yc - bh/2) * h
                x2 = (xc + bw/2) * w
                y2 = (yc + bh/2) * h

                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls))

        gt = {
            'filename': img_file,
            'boxes': np.array(boxes, dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64)
        }
        ground_truths.append(gt)

    print(f"✅ Loaded {len(ground_truths)} ground truth annotations")
    return ground_truths


# ==============================================================================
# METRICS COMPUTATION
# ==============================================================================

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


def compute_ap(recalls, precisions):
    """Compute Average Precision using 11-point interpolation"""
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    return ap


def calculate_map(predictions, ground_truths, iou_threshold=0.5, num_classes=5):
    """
    Calculate mean Average Precision (mAP) at specific IoU threshold

    Args:
        predictions: List of prediction dicts
        ground_truths: List of ground truth dicts
        iou_threshold: IoU threshold for match
        num_classes: Number of classes

    Returns:
        mAP: mean Average Precision across all classes
        per_class_ap: AP for each class
        precision: Overall precision
        recall: Overall recall
    """
    # Match predictions to ground truths by filename
    pred_dict = {p['filename']: p for p in predictions}
    gt_dict = {g['filename']: g for g in ground_truths}

    aps = []
    per_class_ap = {}

    tp_total = 0
    fp_total = 0
    fn_total = 0

    for class_id in range(num_classes):
        class_preds = []
        class_gts = {}

        # Collect predictions for this class
        for filename, pred in pred_dict.items():
            class_mask = pred['labels'] == class_id
            if np.any(class_mask):
                for box, score in zip(pred['boxes'][class_mask], pred['scores'][class_mask]):
                    class_preds.append({
                        'filename': filename,
                        'box': box,
                        'score': score
                    })

        # Collect ground truths for this class
        for filename, gt in gt_dict.items():
            class_mask = gt['labels'] == class_id
            if np.any(class_mask):
                class_gts[filename] = {
                    'boxes': gt['boxes'][class_mask],
                    'detected': np.zeros(len(gt['boxes'][class_mask]))
                }

        if len(class_preds) == 0:
            per_class_ap[class_id] = 0.0
            continue

        class_preds = sorted(class_preds, key=lambda x: x['score'], reverse=True)

        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        total_gt = sum(len(gt['boxes']) for gt in class_gts.values())

        for i, pred in enumerate(class_preds):
            filename = pred['filename']
            if filename not in class_gts:
                fp[i] = 1
                continue

            gt_boxes = class_gts[filename]['boxes']
            detected = class_gts[filename]['detected']

            max_iou = 0
            max_idx = -1

            for j, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred['box'], gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j

            if max_iou >= iou_threshold and detected[max_idx] == 0:
                tp[i] = 1
                detected[max_idx] = 1
            else:
                fp[i] = 1

        fp_cumsum = np.cumsum(fp)
        tp_cumsum = np.cumsum(tp)
        recalls = tp_cumsum / total_gt if total_gt > 0 else np.zeros(len(tp_cumsum))
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = compute_ap(recalls, precisions)
        aps.append(ap)
        per_class_ap[class_id] = ap

        tp_total += int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0
        fp_total += int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0
        fn_total += total_gt - (int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0)

    mAP = np.mean(aps) if len(aps) > 0 else 0.0
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0

    return mAP, per_class_ap, precision, recall


def calculate_map_coco_style(predictions, ground_truths, num_classes=5):
    """
    🆕 Calculate mAP@0.5:0.95 (COCO-style metric)
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps_at_thresholds = []

    print("\n🎯 Computing mAP@0.5:0.95 (COCO-style)...")
    for iou_thresh in tqdm(iou_thresholds, desc="IoU thresholds"):
        mAP, _, _, _ = calculate_map(predictions, ground_truths, iou_threshold=iou_thresh, num_classes=num_classes)
        aps_at_thresholds.append(mAP)

    map_coco = np.mean(aps_at_thresholds)
    return map_coco, aps_at_thresholds, iou_thresholds


# ==============================================================================
# CONFUSION MATRIX
# ==============================================================================

def compute_confusion_matrix(predictions, ground_truths, num_classes=5, iou_threshold=0.5):
    """
    🆕 Compute confusion matrix for object detection
    """
    pred_dict = {p['filename']: p for p in predictions}
    gt_dict = {g['filename']: g for g in ground_truths}

    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

    for filename in gt_dict.keys():
        if filename not in pred_dict:
            continue

        pred = pred_dict[filename]
        gt = gt_dict[filename]

        matched_gts = set()

        for pred_box, pred_label in zip(pred['boxes'], pred['labels']):
            best_iou = 0
            best_gt_idx = -1
            best_gt_label = -1

            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt['boxes'], gt['labels'])):
                if gt_idx in matched_gts:
                    continue

                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_gt_label = gt_label

            if best_gt_idx != -1:
                cm[best_gt_label, pred_label] += 1
                matched_gts.add(best_gt_idx)
            else:
                cm[num_classes, pred_label] += 1

        for gt_idx, gt_label in enumerate(gt['labels']):
            if gt_idx not in matched_gts:
                cm[gt_label, num_classes] += 1

    return cm


def plot_confusion_matrix(cm, class_names=CLASS_NAMES, save_path=None):
    """🆕 Visualize confusion matrix as heatmap"""
    labels = class_names + ['Background/FN']

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels + ['FP'],
                yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Maritime Object Detection', fontsize=14, fontweight='bold')
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xlabel('Prediction', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved: {save_path}")

    plt.show()


# ==============================================================================
# CONFIDENCE THRESHOLD SWEEP
# ==============================================================================

def confidence_threshold_sweep(model, image_dir, ground_truths,
                                thresholds=np.arange(0.1, 1.0, 0.05),
                                iou_threshold=0.5, num_classes=5):
    """
    🆕 Sweep confidence thresholds to find optimal value
    """
    print("🔍 Running confidence threshold sweep...")

    # Run inference once with threshold=0 to get all predictions
    raw_predictions, _ = run_inference_yolo(model, image_dir, confidence_threshold=0.0)

    results = {
        'thresholds': [],
        'mAP': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for threshold in tqdm(thresholds, desc="Threshold sweep"):
        # Filter predictions by threshold
        filtered_preds = []
        for pred in raw_predictions:
            keep = pred['scores'] > threshold
            filtered_pred = {
                'filename': pred['filename'],
                'boxes': pred['boxes'][keep],
                'labels': pred['labels'][keep],
                'scores': pred['scores'][keep]
            }
            filtered_preds.append(filtered_pred)

        mAP, _, precision, recall = calculate_map(filtered_preds, ground_truths, iou_threshold, num_classes)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results['thresholds'].append(threshold)
        results['mAP'].append(mAP)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1'].append(f1)

    optimal_idx = np.argmax(results['f1'])
    optimal_threshold = results['thresholds'][optimal_idx]

    print(f"\n✅ Optimal confidence threshold: {optimal_threshold:.2f}")
    print(f"   mAP: {results['mAP'][optimal_idx]:.4f}")
    print(f"   Precision: {results['precision'][optimal_idx]:.4f}")
    print(f"   Recall: {results['recall'][optimal_idx]:.4f}")
    print(f"   F1: {results['f1'][optimal_idx]:.4f}")

    return results, optimal_threshold


def plot_threshold_sweep(results, save_path=None):
    """🆕 Plot threshold sweep results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ['mAP', 'precision', 'recall', 'f1']
    titles = ['mAP vs Confidence Threshold', 'Precision vs Confidence Threshold',
              'Recall vs Confidence Threshold', 'F1 Score vs Confidence Threshold']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i // 2, i % 2]
        ax.plot(results['thresholds'], results[metric], marker='o', linewidth=2)
        ax.set_xlabel('Confidence Threshold', fontsize=11)
        ax.set_ylabel(metric.upper() if metric == 'mAP' else metric.capitalize(), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if metric == 'f1':
            optimal_idx = np.argmax(results[metric])
            ax.scatter(results['thresholds'][optimal_idx], results[metric][optimal_idx],
                      color='red', s=100, zorder=5, label='Optimal')
            ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Threshold sweep plot saved: {save_path}")

    plt.show()


# ==============================================================================
# SPEED BENCHMARKING
# ==============================================================================

def benchmark_inference_speed(model, image_dir, num_warmup=10, num_iterations=50):
    """
    🆕 Benchmark inference speed for deployment planning
    """
    print(f"⚡ Benchmarking YOLOv8 inference speed...")

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) == 0:
        print("❌ No images found for benchmarking!")
        return None

    # Select random images
    import random
    selected_images = random.sample(image_files, min(num_warmup + num_iterations, len(image_files)))

    times = []

    # Warmup
    print(f"\n🔥 Warming up ({num_warmup} iterations)...")
    for i in range(num_warmup):
        img_path = os.path.join(image_dir, selected_images[i])
        _ = model(img_path, verbose=False)

    # Benchmark
    print(f"\n⏱️ Running benchmark ({num_iterations} iterations)...")
    for i in tqdm(range(num_iterations), desc="Benchmark"):
        img_path = os.path.join(image_dir, selected_images[num_warmup + i])

        start_time = time.time()
        _ = model(img_path, verbose=False)
        elapsed = (time.time() - start_time) * 1000  # ms
        times.append(elapsed)

    speed_stats = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'median_ms': np.median(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'fps': 1000 / np.mean(times),
        'device': str(device)
    }

    print("\n" + "="*60)
    print("⚡ INFERENCE SPEED BENCHMARK RESULTS")
    print("="*60)
    print(f"Device: {device}")
    print(f"Mean inference time: {speed_stats['mean_ms']:.2f} ± {speed_stats['std_ms']:.2f} ms/image")
    print(f"Median inference time: {speed_stats['median_ms']:.2f} ms/image")
    print(f"Min/Max: {speed_stats['min_ms']:.2f} / {speed_stats['max_ms']:.2f} ms/image")
    print(f"Throughput: {speed_stats['fps']:.1f} FPS")
    print("="*60)

    return speed_stats


# ==============================================================================
# SUBMISSION FILE GENERATION
# ==============================================================================

def generate_coco_submission(predictions, output_path='submission.json'):
    """Generate COCO format submission file"""
    results = []

    for pred in predictions:
        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            result = {
                'filename': pred['filename'],
                'category_id': int(label),
                'category_name': CLASS_NAMES[int(label)],
                'bbox': [float(x1), float(y1), float(w), float(h)],
                'score': float(score)
            }
            results.append(result)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Submission file saved: {output_path}")
    print(f"   Total detections: {len(results)}")


# ==============================================================================
# MAIN EVALUATION PIPELINE
# ==============================================================================

def main():
    """Main evaluation pipeline"""

    print("="*70)
    print("🎯 MARITIME OBJECT DETECTION - EVALUATION HARNESS (YOLOv8)")
    print("="*70)
    print(f"🖥️ Evaluation running on: {device}")
    print(f"📂 Base directory: {BASE_DIR}")
    print(f"📂 Validation images: {TILED_VAL_IMAGES}")
    print(f"📂 Validation labels: {TILED_VAL_LABELS}")
    print(f"📂 Output: {OUTPUT_DIR}")
    print(f"🎯 Classes: {CLASS_NAMES}")
    print("="*70)

    # Load model
    model = load_yolo_model(CHECKPOINT_PATH)

    # Load ground truth
    ground_truths = load_ground_truth_yolo(TILED_VAL_LABELS, TILED_VAL_IMAGES, IMG_SIZE)

    # STEP 1: Speed Benchmarking
    print("\n" + "="*70)
    print("STEP 1: INFERENCE SPEED BENCHMARK")
    print("="*70)
    speed_stats = benchmark_inference_speed(model, TILED_VAL_IMAGES, num_warmup=10, num_iterations=50)

    # STEP 2: Confidence Threshold Optimization
    print("\n" + "="*70)
    print("STEP 2: CONFIDENCE THRESHOLD SWEEP")
    print("="*70)
    sweep_results, optimal_conf_threshold = confidence_threshold_sweep(
        model, TILED_VAL_IMAGES, ground_truths,
        thresholds=np.arange(0.1, 0.9, 0.05),
        iou_threshold=IOU_THRESHOLD,
        num_classes=NUM_CLASSES
    )

    plot_threshold_sweep(sweep_results, save_path=os.path.join(OUTPUT_DIR, 'threshold_sweep.png'))

    # Update confidence threshold
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = optimal_conf_threshold
    print(f"\n✅ Updated CONFIDENCE_THRESHOLD to: {CONFIDENCE_THRESHOLD:.2f}")

    # STEP 3: Main Inference
    print("\n" + "="*70)
    print("STEP 3: MAIN INFERENCE")
    print("="*70)
    predictions, inference_times, per_image_logs = run_inference_yolo(
        model, TILED_VAL_IMAGES,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        log_per_image=True
    )

    # Save per-image logs
    with open(os.path.join(OUTPUT_DIR, 'per_image_logs.json'), 'w') as f:
        json.dump(per_image_logs, f, indent=2)
    print(f"✅ Per-image logs saved: {os.path.join(OUTPUT_DIR, 'per_image_logs.json')}")

    # STEP 4: Comprehensive Metrics
    print("\n" + "="*70)
    print("STEP 4: METRICS COMPUTATION")
    print("="*70)

    print("\n📊 Computing mAP@0.5...")
    mAP_50, per_class_ap_50, precision_50, recall_50 = calculate_map(
        predictions, ground_truths, iou_threshold=0.5, num_classes=NUM_CLASSES
    )

    print("\n📊 Computing mAP@0.75...")
    mAP_75, per_class_ap_75, precision_75, recall_75 = calculate_map(
        predictions, ground_truths, iou_threshold=0.75, num_classes=NUM_CLASSES
    )

    mAP_coco, aps_at_thresholds, iou_thresholds = calculate_map_coco_style(
        predictions, ground_truths, num_classes=NUM_CLASSES
    )

    f1_50 = 2 * precision_50 * recall_50 / (precision_50 + recall_50) if (precision_50 + recall_50) > 0 else 0

    print("\n" + "="*70)
    print("📈 COMPREHENSIVE EVALUATION RESULTS")
    print("="*70)
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD:.2f}")
    print(f"\n🎯 mAP Metrics:")
    print(f"   mAP@0.5:      {mAP_50:.4f}")
    print(f"   mAP@0.75:     {mAP_75:.4f}")
    print(f"   mAP@0.5:0.95: {mAP_coco:.4f} (COCO-style)")
    print(f"\n📊 Per-Class AP@0.5:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"   {class_name:25s}: {per_class_ap_50.get(i, 0):.4f}")
    print(f"\n📊 Overall Performance@0.5:")
    print(f"   Precision: {precision_50:.4f}")
    print(f"   Recall:    {recall_50:.4f}")
    print(f"   F1 Score:  {f1_50:.4f}")
    print("="*70)

    # STEP 5: Confusion Matrix
    print("\n" + "="*70)
    print("STEP 5: CONFUSION MATRIX ANALYSIS")
    print("="*70)
    cm = compute_confusion_matrix(predictions, ground_truths, num_classes=NUM_CLASSES, iou_threshold=0.5)
    plot_confusion_matrix(cm, class_names=CLASS_NAMES,
                         save_path=os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))

    # STEP 6: Statistics
    print("\n" + "="*70)
    print("STEP 6: DETECTION STATISTICS")
    print("="*70)

    total_predictions = sum(len(p['boxes']) for p in predictions)
    total_ground_truths = sum(len(g['boxes']) for g in ground_truths)

    per_class_stats = {}
    for i, class_name in enumerate(CLASS_NAMES):
        preds = sum(np.sum(p['labels'] == i) for p in predictions)
        gts = sum(np.sum(g['labels'] == i) for g in ground_truths)
        per_class_stats[class_name] = {'predictions': int(preds), 'ground_truths': int(gts)}

    no_detection = sum(1 for p in predictions if len(p['boxes']) == 0)

    print(f"\n📊 Overall:")
    print(f"   Total predictions: {total_predictions}")
    print(f"   Total ground truths: {total_ground_truths}")
    print(f"\n📊 Per-Class:")
    for class_name, stats in per_class_stats.items():
        print(f"   {class_name:25s}: {stats['predictions']:4d} predictions (GT: {stats['ground_truths']:4d})")
    print(f"\n📊 Coverage:")
    print(f"   Images with detections: {len(predictions) - no_detection} / {len(predictions)}")
    print(f"   Images with no detections: {no_detection} / {len(predictions)}")
    print("="*70)

    # Generate submission file
    print("\n📄 Generating submission file...")
    submission_path = os.path.join(OUTPUT_DIR, 'maritime_detection_submission.json')
    generate_coco_submission(predictions, submission_path)

    # Save comprehensive report
    print("\n💾 Saving comprehensive evaluation report...")
    report = {
        'model_info': {
            'checkpoint_path': CHECKPOINT_PATH,
            'num_classes': NUM_CLASSES,
            'class_names': CLASS_NAMES,
            'architecture': 'YOLOv8l',
            'image_size': IMG_SIZE
        },
        'evaluation_config': {
            'confidence_threshold': float(CONFIDENCE_THRESHOLD),
            'optimal_confidence_threshold': float(optimal_conf_threshold),
            'iou_threshold': float(IOU_THRESHOLD),
            'dataset_size': len(predictions)
        },
        'metrics': {
            'mAP@0.5': float(mAP_50),
            'mAP@0.75': float(mAP_75),
            'mAP@0.5:0.95': float(mAP_coco),
            'per_class_AP@0.5': {
                CLASS_NAMES[i]: float(per_class_ap_50.get(i, 0))
                for i in range(NUM_CLASSES)
            },
            'precision@0.5': float(precision_50),
            'recall@0.5': float(recall_50),
            'f1_score@0.5': float(f1_50)
        },
        'speed_benchmark': speed_stats if speed_stats else {},
        'statistics': {
            'total_images': len(predictions),
            'total_predictions': int(total_predictions),
            'total_ground_truths': int(total_ground_truths),
            'per_class_statistics': per_class_stats,
            'images_no_detection': int(no_detection)
        },
        'threshold_sweep': {
            'tested_thresholds': [float(t) for t in sweep_results['thresholds']],
            'mAP_values': [float(m) for m in sweep_results['mAP']],
            'f1_values': [float(f) for f in sweep_results['f1']]
        },
        'confusion_matrix': cm.tolist()
    }

    report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✅ Comprehensive evaluation report saved: {report_path}")

    # Final summary
    print("\n" + "="*70)
    print("📋 FINAL EVALUATION SUMMARY")
    print("="*70)
    print(f"\n🎯 Performance Metrics:")
    print(f"   mAP@0.5:      {mAP_50:.4f}")
    print(f"   mAP@0.75:     {mAP_75:.4f}")
    print(f"   mAP@0.5:0.95: {mAP_coco:.4f}")
    print(f"   Precision:    {precision_50:.4f}")
    print(f"   Recall:       {recall_50:.4f}")
    print(f"   F1 Score:     {f1_50:.4f}")
    if speed_stats:
        print(f"\n⚡ Speed:")
        print(f"   Inference:    {speed_stats['mean_ms']:.2f} ms/image")
        print(f"   Throughput:   {speed_stats['fps']:.1f} FPS")
    print(f"\n📁 Output Files:")
    print(f"   {os.path.join(OUTPUT_DIR, 'maritime_detection_submission.json')}")
    print(f"   {os.path.join(OUTPUT_DIR, 'evaluation_report.json')}")
    print(f"   {os.path.join(OUTPUT_DIR, 'per_image_logs.json')}")
    print(f"   {os.path.join(OUTPUT_DIR, 'threshold_sweep.png')}")
    print(f"   {os.path.join(OUTPUT_DIR, 'confusion_matrix.png')}")
    print("="*70)
    print("\n✅ EVALUATION COMPLETE!")


if __name__ == "__main__":
    main()
