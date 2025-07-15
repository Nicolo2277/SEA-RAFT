import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import confusion_matrix, jaccard_score
import seaborn as sns
import pandas as pd

def dice_coefficient(y_true, y_pred, class_id):
    """
    Compute Dice coefficient for a specific class (equivalent to second script)
    """
    # Create binary masks for the specific class
    y_true_class = (y_true == class_id).astype(np.int32)
    y_pred_class = (y_pred == class_id).astype(np.int32)
    
    # Calculate intersection and sums
    intersection = np.sum(y_true_class & y_pred_class)
    pred_sum = np.sum(y_pred_class)
    gt_sum = np.sum(y_true_class)
    total = pred_sum + gt_sum
    
    # Avoid division by zero - if both are empty, perfect match
    if total == 0:
        return 1.0
    
    return 2.0 * intersection / total

def jaccard_coefficient(y_true, y_pred, class_id):
    """
    Compute Jaccard coefficient (IoU) for a specific class
    """
    # Create binary masks for the specific class
    y_true_class = (y_true == class_id).astype(np.int32)
    y_pred_class = (y_pred == class_id).astype(np.int32)
    
    # Calculate intersection and union
    intersection = np.sum(y_true_class & y_pred_class)
    union = np.sum(y_true_class | y_pred_class)
    
    # Avoid division by zero - if both are empty, perfect match
    if union == 0:
        return 1.0
    
    return intersection / union

def precision_coefficient(y_true, y_pred, class_id):
    """
    Compute Precision for a specific class
    """
    # Create binary masks for the specific class
    y_true_class = (y_true == class_id).astype(np.int32)
    y_pred_class = (y_pred == class_id).astype(np.int32)
    
    # Calculate true positives and predicted positives
    true_positives = np.sum(y_true_class & y_pred_class)
    predicted_positives = np.sum(y_pred_class)
    
    # Avoid division by zero
    if predicted_positives == 0:
        return 1.0 if np.sum(y_true_class) == 0 else 0.0
    
    return true_positives / predicted_positives

def recall_coefficient(y_true, y_pred, class_id):
    """
    Compute Recall (Sensitivity) for a specific class
    """
    # Create binary masks for the specific class
    y_true_class = (y_true == class_id).astype(np.int32)
    y_pred_class = (y_pred == class_id).astype(np.int32)
    
    # Calculate true positives and actual positives
    true_positives = np.sum(y_true_class & y_pred_class)
    actual_positives = np.sum(y_true_class)
    
    # Avoid division by zero
    if actual_positives == 0:
        return 1.0 if np.sum(y_pred_class) == 0 else 0.0
    
    return true_positives / actual_positives

def f_measure_coefficient(y_true, y_pred, class_id):
    """
    Compute F-measure (F1-score) for a specific class
    """
    precision = precision_coefficient(y_true, y_pred, class_id)
    recall = recall_coefficient(y_true, y_pred, class_id)
    
    # Avoid division by zero
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)

def specificity_coefficient(y_true, y_pred, class_id):
    """
    Compute Specificity for a specific class
    """
    # Create binary masks for the specific class
    y_true_class = (y_true == class_id).astype(np.int32)
    y_pred_class = (y_pred == class_id).astype(np.int32)
    
    # Calculate true negatives and actual negatives
    true_negatives = np.sum((~y_true_class.astype(bool)) & (~y_pred_class.astype(bool)))
    actual_negatives = np.sum(~y_true_class.astype(bool))
    
    # Avoid division by zero
    if actual_negatives == 0:
        return 1.0 if np.sum(~y_pred_class.astype(bool)) == 0 else 0.0
    
    return true_negatives / actual_negatives

def load_mask(file_path):
    """
    Load mask with proper handling based on file type
    """
    if file_path.endswith('.npy'):
        mask = np.load(file_path)
    else:
        # When reading image files, ensure proper class encoding is maintained
        mask = plt.imread(file_path)
        
        # Check if the image has been loaded as float (0-1) instead of int (0,1,2)
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            # If it's a grayscale image with values 0-1, convert to our class indices
            if mask.max() <= 1.0:
                # Scale to 0-255 and round to nearest integer
                mask = np.round(mask * 255).astype(np.int64)
            
        # If image has multiple channels, take first channel (assuming class labels)
        if len(mask.shape) > 2 and mask.shape[2] > 1:
            mask = mask[:,:,0]
    
    # Ensure mask is integer type
    mask = mask.astype(np.int64)
    
    return mask

def compute_comprehensive_metrics(gt_path, pred_path, class_names):
    """
    Compute comprehensive metrics for a pair of ground truth and prediction masks
    """
    # Load the images with proper handling
    gt = load_mask(gt_path)
    pred = load_mask(pred_path)
    
    # Print debugging info for the first few files
    print(f"GT shape: {gt.shape}, dtype: {gt.dtype}, unique values: {np.unique(gt)}")
    print(f"Pred shape: {pred.shape}, dtype: {pred.dtype}, unique values: {np.unique(pred)}")
    
    # Check if shapes match
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: GT {gt.shape} vs Pred {pred.shape}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(gt.flatten(), pred.flatten(), labels=range(len(class_names)))
    
    # Calculate all metrics for each class
    metrics = {
        "dice": {},
        "jaccard": {},
        "precision": {},
        "recall": {},
        "f_measure": {},
        "specificity": {}
    }
    
    for i, class_name in enumerate(class_names):
        metrics["dice"][class_name] = dice_coefficient(gt, pred, i)
        metrics["jaccard"][class_name] = jaccard_coefficient(gt, pred, i)
        metrics["precision"][class_name] = precision_coefficient(gt, pred, i)
        metrics["recall"][class_name] = recall_coefficient(gt, pred, i)
        metrics["f_measure"][class_name] = f_measure_coefficient(gt, pred, i)
        metrics["specificity"][class_name] = specificity_coefficient(gt, pred, i)
    
    # Calculate mean metrics (excluding background - classes 1 and 2 only, following second script)
    for metric_name in metrics:
        if len(class_names) > 1:
            metrics[metric_name]["mean"] = np.mean([metrics[metric_name][class_names[i]] for i in range(1, len(class_names))])
        else:
            metrics[metric_name]["mean"] = list(metrics[metric_name].values())[0]
    
    return cm, metrics

def process_directories(pred_folder, gt_folder, class_names):
    """
    Process all prediction images and corresponding ground truth from separate folders
    """
    total_cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    
    # Initialize metrics storage
    all_metrics = {
        "dice": {class_name: [] for class_name in class_names + ["mean"]},
        "jaccard": {class_name: [] for class_name in class_names + ["mean"]},
        "precision": {class_name: [] for class_name in class_names + ["mean"]},
        "recall": {class_name: [] for class_name in class_names + ["mean"]},
        "f_measure": {class_name: [] for class_name in class_names + ["mean"]},
        "specificity": {class_name: [] for class_name in class_names + ["mean"]}
    }
    
    results = []
    processed_count = 0
    skipped_count = 0
    
    # Find all pred_mask.png files in the prediction folder
    pred_files = []
    for root, _, files in os.walk(pred_folder):
        if "pred_mask.png" in files:
            pred_files.append(os.path.join(root, "pred_mask.png"))
    
    print(f"Found {len(pred_files)} prediction files")
    
    # Look at a sample of the prediction files
    if pred_files:
        sample_pred = load_mask(pred_files[0])
        print(f"Sample prediction mask: shape={sample_pred.shape}, dtype={sample_pred.dtype}, unique values={np.unique(sample_pred)}")
    
    # Process each prediction file
    debug_count = 0
    for pred_file in pred_files:
        # Extract relative path to construct corresponding GT path
        rel_path = os.path.relpath(os.path.dirname(pred_file), pred_folder)
        gt_dir = os.path.join(gt_folder, rel_path)
        gt_file = os.path.join(gt_dir, "ground_truth_mask.png")
        
        # Debugging info for the first few files
        debug_mode = debug_count < 3
        if debug_mode:
            print(f"\nProcessing pair #{debug_count + 1}:")
            print(f"Pred file: {pred_file}")
            print(f"GT file: {gt_file}")
            debug_count += 1
        
        # Check if corresponding GT file exists
        if os.path.exists(gt_file):
            try:
                # Compute comprehensive metrics
                cm, metrics = compute_comprehensive_metrics(gt_file, pred_file, class_names)
                
                if debug_mode:
                    print(f"Confusion matrix: \n{cm}")
                    print(f"Comprehensive metrics: {metrics}")
                
                # Add to totals
                total_cm += cm
                for metric_name in all_metrics:
                    for class_name in metrics[metric_name]:
                        all_metrics[metric_name][class_name].append(metrics[metric_name][class_name])
                
                # Store individual results
                result = {'file': rel_path}
                for metric_name in metrics:
                    for class_name in metrics[metric_name]:
                        result[f'{metric_name}_{class_name}'] = metrics[metric_name][class_name]
                
                results.append(result)
                processed_count += 1
                
                # Print progress
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} files...")
                    
            except Exception as e:
                print(f"Error processing {pred_file} with {gt_file}: {e}")
                skipped_count += 1
        else:
            print(f"Warning: Ground truth file not found: {gt_file}")
            skipped_count += 1
    
    # Calculate average metrics
    final_metrics = {}
    for metric_name in all_metrics:
        final_metrics[metric_name] = {}
        for class_name in all_metrics[metric_name]:
            scores = all_metrics[metric_name][class_name]
            final_metrics[metric_name][class_name] = np.mean(scores) if scores else 0.0
    
    print(f"Processed {processed_count} image pairs, skipped {skipped_count} due to missing ground truth files")
    
    return total_cm, final_metrics, results

def plot_confusion_matrix(cm, class_names, output_file="confusion_matrix.png"):
    """
    Plot and save confusion matrix
    """
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)  # Add small epsilon to avoid division by zero
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    plt.title('Row-Normalized Confusion Matrix (Recall)')
    plt.savefig('row_conf_matrix.png')
    plt.close()

    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / (cm.sum(axis=0)[np.newaxis, :] + 1e-10)  # Add small epsilon to avoid division by zero
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    plt.title('Column-Normalized Confusion Matrix (Precision)')
    plt.savefig('col_conf_matrix.png')
    plt.close()
    
    # Also plot the raw counts
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    plt.title('Confusion Matrix (Raw Counts)')
    plt.savefig("raw_" + output_file)
    plt.close()

def main():
    # Configuration
    pred_folder = "multiclass-segmentation-pred"  # Folder containing prediction masks
    gt_folder = "multiclass-segmentation-pred"      # UPDATED: Changed to separate GT folder
    class_names = ["Background", "Solid", "NonSolid"]
    
    # Import warnings for suppressing sklearn warnings
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning
    
    # Suppress sklearn warnings about jaccard score
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    
    print(f"Processing images from prediction folder: {pred_folder}")
    print(f"Using ground truth from: {gt_folder}")
    
    # Process all images and compute comprehensive metrics
    total_cm, final_metrics, results = process_directories(pred_folder, gt_folder, class_names)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(total_cm, class_names)
    
    # Print comprehensive metrics
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*60)
    
    metric_names = ["dice", "jaccard", "precision", "recall", "f_measure", "specificity"]
    metric_display_names = ["Dice Coefficient", "Jaccard Index (IoU)", "Precision", "Recall", "F-Measure", "Specificity"]
    
    for metric_name, display_name in zip(metric_names, metric_display_names):
        print(f"\n{display_name}:")
        for class_name in class_names:
            print(f"  {class_name}: {final_metrics[metric_name][class_name]:.4f}")
        print(f"  Mean (excluding background): {final_metrics[metric_name]['mean']:.4f}")
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv("comprehensive_segmentation_metrics.csv", index=False)
        print(f"\nDetailed results saved to 'comprehensive_segmentation_metrics.csv'")
        print(f"Total evaluated image pairs: {len(results)}")
        
        # Also save summary metrics
        summary_data = []
        for metric_name in metric_names:
            for class_name in final_metrics[metric_name]:
                summary_data.append({
                    'metric': metric_name,
                    'class': class_name,
                    'value': final_metrics[metric_name][class_name]
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("summary_metrics.csv", index=False)
        print("Summary metrics saved to 'summary_metrics.csv'")
    else:
        print("\nNo matching files found for evaluation")

if __name__ == "__main__":
    main()