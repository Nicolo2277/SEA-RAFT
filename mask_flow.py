import os
import cv2 
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import pandas as pd
from collections import defaultdict

import sys
sys.path.append('core')

from config.parser import parse_args
from raft import RAFT
from utils.utils import load_ckpt

def forward_flow(args, model, image1, image2):
    # Check image dimensions before processing
    H, W = image1.shape[2:]
    if H < 2 or W < 2:
        print(f"Warning: Image too small for optical flow calculation (dimensions: {H}x{W})")
        # Return zero flow and info tensors of the appropriate shape
        batch_size = image1.shape[0]
        zero_flow = torch.zeros((batch_size, 2, H, W), device=image1.device)
        zero_info = torch.zeros((batch_size, 1, H, W), device=image1.device)
        return zero_flow, zero_info
    
    #print(image1.shape) torch.Size([1, 3, 112, 112])
    #print(image2.shape) torch.Size([1, 3, 112, 112])

    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

def calc_flow(args, model, image1, image2):
    # Apply scale factor but ensure resulting dimensions are valid
    scale_factor = 2 ** args.scale
    img1 = F.interpolate(image1, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    
    H, W = img1.shape[2:]
    #print(H) #112
    #print(W) #112
    
    
    # Check if the image is too small after scaling
    if H < 4 or W < 4:  # Need at least 4x4 to handle further downscaling in the optical flow model
        print(f"Warning: Scaled image too small ({H}x{W}). Using original image instead.")
        # Use original image as fallback
        img1 = image1
        img2 = image2
        scale_factor = 1.0
    
    flow, info = forward_flow(args, model, img1, img2)
    
    # Only downsample if we upsampled earlier and the resulting image won't be too small
    if scale_factor > 1.0:
        down_factor = 0.5 ** args.scale
        # Check if downsampling would make the image too small
        new_H, new_W = int(flow.shape[2] * down_factor), int(flow.shape[3] * down_factor)
        if new_H >= 1 and new_W >= 1:
            flow_down = F.interpolate(flow, scale_factor=down_factor, mode='bilinear', align_corners=False) * down_factor
            info_down = F.interpolate(info, scale_factor=down_factor, mode='area')
        else:
            # Skip downsampling if dimensions would be too small
            print(f"Warning: Skipping downsampling that would result in {new_H}x{new_W} dimensions")
            flow_down = flow
            info_down = info
    else:
        flow_down = flow
        info_down = info
    
    return flow_down, info_down
    
def warp_image(image, flow):
    """
    Warp an image using the computed flow field.
    image: Tensor of shape [B, C, H, W]
    flow: Tensor of shape [B, 2, H, W] 
    Returns:
      Warped image of shape [B, C, H, W]
    """
    B, C, H, W = image.shape
    
    grid_x = torch.arange(0, W).view(1, -1).repeat(H, 1)
    grid_y = torch.arange(0, H).view(-1, 1).repeat(1, W)
     
    grid_x = grid_x.view(1, H, W, 1).repeat(B, 1, 1, 1)
    grid_y = grid_y.view(1, H, W, 1).repeat(B, 1, 1, 1)
     
    grid = torch.cat((grid_x, grid_y), 3).float()
    
    if image.is_cuda:
        grid = grid.cuda()

    flow = flow.permute(0, 2, 3, 1)
    
    new_grid = grid + flow
    
    ## scale grid to [-1,1]
    new_grid[:, :, :, 0] = 2.0 * new_grid[:, :, :, 0].clone() / max(W - 1, 1) - 1.0
    new_grid[:, :, :, 1] = 2.0 * new_grid[:, :, :, 1].clone() / max(H - 1, 1) - 1.0

    # For class labels, use nearest neighbor interpolation to preserve exact class values
    warped_image = F.grid_sample(image, new_grid, align_corners=True, mode='nearest')
    
    return warped_image

@torch.no_grad()
def demo_data(args, model, image1, image2):
    # Check image dimensions
    H, W = image1.shape[2:]
    if H < 2 or W < 2:
        print(f"Warning: Image too small for demo_data ({H}x{W})")
        # Return zero flow tensor of the appropriate shape
        batch_size = image1.shape[0]
        return torch.zeros((batch_size, 2, H, W), device=image1.device)
    
    flow, _ = calc_flow(args, model, image1, image2)
    return flow

def calculate_metrics(gt_mask, pred_mask, threshold=0.5):
    """
    Calculate segmentation metrics between ground truth and predicted masks.
    Both masks are expected to be numpy arrays with class values (0, 1, 2, etc.)
    
    Returns a dictionary with:
    - dice: Dice coefficient (F1)
    - jaccard: Jaccard index (IoU)
    - precision: Precision score
    - recall: Recall score
    - accuracy: Accuracy score
    - confusion_matrix: 2x2 confusion matrix [TN, FP, FN, TP]
    """
    # Ensure masks are in the right format for comparison
    if gt_mask.ndim == 3 and gt_mask.shape[2] == 3:
        # Convert RGB to grayscale
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_RGB2GRAY)
    if pred_mask.ndim == 3 and pred_mask.shape[2] == 3:
        pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_RGB2GRAY)
    
    # Flatten masks for comparison
    gt_flat = gt_mask.flatten()
    pred_flat = pred_mask.flatten()
    
    # Get unique class values
    unique_classes = np.unique(np.concatenate([gt_flat, pred_flat]))
    
    # For binary case, we can use normal metrics
    if len(unique_classes) <= 2:
        # Convert to binary
        gt_binary = (gt_mask > threshold * np.max(gt_mask)).astype(np.uint8)
        pred_binary = (pred_mask > threshold * np.max(pred_mask)).astype(np.uint8)
        
        # Calculate intersection and union
        intersection = np.logical_and(gt_binary, pred_binary).sum()
        union = np.logical_or(gt_binary, pred_binary).sum()
        
        # Get confusion matrix values
        tn, fp, fn, tp = confusion_matrix(gt_binary.flatten(), pred_binary.flatten(), labels=[0, 1]).ravel()
        
        # Calculate metrics
        dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
        jaccard = intersection / union if union > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'dice': dice,
            'jaccard': jaccard,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1': f1,
            'confusion_matrix': [tn, fp, fn, tp]
        }
    
    # For multi-class, calculate mean metrics across all classes
    else:
        # Calculate metrics for each class
        class_metrics = {'dice': [], 'jaccard': [], 'precision': [], 'recall': [], 'accuracy': [], 'f1': []}
        
        # For each class (except background class 0)
        for cls in unique_classes:
            if cls == 0:  # Skip background class
                continue
                
            # Create binary masks for this class
            gt_binary = (gt_mask == cls).astype(np.uint8)
            pred_binary = (pred_mask == cls).astype(np.uint8)
            
            # Calculate intersection and union
            intersection = np.logical_and(gt_binary, pred_binary).sum()
            union = np.logical_or(gt_binary, pred_binary).sum()
            
            # Get confusion matrix values
            tn, fp, fn, tp = confusion_matrix(gt_binary.flatten(), pred_binary.flatten(), labels=[0, 1]).ravel()
            
            # Calculate metrics
            dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
            jaccard = intersection / union if union > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Store metrics for this class
            class_metrics['dice'].append(dice)
            class_metrics['jaccard'].append(jaccard)
            class_metrics['precision'].append(precision)
            class_metrics['recall'].append(recall)
            class_metrics['accuracy'].append(accuracy)
            class_metrics['f1'].append(f1)
        
        # Calculate mean metrics across classes
        mean_metrics = {
            'dice': np.mean(class_metrics['dice']) if class_metrics['dice'] else 0.0,
            'jaccard': np.mean(class_metrics['jaccard']) if class_metrics['jaccard'] else 0.0,
            'precision': np.mean(class_metrics['precision']) if class_metrics['precision'] else 0.0,
            'recall': np.mean(class_metrics['recall']) if class_metrics['recall'] else 0.0,
            'accuracy': np.mean(class_metrics['accuracy']) if class_metrics['accuracy'] else 0.0,
            'f1': np.mean(class_metrics['f1']) if class_metrics['f1'] else 0.0,
            'confusion_matrix': [0, 0, 0, 0]  # Not meaningful for multi-class
        }
        
        return mean_metrics

@torch.no_grad()
def mask_refiner(args, model, device=torch.device('cuda')):
    base_dir = args.input_folder
    output_dir = args.output_folder
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics tracking
    all_metrics = defaultdict(list)
    per_video_metrics = defaultdict(lambda: defaultdict(list))
    
    # Find all subdirectories that represent videos
    # Format: Directory/folder/folder/numberofframe/...
    video_dirs = []
    for root, dirs, _ in os.walk(base_dir):
        # Check if this directory contains frame subdirectories
        has_frame_dirs = any(d.isdigit() for d in dirs)
        if has_frame_dirs:
            video_dirs.append(root)
    
    print(f"Found {len(video_dirs)} video sequences")
    
    # Process each video sequence
    for video_dir in video_dirs:
        video_name = os.path.basename(video_dir)
        
        # Get sorted frame directories (they're numbered)
        frame_dirs = sorted([d for d in os.listdir(video_dir) if d.isdigit()], 
                          key=lambda x: int(x))
        
        print(f"Processing video in {video_dir} with {len(frame_dirs)} frames")
        
        # Skip videos with only one frame
        if len(frame_dirs) <= 1:
            print(f"  Skipping - single frame video")
            
            # Optionally copy the single frame's prediction to output
            if len(frame_dirs) == 1:
                frame_dir = os.path.join(video_dir, frame_dirs[0])
                gt_original = os.path.join(frame_dir, 'input.png')
                pred_mask = os.path.join(frame_dir, "pred_mask.png")
                if os.path.exists(pred_mask):
                    # Create equivalent output directory structure
                    relative_path = os.path.relpath(video_dir, base_dir)
                    output_video_dir = os.path.join(output_dir, relative_path)
                    os.makedirs(os.path.join(output_video_dir, frame_dirs[0]), exist_ok=True)
                    
                    dst_path = os.path.join(output_video_dir, frame_dirs[0], "refined_mask.png")
                    gt_dst_path = os.path.join(output_video_dir, frame_dirs[0], "gt.png")
                    import shutil
                    shutil.copy(pred_mask, dst_path)
                    shutil.copy(gt_original, gt_dst_path)
                    print(f"  Copied mask to {dst_path}")
                    
                    # Calculate metrics for this single frame
                    gt_mask_path = os.path.join(frame_dir, "ground_truth_mask.png")
                    if os.path.exists(gt_mask_path):
                        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                        pred_mask = cv2.imread(pred_mask, cv2.IMREAD_GRAYSCALE)
                        
                        metrics = calculate_metrics(gt_mask, pred_mask)
                        
                        # Store metrics
                        for key, value in metrics.items():
                            if key != 'confusion_matrix':
                                all_metrics[key].append(value)
                                per_video_metrics[video_name][key].append(value)
            continue
        
        # Process consecutive frames
        for i in range(1, len(frame_dirs)):
            prev_frame_dir = os.path.join(video_dir, frame_dirs[i-1])
            curr_frame_dir = os.path.join(video_dir, frame_dirs[i])
            
            # Load original frames for optical flow computation
            prev_frame_path = os.path.join(prev_frame_dir, "input.png")
            curr_frame_path = os.path.join(curr_frame_dir, "input.png")
            
            if not os.path.exists(prev_frame_path) or not os.path.exists(curr_frame_path):
                print(f"  Error: Missing frame images for frames {frame_dirs[i-1]}-{frame_dirs[i]}")
                continue
                
            # Load predicted masks to be refined
            prev_mask_path = os.path.join(prev_frame_dir, "pred_mask.png")
            curr_mask_path = os.path.join(curr_frame_dir, "pred_mask.png")
            
            if not os.path.exists(prev_mask_path) or not os.path.exists(curr_mask_path):
                print(f"  Error: Missing predicted masks for frames {frame_dirs[i-1]}-{frame_dirs[i]}")
                continue
            
            # Load ground truth for evaluation
            gt_mask_path = os.path.join(curr_frame_dir, "ground_truth_mask.png")
            has_gt = os.path.exists(gt_mask_path)
            
            # Load images
            prev_image = cv2.imread(prev_frame_path)
            prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2RGB)
            prev_image = cv2.resize(
            prev_image,
            (224, 224),                     # (width, height)
            interpolation=cv2.INTER_LINEAR   
            )
            curr_image = cv2.imread(curr_frame_path)
            curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
            curr_image = cv2.resize(
            curr_image,
            (224, 224),                     # (width, height)
            interpolation=cv2.INTER_LINEAR   
            )
            
            
            # Load masks - preserve class values by loading in grayscale mode
            prev_mask = cv2.imread(prev_mask_path, cv2.IMREAD_GRAYSCALE)
            curr_mask = cv2.imread(curr_mask_path, cv2.IMREAD_GRAYSCALE)
            
            # For optical flow processing, we need RGB tensors
            # Convert to 3-channel but preserve class information
            prev_mask_rgb = np.stack([prev_mask, prev_mask, prev_mask], axis=2)
            curr_mask_rgb = np.stack([curr_mask, curr_mask, curr_mask], axis=2)
            
            # Get image dimensions and check if they're too small
            h, w = prev_image.shape[:2]
            if h < 4 or w < 4:
                print(f"  Warning: Image dimensions are too small ({h}x{w}) for frame {frame_dirs[i-1]}")
                print(f"  Skipping optical flow calculation and copying original mask instead")
                
                # Copy the original mask as the refined mask
                relative_path = os.path.relpath(video_dir, base_dir)
                output_video_dir = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.join(output_video_dir, frame_dirs[i]), exist_ok=True)
                
                out_path = os.path.join(output_video_dir, frame_dirs[i], "refined_mask.png")
                curr_mask_bgr = cv2.cvtColor(curr_mask, cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_path, curr_mask_bgr)
                
                # Calculate metrics if ground truth is available
                if has_gt:
                    gt_mask = cv2.imread(gt_mask_path)
                    metrics = calculate_metrics(gt_mask, curr_mask_bgr)
                    
                    # Store metrics as both original and refined (since they're the same)
                    for key, value in metrics.items():
                        if key != 'confusion_matrix':
                            all_metrics[f'refined_{key}'].append(value)
                            all_metrics[f'original_{key}'].append(value)
                            per_video_metrics[video_name][f'refined_{key}'].append(value)
                            per_video_metrics[video_name][f'original_{key}'].append(value)
                
                continue
            
            # Convert to tensors
            prev_img_tensor = torch.tensor(prev_image, dtype=torch.float32).permute(2, 0, 1)[None].to(device)
            curr_img_tensor = torch.tensor(curr_image, dtype=torch.float32).permute(2, 0, 1)[None].to(device)
            
            # Convert masks to tensors, keeping original class values intact
            # Use single-channel tensors for masks
            prev_mask_tensor = torch.tensor(prev_mask, dtype=torch.float32)[None, None].to(device)
            curr_mask_tensor = torch.tensor(curr_mask, dtype=torch.float32)[None, None].to(device)
            
            try:
                # Compute optical flow using original images
                flow = demo_data(args, model, prev_img_tensor, curr_img_tensor)
                
                # Warp current mask using the flow
                warped_mask = warp_image(prev_mask_tensor, flow)
                
                # Combine masks
                alpha = args.alpha
                '''
                # turn single‑channel into one‑hot
                curr_oh = F.one_hot(curr_mask_tensor.long().squeeze(1), 3).permute(0,3,1,2).float()
                warp_oh = F.one_hot(warped_mask.long().squeeze(1), 3).permute(0,3,1,2).float()

                combined_oh = alpha * curr_oh + (1 - alpha) * warp_oh
                refined = combined_oh.argmax(dim=1, keepdim=True)  # [B,1,H,W]

                final_mask_np = refined.squeeze().cpu().numpy().astype(np.uint8)

                '''
                combined = alpha * curr_mask_tensor + (1 - alpha) * warped_mask

                final_mask_np = torch.round(combined).squeeze().cpu().numpy().astype(np.uint8)

                
                # Save result with equivalent directory structure in output folder
                relative_path = os.path.relpath(video_dir, base_dir)
                output_video_dir = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.join(output_video_dir, frame_dirs[i]), exist_ok=True)
                
                # Convert back to numpy for saving
                
                # Save the refined mask
                out_path = os.path.join(output_video_dir, frame_dirs[i], "refined_mask.png")
                out_gt_path = os.path.join(output_video_dir, frame_dirs[i], "gt.png")
                cv2.imwrite(out_path, final_mask_np)
                cv2.imwrite(out_gt_path, gt_mask)
                print(f"  Refined mask for frame {frame_dirs[i]}")
                
            except Exception as e:
                print(f"  Error during optical flow calculation for frame {frame_dirs[i]}: {e}")
                print(f"  Using original mask instead")
                
                # Copy the original mask as the refined mask
                relative_path = os.path.relpath(video_dir, base_dir)
                output_video_dir = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.join(output_video_dir, frame_dirs[i]), exist_ok=True)
                
                out_path = os.path.join(output_video_dir, frame_dirs[i], "refined_mask.png")
                cv2.imwrite(out_path, curr_mask)
                
                # Calculate metrics if ground truth is available
                if has_gt:
                    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                    metrics = calculate_metrics(gt_mask, curr_mask)
                    
                    # Store metrics as both original and refined (since they're the same)
                    for key, value in metrics.items():
                        if key != 'confusion_matrix':
                            all_metrics[f'refined_{key}'].append(value)
                            all_metrics[f'original_{key}'].append(value)
                            per_video_metrics[video_name][f'refined_{key}'].append(value)
                            per_video_metrics[video_name][f'original_{key}'].append(value)
            
            # Calculate metrics if ground truth is available
            if has_gt:
                gt_mask = cv2.imread(gt_mask_path)
                
                # Metrics for original prediction
                original_metrics = calculate_metrics(gt_mask, cv2.cvtColor(curr_mask, cv2.COLOR_RGB2BGR))
                
                # Metrics for refined mask
                refined_metrics = calculate_metrics(gt_mask, final_mask_np)
                
                # Store metrics
                for key, value in refined_metrics.items():
                    if key != 'confusion_matrix':
                        all_metrics[f'refined_{key}'].append(value)
                        per_video_metrics[video_name][f'refined_{key}'].append(value)
                        
                for key, value in original_metrics.items():
                    if key != 'confusion_matrix':
                        all_metrics[f'original_{key}'].append(value)
                        per_video_metrics[video_name][f'original_{key}'].append(value)
                
                # Print improvement
                dice_improvement = refined_metrics['dice'] - original_metrics['dice']
                iou_improvement = refined_metrics['jaccard'] - original_metrics['jaccard']
                print(f"    Dice: {original_metrics['dice']:.4f} → {refined_metrics['dice']:.4f} ({dice_improvement:+.4f})")
                print(f"    IoU:  {original_metrics['jaccard']:.4f} → {refined_metrics['jaccard']:.4f} ({iou_improvement:+.4f})")
    
    # Save overall metrics
    metrics_df = pd.DataFrame({k: np.mean(v) for k, v in all_metrics.items() if len(v) > 0}, index=[0])
    metrics_df.to_csv(os.path.join(output_dir, 'overall_metrics.csv'), index=False)
    
    # Save per-video metrics
    video_dfs = []
    for video_name, metrics in per_video_metrics.items():
        video_df = pd.DataFrame({k: np.mean(v) for k, v in metrics.items() if len(v) > 0}, index=[0])
        video_df['video'] = video_name
        video_dfs.append(video_df)
    
    if video_dfs:
        all_videos_df = pd.concat(video_dfs)
        all_videos_df.to_csv(os.path.join(output_dir, 'per_video_metrics.csv'), index=False)
    
    # Print summary
    print("\n=== METRICS SUMMARY ===")
    for metric in ['dice', 'jaccard', 'precision', 'recall', 'f1']:
        if f'original_{metric}' in all_metrics and f'refined_{metric}' in all_metrics:
            orig_val = np.mean(all_metrics[f'original_{metric}'])
            refined_val = np.mean(all_metrics[f'refined_{metric}'])
            improvement = refined_val - orig_val
            print(f"{metric.capitalize():10}: {orig_val:.4f} → {refined_val:.4f} ({improvement:+.4f})")
    
    print(f"\nDetailed metrics saved to {output_dir}/overall_metrics.csv and {output_dir}/per_video_metrics.csv")
    
    # Return overall metrics for analysis
    return {k: np.mean(v) for k, v in all_metrics.items() if len(v) > 0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='config/eval/spring-M.json', type=str, help='experiment configure file name')
    parser.add_argument('--path', default=None, type=str, help='checkpoint path')
    parser.add_argument('--url', default='MemorySlices/Tartan-C-T-TSKH-spring540x960-M', type=str, help='checkpoint URL')
    parser.add_argument('--input_folder', required=True, type=str, 
                        help='base directory containing video sequences and frames')
    parser.add_argument('--output_folder', default='refinement_results', type=str, 
                        help='output directory (will mirror input structure)')
    parser.add_argument('--alpha', type=float, default=0.74, 
                        help='alpha parameter for weighted average (weight for previous mask)')
    parser.add_argument('--device', type=str, default='cuda', help='device to run inference')
    parser.add_argument('--comb_method', type=str, default='weighted_average', 
                        help='mode (weighted average or majority voting) for mask combining')
    parser.add_argument('--save_comparison_visualizations', action='store_true',
                        help='save side-by-side visualizations of original, refined and ground truth masks')
    parser.add_argument('--scale', type=int, default=1)
    
    args = parse_args(parser=parser)
    
    # Input validation
    if args.path is None and args.url is None:
        raise ValueError('Either --path or --url must be provided')
    if args.path is not None:
        model = RAFT(args)
        load_ckpt(model, args.path)
    else:
        model = RAFT.from_pretrained(args.url, args=args)
    
    if args.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    if args.alpha > 1 or args.alpha < 0:
        raise ValueError('Alpha must be in [0,1]')
    
    model = model.to(device)
    model.eval()
    metrics = mask_refiner(args=args, model=model, device=device)
    
    print("\nOptical flow mask refinement completed successfully!")

if __name__ == '__main__':
    main()
