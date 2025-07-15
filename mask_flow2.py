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
     
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]  # Confidence information from the optical flow model
    return flow_final, info_final

def calc_flow(args, model, image1, image2):
    # Determine scale factor based on input image size
    H, W = image1.shape[2:]
    target_size = args.target_size
    
    # Calculate adaptive scale factor
    h_scale = target_size / H
    w_scale = target_size / W
    scale_factor = min(h_scale, w_scale)
    
    # Only scale up if necessary (small images)
    if scale_factor < 1.0:
        scale_factor = 1.0
    
    # Apply scale with maximum limits to prevent memory issues
    if scale_factor > 1.0:
        scale_factor = min(scale_factor, args.max_scale)
        img1 = F.interpolate(image1, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    else:
        img1 = image1
        img2 = image2
    
    # Check if the image is too small
    if img1.shape[2] < 4 or img1.shape[3] < 4:
        print(f"Warning: Image too small for flow calculation. Using original images.")
        img1 = image1
        img2 = image2
        scale_factor = 1.0
    
    flow, info = forward_flow(args, model, img1, img2)
    
    # Only downsample if we upsampled earlier
    if scale_factor > 1.0:
        down_factor = 1.0 / scale_factor
        flow_down = F.interpolate(flow, scale_factor=down_factor, mode='bilinear', align_corners=False) * down_factor
        info_down = F.interpolate(info, scale_factor=down_factor, mode='area')
    else:
        flow_down = flow
        info_down = info
    
    return flow_down, info_down
    
def warp_image(image, flow, confidence=None, confidence_threshold=0.5):
    """
    Warp an image using the computed flow field.
    image: Tensor of shape [B, C, H, W]
    flow: Tensor of shape [B, 2, H, W] 
    confidence: Optional tensor of shape [B, 1, H, W] with flow confidence
    confidence_threshold: Threshold for valid flow
    Returns:
      Warped image of shape [B, C, H, W] and validity mask
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
    
    # Create validity mask for out-of-bounds pixels
    valid_x = (new_grid[..., 0] >= 0) & (new_grid[..., 0] <= W - 1)
    valid_y = (new_grid[..., 1] >= 0) & (new_grid[..., 1] <= H - 1)
    valid_mask = (valid_x & valid_y).float().unsqueeze(1)
    
    # Apply confidence threshold if provided
    
    
    ## scale grid to [-1,1]
    new_grid[..., 0] = 2.0 * new_grid[..., 0].clone() / max(W - 1, 1) - 1.0
    new_grid[..., 1] = 2.0 * new_grid[..., 1].clone() / max(H - 1, 1) - 1.0

    # For class labels, use nearest neighbor interpolation to preserve exact class values
    warped_image = F.grid_sample(image, new_grid, align_corners=True, mode='nearest')
    
    return warped_image, valid_mask

@torch.no_grad()
def demo_data(args, model, image1, image2):
    # Check image dimensions
    H, W = image1.shape[2:]
    if H < 2 or W < 2:
        print(f"Warning: Image too small for demo_data ({H}x{W})")
        # Return zero flow tensor of the appropriate shape
        batch_size = image1.shape[0]
        return torch.zeros((batch_size, 2, H, W), device=image1.device), torch.zeros((batch_size, 1, H, W), device=image1.device)
    
    flow, info = calc_flow(args, model, image1, image2)
    return flow, info

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

def majority_vote_combine(warped_mask, curr_mask, valid_mask, class_count=None):
    """
    Combine warped and current masks using majority voting for multi-class segmentation.
    
    Args:
        warped_mask: Tensor of shape [B, 1, H, W] with warped mask from previous frame
        curr_mask: Tensor of shape [B, 1, H, W] with current frame prediction
        valid_mask: Tensor of shape [B, 1, H, W] with 1 for valid warped pixels, 0 for invalid
        class_count: Number of classes in the segmentation (including background)
    
    Returns:
        Combined mask using majority voting
    """
    device = warped_mask.device
    B, _, H, W = warped_mask.shape
    
    # If class count not provided, determine it from the masks
    if class_count is None:
        all_classes = torch.unique(torch.cat([warped_mask, curr_mask]))
        class_count = len(all_classes)
    
    # Convert masks to one-hot encoding
    warped_flat = warped_mask.long().view(B, -1)
    curr_flat = curr_mask.long().view(B, -1)
    valid_flat = valid_mask.view(B, -1)
    
    # Create one-hot encodings
    warped_one_hot = torch.zeros(B, H*W, class_count, device=device)
    curr_one_hot = torch.zeros(B, H*W, class_count, device=device)
    
    # Set one-hot values
    for b in range(B):
        warped_one_hot[b, torch.arange(H*W), warped_flat[b]] = valid_flat[b]
        curr_one_hot[b, torch.arange(H*W), curr_flat[b]] = 1.0
    
    # Combine votes with higher weight for current prediction to break ties
    combined_votes = warped_one_hot + curr_one_hot * 1.01
    
    # Get class with maximum votes
    _, combined_mask = torch.max(combined_votes, dim=2)
    
    return combined_mask.view(B, 1, H, W)

def temporal_consistency_filter(flow, confidence, consistency_threshold=3.0):
    """
    Create a mask that filters out regions with inconsistent optical flow.
    Args:
        flow: Tensor of shape [B, 2, H, W] with optical flow
        confidence: Tensor of shape [B, 1, H, W] with flow confidence
        consistency_threshold: Threshold for flow consistency
    Returns:
        Mask tensor of shape [B, 1, H, W] with 1 for consistent regions, 0 for inconsistent
    """
    # Calculate flow magnitude
    flow_magnitude = torch.sqrt(flow[:, 0, :, :]**2 + flow[:, 1, :, :]**2).unsqueeze(1)
    
    # Apply Gaussian blur to flow magnitude to detect local inconsistencies
    # We're simulating this with average pooling since we don't have spatial filters
    kernel_size = 5
    padding = kernel_size // 2
    avg_flow = F.avg_pool2d(flow_magnitude, kernel_size=kernel_size, stride=1, padding=padding)
    
    # Calculate difference between local average and pixel flow
    flow_diff = torch.abs(flow_magnitude - avg_flow)
    
    # Create consistency mask where difference is below threshold
    consistency_mask = (flow_diff < consistency_threshold).float()
    
    # Combine with confidence
    valid_mask = consistency_mask * (confidence > 0.5).float()
    
    return valid_mask

@torch.no_grad()
def mask_refiner(args, model, device=torch.device('cuda')):
    base_dir = args.input_folder
    output_dir = args.output_folder
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics tracking
    all_metrics = defaultdict(list)
    per_video_metrics = defaultdict(lambda: defaultdict(list))
    
    # Find all subdirectories that represent videos
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
            
            # Copy the single frame's prediction to output if it exists
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
                    
                    # Copy ground truth if available
                    gt_path = os.path.join(frame_dir, "ground_truth_mask.png")
                    if os.path.exists(gt_path):
                        shutil.copy(gt_path, gt_dst_path)
                    elif os.path.exists(gt_original):
                        shutil.copy(gt_original, gt_dst_path)
                    
                    # Calculate metrics for this single frame
                    gt_mask_path = os.path.join(frame_dir, "ground_truth_mask.png")
                    if os.path.exists(gt_mask_path):
                        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                        pred_mask_img = cv2.imread(pred_mask, cv2.IMREAD_GRAYSCALE)
                        
                        metrics = calculate_metrics(gt_mask, pred_mask_img)
                        
                        # Store metrics
                        for key, value in metrics.items():
                            if key != 'confusion_matrix':
                                all_metrics[key].append(value)
                                per_video_metrics[video_name][key].append(value)
            continue
        
        # For multi-frame videos, we'll use a sliding window approach
        # This allows us to use both previous and next frames when available
        temporal_window = args.temporal_window
        alpha_forward = args.alpha_forward  # Weight for current frame vs. warped previous
        alpha_backward = args.alpha_backward  # Weight for current frame vs. warped next
        
        # If we're using confidence thresholding
        confidence_threshold = args.confidence_threshold
        
        # Initialize storage for previous frame's refined mask
        prev_refined_mask = None
        
        # First, process all frames and store their warped versions
        processed_frames = []
        
        print("  Pre-processing frames for optical flow...")
        for i in range(len(frame_dirs)):
            curr_frame_dir = os.path.join(video_dir, frame_dirs[i])
            curr_frame_path = os.path.join(curr_frame_dir, "input.png")
            curr_mask_path = os.path.join(curr_frame_dir, "pred_mask.png")
            
            if not os.path.exists(curr_frame_path) or not os.path.exists(curr_mask_path):
                print(f"  Missing data for frame {frame_dirs[i]}, skipping")
                processed_frames.append(None)
                continue
            
            # Load images in original resolution
            curr_image = cv2.imread(curr_frame_path)
            curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
            
            # Keep track of original dimensions
            orig_h, orig_w = curr_image.shape[:2]
            
            # Load mask in grayscale to preserve class values
            curr_mask = cv2.imread(curr_mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Store frame info
            processed_frames.append({
                'frame_idx': i,
                'frame_dir': curr_frame_dir,
                'image': curr_image,
                'mask': curr_mask,
                'orig_dims': (orig_h, orig_w)
            })
            
            print(f"  Processed frame {i+1}/{len(frame_dirs)}", end='\r')
        
        print("\n  Computing optical flow and refining masks...")
        
        # Now process each frame using the temporal window
        for i in range(len(processed_frames)):
            if processed_frames[i] is None:
                continue
                
            curr_frame = processed_frames[i]
            curr_frame_dir = curr_frame['frame_dir']
            
            # Get ground truth for evaluation if available
            gt_mask_path = os.path.join(curr_frame_dir, "ground_truth_mask.png")
            has_gt = os.path.exists(gt_mask_path)
            
            # Convert current frame to tensor
            curr_img_tensor = torch.tensor(curr_frame['image'], dtype=torch.float32).permute(2, 0, 1)[None].to(device)
            curr_mask_tensor = torch.tensor(curr_frame['mask'], dtype=torch.float32)[None, None].to(device)
            
            # Initialize storage for warped masks
            warped_masks = []
            valid_masks = []
            
            # Process previous frames in the window
            for offset in range(1, min(temporal_window + 1, i + 1)):
                prev_idx = i - offset
                if prev_idx < 0 or processed_frames[prev_idx] is None:
                    continue
                    
                prev_frame = processed_frames[prev_idx]
                
                # Convert to tensor
                prev_img_tensor = torch.tensor(prev_frame['image'], dtype=torch.float32).permute(2, 0, 1)[None].to(device)
                prev_mask_tensor = torch.tensor(prev_frame['mask'], dtype=torch.float32)[None, None].to(device)
                
                try:
                    # Compute forward optical flow (prev → curr)
                    flow, confidence = demo_data(args, model, prev_img_tensor, curr_img_tensor)
                    print('a')
                    
                    # Filter flow for consistency
                    valid_flow = temporal_consistency_filter(flow, confidence, args.consistency_threshold)
                    print('a')
                    
                    # Warp mask using the flow
                    warped_mask, valid_warp = warp_image(prev_mask_tensor, flow, confidence, confidence_threshold)
                    print('a')
                    
                    # Combine validity masks
                    valid_mask = valid_warp * valid_flow
                    
                    # Store warped mask and validity
                    warped_masks.append(warped_mask)
                    valid_masks.append(valid_mask)
                    
                except Exception as e:
                    print(f"  Error during optical flow calculation for frames {prev_idx}-{i}: {e}")
            
            # Process next frames in the window (if available)
            for offset in range(1, min(temporal_window + 1, len(processed_frames) - i)):
                next_idx = i + offset
                if next_idx >= len(processed_frames) or processed_frames[next_idx] is None:
                    continue
                    
                next_frame = processed_frames[next_idx]
                
                # Convert to tensor
                next_img_tensor = torch.tensor(next_frame['image'], dtype=torch.float32).permute(2, 0, 1)[None].to(device)
                next_mask_tensor = torch.tensor(next_frame['mask'], dtype=torch.float32)[None, None].to(device)
                
                
                # Compute backward optical flow (next → curr)
                flow, confidence = demo_data(args, model, next_img_tensor, curr_img_tensor)
                
                # Filter flow for consistency
                valid_flow = temporal_consistency_filter(flow, confidence, args.consistency_threshold)                    
                # Warp mask using the flow
                warped_mask, valid_warp = warp_image(next_mask_tensor, flow, confidence, confidence_threshold)
                
                # Combine validity masks
                valid_mask = valid_warp * valid_flow
                Hc, Wc = curr_mask_tensor.shape[-2:]

                valid_mask  = F.interpolate(valid_mask,  size=(Hc, Wc), mode='nearest')
                warped_mask = F.interpolate(warped_mask, size=(Hc, Wc), mode='nearest')

            
                # Store warped mask and validity with slightly lower weight
                warped_masks.append(warped_mask)
                valid_masks.append(valid_mask * args.backward_weight)
                    
              
            # If we have warped masks, combine them with current prediction
            if warped_masks:
                # Combine warped masks
                if args.comb_method == 'majority_vote':
                    # Stack all masks including current
                    all_masks = torch.cat([mask for mask in warped_masks] + [curr_mask_tensor], dim=1)
                    all_valids = torch.cat([mask for mask in valid_masks] + [torch.ones_like(curr_mask_tensor)], dim=1)
                    
                    # Count unique classes
                    classes = torch.unique(all_masks).cpu().numpy()
                    class_count = len(classes)
                    
                    # Use majority vote to combine
                    stacked_masks = all_masks.unsqueeze(0)  # [1, N, 1, H, W]
                    stacked_valids = all_valids.unsqueeze(0)  # [1, N, 1, H, W]
                    
                    # Count votes for each class at each pixel
                    votes = torch.zeros((1, class_count, all_masks.shape[2], all_masks.shape[3]), device=device)
                    
                    # Special handling for current frame with higher weight
                    curr_idx = len(warped_masks)
                    
                    # Add votes weighted by validity
                    for c in range(class_count):
                        class_val = classes[c]
                        # For each input mask
                        for m in range(len(warped_masks) + 1):
                            if m == curr_idx:
                                # Current mask gets higher weight to break ties
                                weight = args.current_weight
                            else:
                                weight = stacked_valids[0, m, 0]
                            
                            # Add votes where class matches
                            votes[0, c] += ((stacked_masks[0, m, 0] == class_val).float() * weight)
                    
                    # Get class with maximum votes
                    _, refined_mask = torch.max(votes, dim=1)
                    refined_mask = refined_mask.float()
                    
                elif args.comb_method == 'weighted_average':
                    # Initialize with current mask (highest weight)
                    refined_mask = curr_mask_tensor.clone() * args.current_weight
                    total_weight = torch.ones_like(curr_mask_tensor) * args.current_weight
                    
                    # Add weighted contributions from warped masks
                    for warp_mask, valid_mask in zip(warped_masks, valid_masks):
                        refined_mask += warp_mask * valid_mask
                        total_weight += valid_mask
                    
                    # Normalize by total weight
                    refined_mask /= torch.clamp(total_weight, min=1e-5)
                    
                    # Round to nearest class for multi-class segmentation
                    refined_mask = torch.round(refined_mask)
                
                else:
                    print(f"  Unknown combination method: {args.comb_method}, using current mask")
                    refined_mask = curr_mask_tensor
            else:
                # If no warped masks, use current prediction
                refined_mask = curr_mask_tensor
            
            # Convert refined mask back to numpy
            refined_mask_np = refined_mask[0, 0].cpu().numpy().astype(np.uint8)
            
            # Create output directory structure
            relative_path = os.path.relpath(curr_frame_dir, base_dir)
            output_frame_dir = os.path.join(output_dir, relative_path)
            os.makedirs(output_frame_dir, exist_ok=True)
            
            # Save refined mask
            refined_mask_path = os.path.join(output_frame_dir, "refined_mask.png")
            cv2.imwrite(refined_mask_path, refined_mask_np)
            
            # Save ground truth for reference if available
            if has_gt:
                gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                gt_dst_path = os.path.join(output_frame_dir, "gt.png")
                cv2.imwrite(gt_dst_path, gt_mask)
                
                # Calculate metrics
                metrics = calculate_metrics(gt_mask, refined_mask_np)
                
                # Store metrics
                for key, value in metrics.items():
                    if key != 'confusion_matrix':
                        all_metrics[key].append(value)
                        per_video_metrics[video_name][key].append(value)
            
            # Update progress
            print(f"  Processed frame {i+1}/{len(processed_frames)}", end='\r')
        
        print("\n  Video processing complete")
        
        # Calculate mean metrics for this video
        if per_video_metrics[video_name]:
            print(f"  Metrics for {video_name}:")
            for key in per_video_metrics[video_name]:
                mean_val = np.mean(per_video_metrics[video_name][key])
                print(f"    {key}: {mean_val:.4f}")
    
    # Calculate overall metrics
    if all_metrics:
        print("\nOverall metrics across all videos:")
        for key in all_metrics:
            mean_val = np.mean(all_metrics[key])
            print(f"  {key}: {mean_val:.4f}")
    
    # Save metrics to CSV
    if all_metrics:
        # Per-video metrics
        video_metric_rows = []
        for video_name, metrics in per_video_metrics.items():
            row = {'video': video_name}
            for key, values in metrics.items():
                row[f'{key}_mean'] = np.mean(values)
                row[f'{key}_std'] = np.std(values)
            video_metric_rows.append(row)
        
        video_metrics_df = pd.DataFrame(video_metric_rows)
        video_metrics_path = os.path.join(output_dir, 'per_video_metrics.csv')
        video_metrics_df.to_csv(video_metrics_path, index=False)
        
        # Overall metrics
        overall_metrics = {'metric': [], 'mean': [], 'std': []}
        for key, values in all_metrics.items():
            overall_metrics['metric'].append(key)
            overall_metrics['mean'].append(np.mean(values))
            overall_metrics['std'].append(np.std(values))
        
        overall_metrics_df = pd.DataFrame(overall_metrics)
        overall_metrics_path = os.path.join(output_dir, 'overall_metrics.csv')
        overall_metrics_df.to_csv(overall_metrics_path, index=False)
        
        print(f"\nMetrics saved to {output_dir}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Mask Refiner - Apply temporal consistency to segmentation masks')
        
    # RAFT model arguments
    parser.add_argument('--iters', type=int, default=12, help='Number of iterations')
    parser.add_argument('--target-size', type=int, default=224, help='Target size for optical flow calculation')
    parser.add_argument('--max-scale', type=float, default=4.0, help='Maximum scale factor for resizing')
    
    # Refinement parameters
    parser.add_argument('--temporal-window', type=int, default=2, help='Number of frames to look ahead/behind')
    parser.add_argument('--alpha-forward', type=float, default=0.7, help='Weight for current vs warped previous')
    parser.add_argument('--alpha-backward', type=float, default=0.6, help='Weight for current vs warped next')
    parser.add_argument('--confidence-threshold', type=float, default=None, help='Threshold for flow confidence')
    parser.add_argument('--consistency-threshold', type=float, default=3.0, help='Threshold for flow consistency')
    parser.add_argument('--backward-weight', type=float, default=0.8, help='Weight for backward flow warped masks')
    parser.add_argument('--current-weight', type=float, default=1.2, help='Weight for current frame prediction')
    parser.add_argument('--comb-method', type=str, default='majority_vote', 
                       choices=['majority_vote', 'weighted_average'], 
                       help='Method to combine warped masks')
    
    parser.add_argument('--cfg', default='config/eval/spring-M.json', type=str, help='experiment configure file name')
    parser.add_argument('--path', default=None, type=str, help='checkpoint path')
    parser.add_argument('--url', default='MemorySlices/Tartan-C-T-TSKH-spring540x960-M', type=str, help='checkpoint URL')
    parser.add_argument('--input_folder', required=True, type=str, 
                        help='base directory containing video sequences and frames')
    parser.add_argument('--output_folder', default='refinement_results', type=str, 
                        help='output directory (will mirror input structure)')
    parser.add_argument('--alpha', type=float, default=0.99999997, 
                        help='alpha parameter for weighted average (weight for previous mask)')
    parser.add_argument('--device', type=str, default='cuda', help='device to run inference')
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