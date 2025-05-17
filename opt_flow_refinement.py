import os
import cv2 
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.append('core')

from config.parser import parse_args
from raft import RAFT
from utils.utils import load_ckpt

def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    #print("Upscaled image shape:", img1.shape)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    #print("Upscaled image shape:", img1.shape)
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
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

    warped_image = F.grid_sample(image, new_grid, align_corners=True, mode='nearest')
    
    return warped_image
 
 
@torch.no_grad()
def demo_data(args, model, image1, image2):
    #os.system(f"mkdir -p {path}")
    flow, _ = calc_flow(args, model, image1, image2)
    return flow

@torch.no_grad()
def mask_refiner(args, model, device=torch.device('cuda')):
    mask_folder = args.mask_folder
    input_folder = args.input_folder
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    #Get a sorted list of mask files and img files (assumes filenames are in order)
    image_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder)
                         if f.endswith('.png') or f.endswith('.jpg')])
    mask_files = sorted([os.path.join(mask_folder, f) for f in os.listdir(mask_folder)
                         if f.endswith('.png') or f.endswith('.jpg')])

    for t in range(1, len(mask_files)):#Here i am assuming that mask[i] is the mask of img i
        previous_image = cv2.imread(image_files[t - 1])
        previous_image = cv2.cvtColor(previous_image, cv2.COLOR_BGR2RGB)
        current_image = cv2.imread(image_files[t])
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        previous_mask = cv2.imread(mask_files[t - 1]) 
        previous_mask = cv2.cvtColor(previous_mask, cv2.COLOR_BGR2RGB)
        current_mask = cv2.imread(mask_files[t])
        current_mask = cv2.cvtColor(current_mask, cv2.COLOR_BGR2RGB)
        
        if previous_image is None:
            print(f'Error reading the {t-1} image')
        elif current_image is None:
            print(f'Error reading the {t} image')
       
        if previous_mask is None:
            print(f'Error reading the {t-1} mask')
        elif current_mask is None:
            print(f'Error reading the {t} mask')
       
        previous_image_tensor = torch.tensor(previous_image, dtype=torch.float32).permute(2, 0, 1)
        current_image_tensor = torch.tensor(current_image, dtype=torch.float32).permute(2, 0, 1)
        previous_mask_tensor = torch.tensor(previous_mask, dtype=torch.float32).permute(2, 0, 1)
        current_mask_tensor = torch.tensor(current_mask, dtype=torch.float32).permute(2, 0, 1)
        
        previous_image_tensor = previous_image_tensor[None].to(device)
        current_image_tensor = current_image_tensor[None].to(device)
        previous_mask_tensor = previous_mask_tensor[None].to(device)
        current_mask_tensor = current_mask_tensor[None].to(device)
        
        flow = demo_data(args, model, previous_image_tensor, current_image_tensor)
        
        warped_mask = warp_image(current_mask_tensor, flow)
                
        #Combined the warped mask with the current one via weighted average, 
        #alpha for current, 1-alpha for average, default is alpha=0.5
        alpha = args.alpha
        combined_mask = alpha * previous_mask_tensor + (1 - alpha) * warped_mask #shape: [B, C, H, W]
        final_mask = combined_mask.squeeze() #shape: [C, H, W]
        #print(final_mask.shape)
            
        #Back to numpy array:
        final_mask_np = final_mask.cpu().numpy().astype(np.uint8)
        
        final_mask_np = np.transpose(final_mask_np, (1, 2, 0))
        
        #So it's correctly formatted
        final_mask_np = cv2.cvtColor(final_mask_np, cv2.COLOR_RGB2BGR)
        
        out_path = os.path.join(args.output_folder, f"refined_mask{t-1}.png")
        cv2.imwrite(out_path, final_mask_np)
        print(f"Saved combined mask: {out_path}")
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='experiment configure file name')
    parser.add_argument('--path', default=None, type=str, help='checkpoint path')
    parser.add_argument('--url', default=None, type=str, help='checkpoint URL')
    parser.add_argument('--input_folder', required=True, type=str, help='input images folder')
    parser.add_argument('--mask_folder', required=True, type=str, help='segmentation masks folder')
    parser.add_argument('--output_folder', required=True, type=str, help='model output folder')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha parameter for  weighted average')
    parser.add_argument('--device', type=str, default='cpu', help='device to run inference')
    args = parse_args(parser=parser)
        
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
    mask_refiner(args=args, model=model, device=device)


if __name__ == '__main__':
    main()
    
                    
