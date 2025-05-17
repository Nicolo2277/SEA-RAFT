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
    #To be modified we need input folders for both masks and images
    input_folder = args.input_folder
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    #Get a sorted list of mask files (assumes filenames are in order)
    mask_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder)
                         if f.endswith('.png') or f.endswith('.jpg')])
    #print(mask_files)
    for t in range(1, len(mask_files)):
        #Here we need to also include images 
        previous_mask = cv2.imread(mask_files[t-1]) #In BRG format, so we convert it to RGB
        previous_mask = cv2.cvtColor(previous_mask, cv2.COLOR_BGR2RGB)
        current_mask = cv2.imread(mask_files[t])
        current_mask = cv2.cvtColor(current_mask, cv2.COLOR_BGR2RGB)
        if previous_mask is None:
            print(f'Error reading the {t-1} mask')
        elif current_mask is None:
            print(f'Error reading the {t} mask')
       
        previous_mask_tensor = torch.tensor(previous_mask, dtype=torch.float32).permute(2, 0, 1)
        current_mask_tensor = torch.tensor(current_mask, dtype=torch.float32).permute(2, 0, 1)
        
        previous_mask_tensor = previous_mask_tensor[None].to(device)
        current_mask_tensor = current_mask_tensor[None].to(device)
        
        #The flow must be computed between pairs of images, not masks
        flow = demo_data(args, model, previous_mask_tensor, current_mask_tensor)
        
        warped_image = warp_image(current_mask_tensor, flow)
        
        if args.comb_method == 'majority_voting':#Can erase this, not worth it with 2 frames
                previous_mask_tensor = previous_mask_tensor.cpu()
                current_mask_tensor = current_mask_tensor.cpu()
                
                current_labels = torch.argmax(current_mask_tensor, dim=1)
                previous_labels = torch.argmax(previous_mask_tensor, dim=1)
                
                stacked_labels = torch.stack([current_labels, previous_labels], dim=1)
                
                final_mask = torch.mode(stacked_labels, dim=1).values #shape: [B, H, W]
                #print(final_mask.shape)
                
        elif args.comb_method == 'weighted_average':    
                #Combined the warped mask with the current one via weighted average, 
                #alpha for current, 1-alpha for average, default is alpha=0.5
                alpha = args.alpha
                combined_mask = alpha * previous_mask_tensor + (1 - alpha) * warped_image #shape: [B, C, H, W]
                final_mask = combined_mask.squeeze() #shape: [C, H, W]
                #print(final_mask.shape)
                                
        else:
                raise ValueError('Invalid method selected')
            
            
        #Back to numpy array:
        final_mask_np = final_mask.cpu().numpy().astype(np.uint8)
        
        final_mask_np = np.transpose(final_mask_np, (1, 2, 0))
        
        #So it's correctly formatted
        final_mask_np = cv2.cvtColor(final_mask_np, cv2.COLOR_RGB2BGR)
        
        out_path = os.path.join(args.output_folder, f"combined_mask.png")
        cv2.imwrite(out_path, final_mask_np)
        print(f"Saved combined mask: {out_path}")
        

def main():
    parser = argparse.ArgumentParser()
    #add also parsser for input images folder
    parser.add_argument('--cfg', required=True, type=str, help='experiment configure file name')
    parser.add_argument('--path', default=None, type=str, help='checkpoint path')
    parser.add_argument('--url', default=None, type=str, help='checkpoint URL')
    parser.add_argument('--input_folder', required=True, type=str, help='segmentation masks folder')
    parser.add_argument('--output_folder', required=True, type=str, help='model output folder')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha parameter for  weighted average')
    parser.add_argument('--device', type=str, default='cpu', help='device to run inference')
    parser.add_argument('--comb_method', type=str, default='weighted_average', help='mode (weighted average or majority voting) for mask combining')
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
    
                    
#To adapt this scirpt with actual segmentation masks input, just compute the flow between
#frames i and i+1 and warp the flow with the mask at frame i and then combine it with 
#the one at frame i+1, in particular in grid_sample use mode the nearest as we are not 
#working with integer values