import os
import cv2
import time
import torch
import argparse
import numpy as np
from PIL import Image
from libs.SPN1 import SPN
import torchvision.utils as vutils
from libs.utils import print_options
from libs.MatrixTest1 import MulLayer
import torch.backends.cudnn as cudnn
from libs.Loader1 import Dataset
from libs.models import encoder3, encoder4
from libs.models import decoder3, decoder4
import torchvision.transforms as transforms
from libs.smooth_filter1 import smooth_filter
import torch.multiprocessing as mp
from multiprocessing import freeze_support

mp.set_start_method("spawn")
freeze_support()

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                    help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                    help='pre-trained decoder path')
parser.add_argument("--matrixPath", default='models/r41.pth',
                    help='pre-trained model path')
parser.add_argument("--stylePath", default="data/photo_real/style/images/",
                        help='path to style image')
parser.add_argument("--styleSegPath", default="data/photo_real/styleSeg/",
                        help='path to style image masks')
parser.add_argument("--contentPath", default="data/photo_real/content/images/",
                        help='path to content image')
parser.add_argument("--contentSegPath", default="data/photo_real/contentSeg/",
                        help='path to content image masks')
parser.add_argument("--outf", default="PhotoReal_orig/",
                        help='path to save output images')
parser.add_argument("--batchSize", type=int, default=1,
                    help='batch size')
parser.add_argument('--fineSize', type=int, default=512,
                    help='image size')
parser.add_argument("--layer", default="r41",
                    help='features of which layer to transform, either r31 or r41')
parser.add_argument("--spn_dir", default='models/r41_spn.pth',
                    help='path to pretrained SPN model')
parser.add_argument("--specific_style", default=None,
                    help='specific style image to use (filename without path)')
parser.add_argument("--specific_content", default=None,
                    help='specific content image to use (filename without path)')

################# PREPARATIONS #################
opt = parser.parse_args()
opt.cuda = torch.cuda.is_available()
print_options(opt)

os.makedirs(opt.outf, exist_ok=True)

cudnn.benchmark = True

################# MODEL #################
if(opt.layer == 'r31'):
    vgg = encoder3()
    dec = decoder3()
elif(opt.layer == 'r41'):
    vgg = encoder4()
    dec = decoder4()
matrix = MulLayer(opt.layer)
vgg.load_state_dict(torch.load(opt.vgg_dir))
dec.load_state_dict(torch.load(opt.decoder_dir))
matrix.load_state_dict(torch.load(opt.matrixPath))
spn = SPN()
spn.load_state_dict(torch.load(opt.spn_dir))

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    spn.cuda()
    matrix.cuda()


################# CUSTOM DATA LOADING #################
def load_image_list(image_dir):
    """Load list of image files from a directory."""
    if not os.path.exists(image_dir):
        print(f"Warning: Directory {image_dir} does not exist.")
        return []
        
    image_list = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_list.append(filename)
    
    return image_list


def process_image_pair(content_path, style_path, content_seg_path=None, style_seg_path=None):
    """Process a single content-style image pair."""
    
    # Load preprocessing function
    preprocess = transforms.Compose([
        transforms.Resize(opt.fineSize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load content image
    try:
        content_img = Image.open(content_path).convert('RGB')
        content_img = preprocess(content_img).unsqueeze(0)
    except Exception as e:
        print(f"Error loading content image {content_path}: {e}")
        return None, None, None, None, None, None
    
    # Load style image
    try:
        style_img = Image.open(style_path).convert('RGB')
        style_img = preprocess(style_img).unsqueeze(0)
    except Exception as e:
        print(f"Error loading style image {style_path}: {e}")
        return None, None, None, None, None, None
    
    # Create a white noise image
    whiten_img = content_img.clone()
    
    # Default masks (all ones)
    cmask = torch.ones(1, 1, opt.fineSize, opt.fineSize)
    smask = torch.ones(1, 1, opt.fineSize, opt.fineSize)
    
    # Load content mask if available
    if content_seg_path and os.path.exists(content_seg_path):
        try:
            cmask_img = Image.open(content_seg_path).convert('L')
            cmask_img = transforms.Resize(opt.fineSize)(cmask_img)
            cmask = transforms.ToTensor()(cmask_img).unsqueeze(0)
            cmask = (cmask > 0.5).float()
        except Exception as e:
            print(f"Error loading content mask {content_seg_path}: {e}")
    
    # Load style mask if available
    if style_seg_path and os.path.exists(style_seg_path):
        try:
            smask_img = Image.open(style_seg_path).convert('L')
            smask_img = transforms.Resize(opt.fineSize)(smask_img)
            smask = transforms.ToTensor()(smask_img).unsqueeze(0)
            smask = (smask > 0.5).float()
        except Exception as e:
            print(f"Error loading style mask {style_seg_path}: {e}")
    
    # Get content file name without extension for saving results
    content_name = os.path.basename(content_path)
    content_name = os.path.splitext(content_name)[0]
    
    # Get style file name for naming
    style_name = os.path.basename(style_path)
    style_name = os.path.splitext(style_name)[0]
    
    # Move to GPU if available
    if opt.cuda:
        content_img = content_img.cuda()
        style_img = style_img.cuda()
        whiten_img = whiten_img.cuda()
        cmask = cmask.cuda()
        smask = smask.cuda()
    
    return content_img, style_img, whiten_img, cmask, smask, f"{content_name}_styled_by_{style_name}"


# Main execution
if __name__ == "__main__":
    # Get list of content and style images
    content_images = load_image_list(opt.contentPath)
    style_images = load_image_list(opt.stylePath)
    
    # Filter for specific images if requested
    if opt.specific_content:
        content_images = [img for img in content_images if opt.specific_content in img]
    
    if opt.specific_style:
        style_images = [img for img in style_images if opt.specific_style in img]
    
    if not content_images:
        print(f"No content images found in {opt.contentPath}")
        exit(1)
        
    if not style_images:
        print(f"No style images found in {opt.stylePath}")
        exit(1)
    
    print(f"Found {len(content_images)} content images and {len(style_images)} style images")
    
    # Process each content image with each style image
    for content_file in content_images:
        content_path = os.path.join(opt.contentPath, content_file)
        content_name = os.path.splitext(content_file)[0]
        
        # Try to find matching content mask
        content_seg_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = os.path.join(opt.contentSegPath, content_name + ext)
            if os.path.exists(potential_path):
                content_seg_path = potential_path
                break
        
        for style_file in style_images:
            style_path = os.path.join(opt.stylePath, style_file)
            style_name = os.path.splitext(style_file)[0]
            
            # Try to find matching style mask
            style_seg_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_path = os.path.join(opt.styleSegPath, style_name + ext)
                if os.path.exists(potential_path):
                    style_seg_path = potential_path
                    break
            
            print(f"\nProcessing content: {content_file} with style: {style_file}")
            
            # Process this image pair
            contentImg, styleImg, whitenImg, cmasks, smasks, output_name = process_image_pair(
                content_path, style_path, content_seg_path, style_seg_path
            )
            
            if contentImg is None:
                print("Skipping this pair due to loading errors")
                continue
            
            # Forward pass through the network
            sF = vgg(styleImg)
            cF = vgg(contentImg)
            
            with torch.no_grad():
                if(opt.layer == 'r41'):
                    feature = matrix(cF[opt.layer], sF[opt.layer], cmasks, smasks)
                else:
                    feature = matrix(cF, sF, cmasks, smasks)
                transfer = dec(feature)
                filtered = spn(transfer, whitenImg)
            
            # Save intermediate result
            vutils.save_image(transfer, os.path.join(opt.outf, f'{output_name}_transfer.png'))
            
            # Post-process and save final results
            filtered = filtered.clamp(0, 1)
            filtered = filtered.cpu()
            vutils.save_image(filtered, f'{opt.outf}/{output_name}_filtered.png')
            
            # Convert to numpy for smooth filter
            out_img = filtered.squeeze(0).mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            content = contentImg.squeeze(0).mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            
            # Apply smooth filtering
            try:
                smoothed = smooth_filter(out_img.copy(), content.copy(), f_radius=15, f_edge=1e-1)
                smoothed.save(f'{opt.outf}/{output_name}_smooth.png')
                print(f'Processed images saved with prefix: {opt.outf}/{output_name}')
            except Exception as e:
                print(f"Smooth filtering failed: {e}")
                # Save fallback result if smoothing fails
                filtered_img = Image.fromarray(out_img)
                filtered_img.save(f'{opt.outf}/{output_name}_final.png')
                print(f'Fallback image saved at: {opt.outf}/{output_name}_final.png')
    
    print("\nAll processing complete!")