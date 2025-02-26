import os
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
from libs.Matrix import MulLayer
from libs.models import encoder4, decoder4

def load_image(image_path):
    """
    Load and preprocess a single image from path.
    Returns a normalized tensor ready for the model.
    """
    # Load image using PIL
    image = Image.open(image_path).convert('RGB')
    
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    
    # Transform and add batch dimension
    return transform(image).unsqueeze(0)

def style_transfer_per_blur(content_dir, blurred_styles_dir, output_dir, vgg_path, decoder_path, matrix_path):
    """
    Perform style transfer and organize outputs by style image.
    Each style will have its own directory containing all content images processed with that style
    at different blur levels.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models and move to GPU
    vgg = encoder4()
    decoder = decoder4()
    matrix = MulLayer('r41')
    
    vgg.load_state_dict(torch.load(vgg_path))
    decoder.load_state_dict(torch.load(decoder_path))
    matrix.load_state_dict(torch.load(matrix_path))
    
    vgg.cuda().eval()
    decoder.cuda().eval()
    matrix.cuda().eval()
    
    # Get content images list
    content_images = [f for f in os.listdir(content_dir) if f.endswith(('.jpg', '.png'))]
    
    # Get all style images from the first blur level to use as reference
    first_sigma = sorted([d for d in os.listdir(blurred_styles_dir) if os.path.isdir(os.path.join(blurred_styles_dir, d))])[0]
    style_images = [f for f in os.listdir(os.path.join(blurred_styles_dir, first_sigma)) if f.endswith(('.jpg', '.png'))]
    
    # Process each style image
    for style_name in tqdm(style_images, desc="Processing styles"):
        # Create directory for this style
        style_base = os.path.splitext(style_name)[0]
        style_output_dir = os.path.join(output_dir, f"style_{style_base}")
        os.makedirs(style_output_dir, exist_ok=True)
        
        # Process each blur level for this style
        sigma_folders = sorted([d for d in os.listdir(blurred_styles_dir) if os.path.isdir(os.path.join(blurred_styles_dir, d))])
        
        for sigma in tqdm(sigma_folders, desc=f"Processing blur levels for style {style_base}", leave=False):
            # Load style image for this blur level
            style_path = os.path.join(blurred_styles_dir, sigma, style_name)
            style = load_image(style_path).cuda()
            
            # Process each content image with this style at this blur level
            for content_name in tqdm(content_images, desc=f"Processing contents for {sigma}", leave=False):
                # Load content image
                content_path = os.path.join(content_dir, content_name)
                content = load_image(content_path).cuda()
                
                # Get features and perform style transfer
                with torch.no_grad():
                    cF = vgg(content)
                    sF = vgg(style)
                    feature, _ = matrix(cF['r41'], sF['r41'])
                    transfer = decoder(feature)
                
                transfer = transfer.clamp(0, 1)
                
                # Create output filename
                content_base = os.path.splitext(content_name)[0]
                output_filename = f"{content_base}_{sigma}.png"
                output_path = os.path.join(style_output_dir, output_filename)
                
                # Save the styled image
                vutils.save_image(transfer, output_path, normalize=True, scale_each=True)

def main():
    parser = argparse.ArgumentParser(description="Neural Style Transfer with Style-based Organization")
    
    parser.add_argument("--contentPath", default="data/content/",
                        help="Directory containing content images")
    parser.add_argument("--blurredStylesPath", default="data/blurred_styles/",
                        help="Directory containing blur-level subdirectories of style images")
    parser.add_argument("--outputPath", default="data/styled_outputs/",
                        help="Directory to save style transfer outputs")
    parser.add_argument("--vggPath", default="models/vgg_r41.pth",
                        help="Path to pre-trained VGG model")
    parser.add_argument("--decoderPath", default="models/dec_r41.pth",
                        help="Path to pre-trained decoder model")
    parser.add_argument("--matrixPath", default="models/r41.pth",
                        help="Path to pre-trained matrix")
    
    opt = parser.parse_args()
    
    style_transfer_per_blur(
        opt.contentPath,
        opt.blurredStylesPath,
        opt.outputPath,
        opt.vggPath,
        opt.decoderPath,
        opt.matrixPath
    )

if __name__ == "__main__":
    main()