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
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def style_transfer_per_sharp(content_dir, sharpened_styles_dir, output_dir, vgg_path, decoder_path, matrix_path):
    """
    Perform style transfer and organize outputs by style image,
    working with the sharpening-based directory structure.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models and move to GPU
    vgg = encoder4()
    decoder = decoder4()
    matrix = MulLayer('r41')
    
    # Load model weights with weights_only=True for security
    vgg.load_state_dict(torch.load(vgg_path, weights_only=True))
    decoder.load_state_dict(torch.load(decoder_path, weights_only=True))
    matrix.load_state_dict(torch.load(matrix_path, weights_only=True))
    
    vgg.cuda().eval()
    decoder.cuda().eval()
    matrix.cuda().eval()
    
    # Get content images list
    content_images = [f for f in os.listdir(content_dir) if f.endswith(('.jpg', '.png'))]
    
    # Get all sharpening directories
    sharp_dirs = sorted([d for d in os.listdir(sharpened_styles_dir) 
                        if os.path.isdir(os.path.join(sharpened_styles_dir, d))],
                       key=lambda x: float(x.split('_')[1]))  # Sort by sharpening value
    
    if not sharp_dirs:
        raise ValueError(f"No sharpening directories found in {sharpened_styles_dir}")
    
    # Get style images from first sharpening directory
    first_sharp_dir = os.path.join(sharpened_styles_dir, sharp_dirs[0])
    style_images = [f for f in os.listdir(first_sharp_dir) if f.endswith(('.jpg', '.png'))]
    
    # Process each style image
    for style_name in tqdm(style_images, desc="Processing styles"):
        # Extract style number from filename (assumes format "XX_sharp_Y.Y.png")
        style_num = style_name.split('_')[0]
        
        # Create directory for this style
        style_output_dir = os.path.join(output_dir, f"style_{style_num}")
        os.makedirs(style_output_dir, exist_ok=True)
        
        # Process each sharpening level for this style
        for sharp_dir in tqdm(sharp_dirs, desc=f"Processing sharp levels for style {style_num}", leave=False):
            # Extract sharpening value for directory naming
            sharp_value = sharp_dir.split('_')[1]
            sharp_subdir = os.path.join(style_output_dir, f"sharp_{sharp_value}")
            os.makedirs(sharp_subdir, exist_ok=True)
            
            # Construct correct style image path
            style_filename = f"{style_num}_sharp_{sharp_value}.png"
            style_path = os.path.join(sharpened_styles_dir, sharp_dir, style_filename)
            
            if not os.path.exists(style_path):
                print(f"Warning: Style file not found: {style_path}")
                continue
                
            try:
                # Load style image
                style = load_image(style_path).cuda()
                
                # Process each content image with this style at this sharpening level
                for content_name in tqdm(content_images, desc=f"Processing contents for sharp_{sharp_value}", leave=False):
                    # Load content image
                    content_path = os.path.join(content_dir, content_name)
                    content = load_image(content_path).cuda()
                    
                    # Perform style transfer
                    with torch.no_grad():
                        cF = vgg(content)
                        sF = vgg(style)
                        feature, _ = matrix(cF['r41'], sF['r41'])
                        transfer = decoder(feature)
                    
                    # Save the styled image
                    content_base = os.path.splitext(content_name)[0]
                    output_filename = f"{content_base}_sharp_{sharp_value}.png"
                    output_path = os.path.join(sharp_subdir, output_filename)
                    
                    vutils.save_image(
                        transfer.clamp(0, 1),
                        output_path,
                        normalize=True,
                        scale_each=True
                    )
                    
            except Exception as e:
                print(f"Error processing {style_path}: {str(e)}")
                continue

def main():
    parser = argparse.ArgumentParser(description="Neural Style Transfer with Sharpening-based Organization")
    
    parser.add_argument("--contentPath", default="data/content/",
                        help="Directory containing content images")
    parser.add_argument("--sharpenedStylesPath", default="data/sharpened_styles/",
                        help="Directory containing sharpening-level subdirectories of style images")
    parser.add_argument("--outputPath", default="data/sharpened_styled_outputs/",
                        help="Directory to save style transfer outputs")
    parser.add_argument("--vggPath", default="models/vgg_r41.pth",
                        help="Path to pre-trained VGG model")
    parser.add_argument("--decoderPath", default="models/dec_r41.pth",
                        help="Path to pre-trained decoder model")
    parser.add_argument("--matrixPath", default="models/r41.pth",
                        help="Path to pre-trained matrix")
    
    opt = parser.parse_args()
    
    style_transfer_per_sharp(
        opt.contentPath,
        opt.sharpenedStylesPath,
        opt.outputPath,
        opt.vggPath,
        opt.decoderPath,
        opt.matrixPath
    )

if __name__ == "__main__":
    main()