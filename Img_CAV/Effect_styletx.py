
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

def style_transfer_per_effect(content_dir, processed_styles_dir, output_dir, vgg_path, decoder_path, matrix_path):
    """
    Perform style transfer and organize outputs by effect first, then style.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models and move to GPU
    vgg = encoder4()
    decoder = decoder4()
    matrix = MulLayer('r41')
    
    # Load model weights with weights_only=True
    vgg.load_state_dict(torch.load(vgg_path, weights_only=True))
    decoder.load_state_dict(torch.load(decoder_path, weights_only=True))
    matrix.load_state_dict(torch.load(matrix_path, weights_only=True))
    
    vgg.cuda().eval()
    decoder.cuda().eval()
    matrix.cuda().eval()
    
    # Get content images list
    content_images = [f for f in os.listdir(content_dir) if f.endswith(('.jpg', '.png'))]
    
    # Get all effect directories
    effect_dirs = [d for d in os.listdir(processed_styles_dir) 
                  if os.path.isdir(os.path.join(processed_styles_dir, d)) and d.endswith('_styles')]
    
    if not effect_dirs:
        raise ValueError(f"No effect directories found in {processed_styles_dir}")
    
    # Process each effect type
    for effect_dir_name in tqdm(effect_dirs, desc="Processing effects"):
        effect_name = effect_dir_name.replace('_styles', '')
        effect_base_dir = os.path.join(processed_styles_dir, effect_dir_name)
        
        # Create main effect directory in output
        effect_output_dir = os.path.join(output_dir, effect_name)
        os.makedirs(effect_output_dir, exist_ok=True)
        
        # Get all intensity levels for this effect
        intensity_dirs = sorted([d for d in os.listdir(effect_base_dir) 
                               if os.path.isdir(os.path.join(effect_base_dir, d))],
                              key=lambda x: float(x.split('_')[1]))
        
        # Get style images from first intensity directory
        first_intensity_dir = os.path.join(effect_base_dir, intensity_dirs[0])
        style_images = [f for f in os.listdir(first_intensity_dir) if f.endswith(('.jpg', '.png'))]
        
        # Process each intensity level
        for intensity_dir in tqdm(intensity_dirs, desc=f"Processing {effect_name} levels", leave=False):
            intensity_value = intensity_dir.split('_')[1]
            
            # Create intensity level directory
            intensity_output_dir = os.path.join(effect_output_dir, f"{effect_name}_{intensity_value}")
            os.makedirs(intensity_output_dir, exist_ok=True)
            
            # Process each style at this intensity
            for style_name in tqdm(style_images, desc=f"Processing styles for {effect_name}_{intensity_value}", leave=False):
                style_num = style_name.split('_')[0]
                
                # Create style directory within intensity level
                style_output_dir = os.path.join(intensity_output_dir, f"style_{style_num}")
                os.makedirs(style_output_dir, exist_ok=True)
                
                # Construct style image path
                style_filename = f"{style_num}_{effect_name}_{intensity_value}.png"
                style_path = os.path.join(effect_base_dir, intensity_dir, style_filename)
                
                if not os.path.exists(style_path):
                    print(f"Warning: Style file not found: {style_path}")
                    continue
                
                try:
                    # Load style image
                    style = load_image(style_path).cuda()
                    
                    # Process each content image with this style
                    for content_name in tqdm(content_images, desc=f"Processing contents", leave=False):
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
                        output_filename = f"{content_base}_style_{style_num}.png"
                        output_path = os.path.join(style_output_dir, output_filename)
                        
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
    parser = argparse.ArgumentParser(description="Neural Style Transfer with Effect-First Organization")
    
    parser.add_argument("--contentPath", default="data/content/",
                        help="Directory containing content images")
    parser.add_argument("--processedStylesPath", default="data/processed_styles/",
                        help="Directory containing effect-based subdirectories of style images")
    parser.add_argument("--outputPath", default="data/effect_styled_outputs/",
                        help="Directory to save style transfer outputs")
    parser.add_argument("--vggPath", default="models/vgg_r41.pth",
                        help="Path to pre-trained VGG model")
    parser.add_argument("--decoderPath", default="models/dec_r41.pth",
                        help="Path to pre-trained decoder model")
    parser.add_argument("--matrixPath", default="models/r41.pth",
                        help="Path to pre-trained matrix")
    
    opt = parser.parse_args()
    
    style_transfer_per_effect(
        opt.contentPath,
        opt.processedStylesPath,
        opt.outputPath,
        opt.vggPath,
        opt.decoderPath,
        opt.matrixPath
    )

if __name__ == "__main__":
    main()
