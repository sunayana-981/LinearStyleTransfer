import os
import torch
import argparse
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from torchvision import transforms
from PIL import Image
from sklearn.svm import LinearSVC
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from libs.Loader import Dataset
from libs.Matrix import MulLayer
from libs.utils import print_options
from libs.models import encoder3, encoder4, decoder3, decoder4
from tqdm import tqdm
import logging
import sys
import cv2

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])

# Add smoothing functions from smooth_filter1.py
def smooth_filter_enhanced(init_img, content_img, f_radius=15, f_edge=1e-1):
    """
    Enhanced smooth filter with multiple stages to reduce artifacts.
    
    Args:
        init_img: Initial stylized image (PIL Image, numpy array, or path)
        content_img: Content image (PIL Image, numpy array, or path)
        f_radius: Filter radius
        f_edge: Filter edge preservation strength
        
    Returns:
        PIL Image with the smoothed result
    """
    # Load images
    if isinstance(init_img, str):
        init_img = Image.open(init_img).convert("RGB")
    
    if isinstance(content_img, str):
        content_img = Image.open(content_img).convert("RGB")
    
    # Convert to numpy arrays
    if isinstance(init_img, Image.Image):
        stylized_np = np.array(init_img)
    else:
        stylized_np = init_img.copy()  # Assume numpy array
    
    h, w, _ = stylized_np.shape
    
    # Resize content to match stylized
    if isinstance(content_img, Image.Image):
        content_np = np.array(content_img.resize((w, h)))
    elif isinstance(content_img, np.ndarray):
        if content_img.shape[:2] != (h, w):
            content_np = cv2.resize(content_img, (w, h))
        else:
            content_np = content_img.copy()
    
    # Convert to float32 for processing
    stylized_float = stylized_np.astype(np.float32) / 255.0
    content_float = content_np.astype(np.float32) / 255.0
    
    # Convert to BGR for OpenCV processing
    stylized_bgr = cv2.cvtColor(stylized_np, cv2.COLOR_RGB2BGR)
    content_bgr = cv2.cvtColor(content_np, cv2.COLOR_RGB2BGR)
    
    # 1. Apply guided filter first (uses content as guide)
    radius = int(f_radius)
    eps = f_edge * 0.1
    guided_filter_result = cv2.ximgproc.guidedFilter(
        guide=content_bgr,
        src=stylized_bgr,
        radius=radius,
        eps=eps
    )
    
    # 2. Apply domain transform filter (edge-preserving)
    try:
        dt_filter_result = cv2.ximgproc.dtFilter(
            guide=content_bgr,
            src=guided_filter_result,
            sigmaSpatial=f_radius,
            sigmaColor=f_edge,
            mode=cv2.ximgproc.DTF_RF  # Rolling guidance filter mode
        )
    except Exception as e:
        print(f"Domain transform filter failed: {e}")
        # Fall back to recursive edge-preserving filter
        dt_filter_result = cv2.edgePreservingFilter(
            guided_filter_result, 
            flags=cv2.RECURS_FILTER,
            sigma_s=f_radius,
            sigma_r=f_edge
        )
    
    # 3. Apply bilateral filter to reduce remaining noise while preserving edges
    try:
        bilateral_result = cv2.bilateralFilter(
            dt_filter_result,
            d=9,  # Diameter of pixel neighborhood
            sigmaColor=f_edge * 100,
            sigmaSpace=f_radius / 4
        )
    except Exception as e:
        print(f"Bilateral filter failed: {e}")
        bilateral_result = dt_filter_result
    
    # 4. Structure-texture decomposition to preserve important structure
    try:
        # Use relative total variation for structure-texture decomposition
        structure_result = cv2.ximgproc.l0Smooth(
            bilateral_result,
            lambda_=0.02
        )
        
        # Blend structure with filtered result
        alpha = 0.6  # Adjust blend factor
        blended_result = cv2.addWeighted(
            structure_result, alpha,
            bilateral_result, 1 - alpha,
            0
        )
    except Exception as e:
        print(f"Structure-texture decomposition failed: {e}")
        blended_result = bilateral_result
    
    # Convert back to RGB
    filtered_rgb = cv2.cvtColor(blended_result, cv2.COLOR_BGR2RGB)
    
    # Return as PIL image
    return Image.fromarray(filtered_rgb)

def smooth_filter_fallback(init_img, content_img, f_radius=15, f_edge=1e-1):
    """
    Basic fallback smooth filter method.
    """
    # Load images
    if isinstance(init_img, str):
        init_img = Image.open(init_img).convert("RGB")
    
    if isinstance(content_img, str):
        content_img = Image.open(content_img).convert("RGB")
    
    # Convert to numpy arrays
    if isinstance(init_img, Image.Image):
        stylized_np = np.array(init_img)
    else:
        stylized_np = init_img.copy()  # Assume numpy array
    
    h, w, _ = stylized_np.shape
    
    # Resize content to match stylized
    if isinstance(content_img, Image.Image):
        content_np = np.array(content_img.resize((w, h)))
    elif isinstance(content_img, np.ndarray):
        if content_img.shape[:2] != (h, w):
            content_np = cv2.resize(content_img, (w, h))
        else:
            content_np = content_img.copy()
    
    # Convert to BGR for OpenCV
    stylized_bgr = cv2.cvtColor(stylized_np, cv2.COLOR_RGB2BGR)
    content_bgr = cv2.cvtColor(content_np, cv2.COLOR_RGB2BGR)
    
    # Simple bilateral filter
    filtered_bgr = cv2.bilateralFilter(
        stylized_bgr, 
        d=int(f_radius), 
        sigmaColor=f_edge*100, 
        sigmaSpace=f_radius/2
    )
    
    # Convert back to RGB
    filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
    
    # Return as PIL image
    return Image.fromarray(filtered_rgb)

def smooth_filter(init_img, content_img, f_radius=15, f_edge=1e-1):
    """
    Main smooth filter function that tries multiple approaches.
    
    Args:
        init_img: Initial stylized image (PIL Image, numpy array, or path)
        content_img: Content image (PIL Image, numpy array, or path)
        f_radius: Filter radius
        f_edge: Filter edge preservation strength
        
    Returns:
        PIL Image with the smoothed result
    """
    try:
        # Try the enhanced filter first
        print("Applying enhanced filtering...")
        return smooth_filter_enhanced(init_img, content_img, f_radius, f_edge)
    except Exception as e:
        print(f"Enhanced filtering failed: {e}")
        print("Falling back to basic filtering...")
        try:
            # Fall back to simple method
            return smooth_filter_fallback(init_img, content_img, f_radius, f_edge)
        except Exception as e2:
            print(f"Fallback filtering also failed: {e2}")
            # If all else fails, return original image
            if isinstance(init_img, str):
                return Image.open(init_img).convert("RGB")
            elif isinstance(init_img, np.ndarray):
                return Image.fromarray(init_img)
            else:
                return init_img

# Helper function to convert tensor to PIL Image
def tensor_to_pil(tensor):
    """Convert a torch tensor to PIL Image"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.cpu().detach()
    tensor = tensor.mul(255).clamp(0, 255).byte()
    tensor = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(tensor)

# Helper function to convert PIL to tensor
def pil_to_tensor(pil_image, device='cuda'):
    """Convert a PIL Image to torch tensor"""
    transform = transforms.ToTensor()
    tensor = transform(pil_image).unsqueeze(0)
    if device == 'cuda' and torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

class StyleClassController:
    def __init__(self, vgg, matrix, decoder, layer_name='r41', device='cuda'):
        self.vgg = vgg
        self.matrix = matrix
        self.decoder = decoder
        self.layer_name = layer_name
        self.device = device

    def get_style_transfer_features(self, content_img, style_img):
        """Extract features from style transfer at specified layer"""
        device = next(self.decoder.parameters()).device
        dtype = next(self.decoder.parameters()).dtype
        
        # Ensure inputs are on correct device and dtype
        content_img = content_img.to(device=device, dtype=dtype)
        style_img = style_img.to(device=device, dtype=dtype)

        with torch.no_grad():
            content_features = self.vgg(content_img)[self.layer_name]
            style_features = self.vgg(style_img)[self.layer_name]
            transformed_features, _ = self.matrix(content_features, style_features)
            return transformed_features
    
    def learn_class_direction(self, content_img, target_styles, other_styles, batch_size=10):
        """Learn direction between two style classes"""
        logging.info(f"Learning class direction between {len(target_styles)} target and {len(other_styles)} other styles")
        
        # Process target styles in batches
        target_features = []
        for i in tqdm(range(0, len(target_styles), batch_size), desc="Processing target styles"):
            batch = target_styles[i:i + batch_size]
            batch_features = [self.get_style_transfer_features(content_img, style) for style in batch]
            batch_features = [f.view(f.size(0), -1).cpu().numpy() for f in batch_features]
            target_features.extend(batch_features)
        target_features = np.concatenate(target_features, axis=0)
        
        # Process other styles in batches
        other_features = []
        for i in tqdm(range(0, len(other_styles), batch_size), desc="Processing other styles"):
            batch = other_styles[i:i + batch_size]
            batch_features = [self.get_style_transfer_features(content_img, style) for style in batch]
            batch_features = [f.view(f.size(0), -1).cpu().numpy() for f in batch_features]
            other_features.extend(batch_features)
        other_features = np.concatenate(other_features, axis=0)

        # Train SVM to find direction
        X = np.concatenate([target_features, other_features])
        y = np.concatenate([np.ones(len(target_features)), np.zeros(len(other_features))])
        
        svm = LinearSVC(C=0.01, dual=False, random_state=42)
        svm.fit(X, y)
        
        # Convert to tensor with correct shape
        feature_shape = self.get_style_transfer_features(content_img, target_styles[0]).shape[1:]
        direction = torch.tensor(svm.coef_[0], device=self.device).reshape(feature_shape)
        return direction

    def apply_class_direction(self, content_img, style_img, direction, strength=1.0, apply_smoothing=False, 
                              smoothing_radius=15, smoothing_edge=1e-1):
        """Apply learned class direction during style transfer with optional smoothing"""
        # Get device and dtype from decoder
        device = next(self.decoder.parameters()).device
        dtype = next(self.decoder.parameters()).dtype
        
        # Move all inputs to correct device and dtype
        content_img = content_img.to(device=device, dtype=dtype)
        style_img = style_img.to(device=device, dtype=dtype)
        direction = direction.to(device=device, dtype=dtype)
        
        with torch.no_grad():
            # Get style transfer features
            content_features = self.vgg(content_img)[self.layer_name]
            style_features = self.vgg(style_img)[self.layer_name]
            transformed_features, matrix = self.matrix(content_features, style_features)
            
            # Normalize direction application
            feature_norm = torch.norm(transformed_features)
            direction_norm = torch.norm(direction)
            scale_factor = feature_norm / direction_norm
            adaptive_strength = strength * scale_factor * 0.1
            
            # Apply direction
            modified_features = transformed_features + (adaptive_strength * direction.unsqueeze(0))
            result = self.decoder(modified_features)
            
            # Apply smoothing if requested
            if apply_smoothing:
                # Convert to PIL for smoothing
                result_pil = tensor_to_pil(result)
                content_pil = tensor_to_pil(content_img)
                
                # Apply smoothing filter
                smoothed_result = smooth_filter(result_pil, content_pil, 
                                               f_radius=smoothing_radius, 
                                               f_edge=smoothing_edge)
                
                # Convert back to tensor
                result = pil_to_tensor(smoothed_result, device=device)
            
            return result.clamp(0, 1), matrix

class StyleVisualizer:
    def __init__(self, num_strengths):
        self.fig_size = (15, 8)
        self.num_strengths = num_strengths
    
    def create_comparison_plot(self, transfers, strengths, content_img, style_img, 
                             save_dir, content_name, style_class_name):
        # Create figure
        fig = plt.figure(figsize=self.fig_size, constrained_layout=True)
        gs = GridSpec(2, self.num_strengths + 2, figure=fig)
        
        def tensor_to_image(tensor):
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            return tensor.cpu().permute(1, 2, 0).numpy()
        
        # Plot content and style images
        ax_content = fig.add_subplot(gs[0, 0])
        ax_content.imshow(tensor_to_image(content_img))
        ax_content.set_title('Content Image')
        ax_content.axis('off')
        
        ax_style = fig.add_subplot(gs[1, 0])
        ax_style.imshow(tensor_to_image(style_img))
        ax_style.set_title('Style Image')
        ax_style.axis('off')
        
        # Plot variations
        for idx, (transfer, strength) in enumerate(zip(transfers, strengths)):
            ax = fig.add_subplot(gs[:, idx + 1])
            ax.imshow(tensor_to_image(transfer))
            ax.set_title(f'Class Strength: {strength:.1f}')
            ax.axis('off')
        
        # Add strength colorbar
        norm = plt.Normalize(min(strengths), max(strengths))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=norm)
        cbar_ax = fig.add_subplot(gs[:, -1])
        plt.colorbar(sm, cax=cbar_ax, label='Class Direction Strength')
        
        # Save plot
        plot_filename = f'class_comparison_{content_name}_{style_class_name}.png'
        plot_path = os.path.join(save_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

def smooth_style_transfer(content_img, style_img, transfer_img, radius=15, edge=1e-1):
    """Standalone function to smooth an existing style transfer result"""
    # Convert transfer tensor to PIL Image
    transfer_pil = tensor_to_pil(transfer_img)
    content_pil = tensor_to_pil(content_img)
    
    # Apply smoothing
    smoothed = smooth_filter(transfer_pil, content_pil, f_radius=radius, f_edge=edge)
    
    # Convert back to tensor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smoothed_tensor = pil_to_tensor(smoothed, device=device)
    
    return smoothed_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_image", default="sampled_images/in00.png",
                      help='path to base content image')
    parser.add_argument("--target_style_class", default="sampled_images/day/",
                      help='directory containing target style class images')
    parser.add_argument("--other_style_class", default="sampled_images/night/",
                      help='directory containing other style class images')
    parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                      help='pre-trained encoder path')
    parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                      help='pre-trained decoder path')
    parser.add_argument("--matrixPath", default='models/r41.pth',
                      help='pre-trained matrix path')
    parser.add_argument("--outf", default="results/style_class_cav2/",
                      help='path to output images')
    parser.add_argument("--batch_size", type=int, default=10,
                      help='batch size for processing images')
    parser.add_argument("--num_images", type=int, default=100,
                      help='number of images to use from each class')
    parser.add_argument("--strengths", type=float, nargs='+',
                      default=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--apply_smoothing", action='store_true',
                      help='apply smoothing to the results')
    parser.add_argument("--smoothing_radius", type=float, default=15.0,
                      help='radius for smoothing filter')
    parser.add_argument("--smoothing_edge", type=float, default=1e-1,
                      help='edge preservation strength for smoothing')
    
    opt = parser.parse_args()
    
    # Create output directories
    os.makedirs(opt.outf, exist_ok=True)
    comparison_dir = os.path.join(opt.outf, 'comparisons')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Initialize models
    vgg = encoder4()
    dec = decoder4()
    matrix = MulLayer('r41')
    
    # Load weights
    vgg.load_state_dict(torch.load(opt.vgg_dir, map_location='cpu'))
    dec.load_state_dict(torch.load(opt.decoder_dir, map_location='cpu'))
    matrix.load_state_dict(torch.load(opt.matrixPath, map_location='cpu'))
    
    # Move to GPU if available
    if torch.cuda.is_available():
        vgg.cuda()
        dec.cuda()
        matrix.cuda()
    
    # Load images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    
    # Load content image
    content_img = transform(Image.open(opt.content_image).convert('RGB')).unsqueeze(0)
    if torch.cuda.is_available():
        content_img = content_img.cuda()
    
    # Load style images
    def load_style_images(style_dir, max_images):
        dataset = Dataset(style_dir, 256, 256)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        images = []
        for i, (img, _) in enumerate(loader):
            if i >= max_images:
                break
            if torch.cuda.is_available():
                img = img.cuda()
            images.append(img)
        return images
    
    # Load style class images
    target_styles = load_style_images(opt.target_style_class, opt.num_images)
    other_styles = load_style_images(opt.other_style_class, opt.num_images)
    
    logging.info(f"Loaded {len(target_styles)} target style images and {len(other_styles)} other style images")
    
    # Initialize controllers
    style_controller = StyleClassController(vgg, matrix, dec)
    visualizer = StyleVisualizer(len(opt.strengths))
    
    # Learn class direction
    class_direction = style_controller.learn_class_direction(
        content_img,
        target_styles,
        other_styles,
        batch_size=opt.batch_size
    )
    
    # Process each target style
    for style_idx, base_style in enumerate(tqdm(target_styles, desc="Processing styles")):
        transfers = []
        
        # Generate variations
        for strength in opt.strengths:
            # Apply class direction with optional smoothing
            transfer, _ = style_controller.apply_class_direction(
                content_img,
                base_style,
                class_direction,
                strength,
                apply_smoothing=opt.apply_smoothing,
                smoothing_radius=opt.smoothing_radius,
                smoothing_edge=opt.smoothing_edge
            )
            transfers.append(transfer)
            
            # Save individual result
            smoothing_suffix = "_smoothed" if opt.apply_smoothing else ""
            output_filename = f'content_base_style_{style_idx}_strength_{strength:.1f}{smoothing_suffix}.png'
            output_path = os.path.join(opt.outf, output_filename)
            vutils.save_image(transfer, output_path)
        
        # Create comparison plot
        target_class_name = os.path.basename(opt.target_style_class.rstrip('/'))
        smoothing_suffix = "_smoothed" if opt.apply_smoothing else ""
        visualizer.create_comparison_plot(
            transfers,
            opt.strengths,
            content_img,
            base_style,
            comparison_dir,
            'base',
            f'{target_class_name}_{style_idx}{smoothing_suffix}'
        )

if __name__ == "__main__":
    main()