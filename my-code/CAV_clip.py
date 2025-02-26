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
import logging
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from matplotlib.gridspec import GridSpec

# Import required modules
from libs.Loader import Dataset
from libs.Matrix import MulLayer
from libs.utils import print_options
from libs.models import encoder3, encoder4, decoder3, decoder4

class StyleCAVController:
    """Controls the learning and application of CAVs from style transfer variations."""
    def __init__(self, vgg, matrix, decoder, layer_name='r41', device='cuda'):
        self.vgg = vgg
        self.matrix = matrix
        self.decoder = decoder
        self.layer_name = layer_name
        self.device = device
        self.activations = []
        self._register_hooks()
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _register_hooks(self):
        """Registers hooks to capture activations after the matrix layer."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.activations.append(output[0].detach())
            else:
                self.activations.append(output.detach())
        self.matrix.register_forward_hook(hook_fn)

    def get_style_features(self, image_path):
        """Extracts transformed style features from a given image."""
        self.activations = []
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor()
            ])
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.vgg(image_tensor)
                if isinstance(features, dict):
                    features = features[self.layer_name]
                transformed_features, _ = self.matrix(features, features)
            
            if not self.activations:
                raise ValueError("No activations captured during forward pass")
            
            return self.activations[-1]
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            raise

    def collect_training_images(self, style_dir):
        """Collects paths to training images from variant directories."""
        style_path = Path(style_dir)
        self.logger.info(f"Looking for variant directories in: {style_dir}")
        
        # Find and sort variant directories
        variant_dirs = []
        for item in style_path.glob("variant_*"):
            if item.is_dir():
                variant_value = float(item.name.split('_')[1])
                variant_dirs.append((variant_value, item))
        
        if not variant_dirs:
            raise ValueError(f"No variant directories found in {style_dir}")
        
        variant_dirs.sort(key=lambda x: x[0])
        
        # Collect image paths
        image_paths = []
        for _, variant_dir in variant_dirs:
            png_files = list(variant_dir.glob("*.png"))
            if png_files:
                image_paths.append(png_files[0])
                self.logger.info(f"Found image: {png_files[0]}")
        
        if not image_paths:
            raise ValueError("No valid image files found")
        
        return str(image_paths[0]), [str(p) for p in image_paths[1:]]

    def learn_concept_cav(self, style_dir):
        """Learns CAV from existing style transfer variants."""
        self.logger.info(f"Learning CAV from directory: {style_dir}")
        
        try:
            # Get original and variant images
            original_path, variant_paths = self.collect_training_images(style_dir)
            
            # Get features for original style
            original_features = self.get_style_features(original_path)
            original_flat = original_features.view(original_features.size(0), -1).cpu().numpy()
            
            # Get features for variants
            variant_features = []
            for variant_path in variant_paths:
                features = self.get_style_features(variant_path)
                variant_features.append(features.view(features.size(0), -1).cpu().numpy())
            
            variant_flat = np.concatenate(variant_features, axis=0)
            
            # Prepare training data
            features = np.concatenate([original_flat, variant_flat])
            labels = np.concatenate([
                np.zeros(len(original_flat)),
                np.ones(len(variant_flat))
            ])
            
            # Train SVM to find concept direction
            svm = LinearSVC(C=1.0, dual="auto", max_iter=1000)
            svm.fit(features, labels)
            
            # Get learned CAV
            cav = torch.tensor(svm.coef_[0], dtype=torch.float32).to(self.device)
            
            # Reshape to match feature dimensions
            cav = cav.view(original_features.shape[1:])
            
            self.logger.info("Successfully learned CAV")
            return cav
            
        except Exception as e:
            self.logger.error(f"Error learning CAV: {str(e)}")
            raise

    def apply_cav(self, content_image, style_image, cav, strength=1.0):
        """Applies learned CAV during style transfer."""
        try:
            with torch.no_grad():
                # Extract and transform features
                content_features = self.vgg(content_image)[self.layer_name]
                style_features = self.vgg(style_image)[self.layer_name]
                transformed_features, matrix = self.matrix(content_features, style_features)
                
                # Log feature statistics
                self.logger.debug(f"Feature stats - Range: [{transformed_features.min():.3f}, {transformed_features.max():.3f}]")
                self.logger.debug(f"Mean: {transformed_features.mean():.3f}")
                
                # Apply CAV with adaptive scaling
                cav = cav.to(transformed_features.device)
                scale_factor = torch.norm(transformed_features) / (torch.norm(cav) + 1e-8)
                modified_features = transformed_features + (strength * scale_factor * cav)
                
                # Generate output
                result = self.decoder(modified_features)
                return result.clamp(0, 1), matrix
                
        except Exception as e:
            self.logger.error(f"Error applying CAV: {str(e)}")
            raise

class CAVVisualizer:
    """Creates comparison visualizations for different CAV strengths."""
    def __init__(self, num_strengths):
        self.fig_size = (15, 8)
        self.num_strengths = num_strengths
    
    def create_comparison_plot(self, transfers, strengths, content_img, style_img, 
                             save_dir, content_name, style_num):
        fig = plt.figure(figsize=self.fig_size, constrained_layout=True)
        gs = GridSpec(2, self.num_strengths + 2, figure=fig)
        
        def tensor_to_image(tensor):
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            return tensor.cpu().permute(1, 2, 0).numpy()
        
        # Plot original images
        ax_content = fig.add_subplot(gs[0, 0])
        ax_content.imshow(tensor_to_image(content_img))
        ax_content.set_title('Content')
        ax_content.axis('off')
        
        ax_style = fig.add_subplot(gs[1, 0])
        ax_style.imshow(tensor_to_image(style_img))
        ax_style.set_title('Style')
        ax_style.axis('off')
        
        # Plot variations
        for idx, (transfer, strength) in enumerate(zip(transfers, strengths)):
            ax = fig.add_subplot(gs[:, idx + 1])
            ax.imshow(tensor_to_image(transfer))
            ax.set_title(f'Strength: {strength:.1f}')
            ax.axis('off')
        
        # Add strength scale
        norm = plt.Normalize(min(strengths), max(strengths))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=norm)
        cbar_ax = fig.add_subplot(gs[:, -1])
        plt.colorbar(sm, cax=cbar_ax, label='CAV Strength')
        
        # Save plot
        plot_filename = f'comparison_content{content_name}_style_{style_num:02d}.png'
        plot_path = os.path.join(save_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Style Transfer CAV Training')
    parser.add_argument("--data_dir", default="data",
                      help='base directory containing styled_outputs')
    parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                      help='pre-trained encoder path')
    parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                      help='pre-trained decoder path')
    parser.add_argument("--matrix_path", default='models/r41.pth',
                      help='pre-trained model path')
    parser.add_argument("--style_dir", default="data/style/",
                      help='path to style images')
    parser.add_argument("--content_dir", default="data/content/",
                      help='path to content images')
    parser.add_argument("--outf", default="output/cav_results1/",
                      help='path to output images')
    parser.add_argument("--matrix_outf", default="Matrices/",
                      help='path to save transformation matrices')
    parser.add_argument("--load_size", type=int, default=256,
                      help='scale image size')
    parser.add_argument("--fine_size", type=int, default=256,
                      help='crop image size')
    parser.add_argument("--layer", default="r41",
                      help='which features to transfer')
    parser.add_argument("--cav_strengths", type=float, nargs='+',
                      default=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
                      help='CAV strength values to test')
    
    opt = parser.parse_args()
    opt.cuda = torch.cuda.is_available()
    print_options(opt)
    
    # Create output directories
    os.makedirs(opt.outf, exist_ok=True)
    os.makedirs(opt.matrix_outf, exist_ok=True)
    
    # Initialize models
    vgg = encoder4() if opt.layer == 'r41' else encoder3()
    dec = decoder4() if opt.layer == 'r41' else decoder3()
    matrix = MulLayer(opt.layer)
    
    vgg.load_state_dict(torch.load(opt.vgg_dir))
    dec.load_state_dict(torch.load(opt.decoder_dir))
    matrix.load_state_dict(torch.load(opt.matrix_path))
    
    if opt.cuda:
        vgg.cuda()
        dec.cuda()
        matrix.cuda()
    
    # Initialize CAV controller and data loaders
    cav_controller = StyleCAVController(vgg, matrix, dec, opt.layer)
    
    content_dataset = Dataset(opt.content_dir, opt.load_size, opt.fine_size)
    content_loader = torch.utils.data.DataLoader(
        dataset=content_dataset,
        batch_size=1,
        shuffle=False
    )
    
    style_dataset = Dataset(opt.style_dir, opt.load_size, opt.fine_size)
    style_loader = torch.utils.data.DataLoader(
        dataset=style_dataset,
        batch_size=1,
        shuffle=False
    )
    
    # Setup visualization
    comparison_dir = os.path.join(opt.outf, 'comparisons')
    os.makedirs(comparison_dir, exist_ok=True)
    visualizer = CAVVisualizer(len(opt.cav_strengths))
    
    # Process each content-style pair
    for content_idx, (content, content_name) in enumerate(content_loader):
        content = content.cuda() if opt.cuda else content
        
        for style_idx, (style, style_name) in enumerate(style_loader, 1):
            style = style.cuda() if opt.cuda else style
            
            try:
                # Get style directory path
                style_dir = os.path.join(opt.data_dir, 'styled_outputs', f'style_{style_idx:02d}')
                
                # Learn CAV from style variants
                cav = cav_controller.learn_concept_cav(style_dir)
                
                # Apply CAV with different strengths
                transfers = []
                for strength in opt.cav_strengths:
                    # Apply CAV modification
                    transfer, matrix = cav_controller.apply_cav(content, style, cav, strength)
                    transfers.append(transfer)
                    
                    # Save output image
                    output_filename = f'cav_content{content_idx:02d}_style_{style_idx:02d}_strength_{strength:.1f}.png'
                    output_path = os.path.join(opt.outf, output_filename)
                    vutils.save_image(transfer, output_path)
                    
                    # Save transformation matrix
                    matrix_filename = f'cav_content{content_idx:02d}_style_{style_idx:02d}_strength_{strength:.1f}_matrix.pth'
                    matrix_path = os.path.join(opt.matrix_outf, matrix_filename)
                    torch.save(matrix, matrix_path)
                
                # Create comparison visualization
                visualizer.create_comparison_plot(
                    transfers,
                    opt.cav_strengths,
                    content,
                    style,
                    comparison_dir,
                    content_idx,
                    style_idx
                )
                
            except Exception as e:
                logging.error(f"Error processing content {content_idx} with style {style_idx}: {str(e)}")
                continue

if __name__ == "__main__":
    main()