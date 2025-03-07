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

# Import existing style transfer components
from libs.Loader import Dataset
from libs.Matrix import MulLayer
from libs.utils import print_options
from libs.models import encoder3, encoder4, decoder3, decoder4

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec

class CAVVisualizer:
    """
    Creates and saves comparison plots of different CAV strength applications for style transfer.
    Each plot is saved with a unique filename based on the content and style identifiers.
    """
    def __init__(self, num_strengths):
        self.fig_size = (15, 8)
        self.num_strengths = num_strengths
    
    def create_comparison_plot(self, transfers, strengths, content_img, style_img, 
                             save_dir, content_name, style_num):
        """
        Creates and saves a comprehensive comparison plot with a unique filename.
        
        Args:
            transfers: List of tensors containing stylized images with different CAV strengths
            strengths: List of CAV strength values used
            content_img: Original content image tensor
            style_img: Original style image tensor
            save_dir: Directory to save the comparison plot
            content_name: Name or identifier of the content image
            style_num: Style number or identifier
        """
        # Create figure with gridspec for flexible layout
        fig = plt.figure(figsize=self.fig_size, constrained_layout=True)
        gs = GridSpec(2, self.num_strengths + 2, figure=fig)
        
        # Helper function to convert tensor to displayable image
        def tensor_to_image(tensor):
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            return tensor.cpu().permute(1, 2, 0).numpy()
        
        # Plot original content and style images
        ax_content = fig.add_subplot(gs[0, 0])
        ax_content.imshow(tensor_to_image(content_img))
        ax_content.set_title('Content Image')
        ax_content.axis('off')
        
        ax_style = fig.add_subplot(gs[1, 0])
        ax_style.imshow(tensor_to_image(style_img))
        ax_style.set_title('Style Image')
        ax_style.axis('off')
        
        # Plot CAV variations
        for idx, (transfer, strength) in enumerate(zip(transfers, strengths)):
            ax = fig.add_subplot(gs[:, idx + 1])
            ax.imshow(tensor_to_image(transfer))
            ax.set_title(f'CAV Strength: {strength:.1f}')
            ax.axis('off')
        
        # Add color bar showing CAV strength scale
        norm = plt.Normalize(min(strengths), max(strengths))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=norm)
        cbar_ax = fig.add_subplot(gs[:, -1])
        plt.colorbar(sm, cax=cbar_ax, label='CAV Strength')
        
        # Create unique filename for this content-style pair
        plot_filename = f'comparison_content{content_name}_style_{style_num:02d}.png'
        plot_path = os.path.join(save_dir, plot_filename)
        
        # Save the plot
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Saved comparison plot for content {content_name}, style {style_num} at: {plot_path}")


def apply_cav_and_visualize(cav_controller, content_image, style_image, cav, strengths, output_dir):
    """
    Applies CAV with different strengths and creates a comparison visualization.
    
    Args:
        cav_controller: StyleCAVController instance
        content_image: Content image tensor
        style_image: Style image tensor
        cav: Learned CAV tensor
        strengths: List of CAV strengths to apply
        output_dir: Directory to save the visualization
    """
    # Collect all CAV variations
    transfers = []
    for strength in strengths:
        transfer, _ = cav_controller.apply_cav(content_image, style_image, cav, strength)
        transfers.append(transfer)
    
    # Create visualizer and generate comparison plot
    visualizer = CAVVisualizer(len(strengths))
    plot_path = os.path.join(output_dir, 'cav_comparison.png')
    visualizer.create_comparison_plot(transfers, strengths, content_image, style_image, plot_path)
    logger.info(f"Saved CAV comparison plot at: {plot_path}")



class StyleCAVController:
    """
    Controls the learning and application of Concept Activation Vectors (CAVs)
    for style transfer modification. This class integrates with the existing
    style transfer architecture to enable concept-based control over styling.
    """
    def __init__(self, vgg, matrix, decoder, layer_name='r41', device='cuda'):
        self.vgg = vgg
        self.matrix = matrix
        self.decoder = decoder
        self.layer_name = layer_name
        self.device = device
        self.activations = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Registers hooks to capture feature activations after the matrix layer."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.activations.append(output[0].detach())
            else:
                self.activations.append(output.detach())
        
        self.matrix.register_forward_hook(hook_fn)
    
    def get_style_features(self, image_path):
        """
        Extracts transformed style features from a given image.
        
        Args:
            image_path: Path to the image file to process
        Returns:
            Tensor of transformed features at the target layer
        """
        self.activations = []
        
        try:
            # Load and preprocess the image
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
                    content_features = features[self.layer_name]
                    style_features = features[self.layer_name]
                else:
                    content_features = features
                    style_features = features
                
                transformed_features, _ = self.matrix(content_features, style_features)
            
            if not self.activations:
                raise ValueError("No activations captured during forward pass")
            
            return self.activations[-1]
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise
    
    def collect_training_images(self, style_dir):
        """
        Collects paths to training images from the directory structure.
        
        Args:
            style_dir: Base directory for a style (e.g., 'data/styled_outputs/style_01')
        Returns:
            Tuple of (original image path, list of blur variation paths)
        """
        style_path = Path(style_dir)
        logger.info(f"Looking for sigma directories in: {style_dir}")
        
        # Find and sort sigma directories
        sigma_dirs = []
        for item in style_path.glob("sigma_*"):
            if item.is_dir():
                sigma_value = float(item.name.split('_')[1])
                sigma_dirs.append((sigma_value, item))
        
        if not sigma_dirs:
            raise ValueError(f"No sigma directories found in {style_dir}")
        
        sigma_dirs.sort(key=lambda x: x[0])
        
        # Collect image paths
        image_paths = []
        for _, sigma_dir in sigma_dirs:
            png_files = list(sigma_dir.glob("*.png"))
            if png_files:
                image_paths.append(png_files[0])
                logger.info(f"Found image: {png_files[0]}")
        
        if not image_paths:
            raise ValueError("No valid image files found")
        
        return str(image_paths[0]), [str(p) for p in image_paths[1:]]
    
    def learn_cav_from_directory(self, style_dir):
        """
        Learns the CAV from existing blur variations in a style directory.
        
        Args:
            style_dir: Directory containing different blur sigma versions
        Returns:
            Tensor containing the learned concept direction
        """
        logger.info(f"Learning CAV from directory: {style_dir}")
        original_path, blur_paths = self.collect_training_images(style_dir)
        
        # Get features for original (least blurred) image
        original_features = self.get_style_features(original_path)
        original_flat = original_features.view(original_features.size(0), -1).cpu().numpy()
        
        # Get features for all blur variations
        blur_features = []
        for blur_path in blur_paths:
            features = self.get_style_features(blur_path)
            blur_features.append(features.view(features.size(0), -1).cpu().numpy())
        
        blur_flat = np.concatenate(blur_features, axis=0)
        
        # Prepare data for SVM training
        features = np.concatenate([original_flat, blur_flat])
        labels = np.concatenate([
            np.zeros(len(original_flat)),
            np.ones(len(blur_flat))
        ])
        
        # Train SVM to find concept direction
        svm = LinearSVC(C=1.0, dual="auto")
        svm.fit(features, labels)
        
        # Convert SVM direction to tensor
        cav = torch.tensor(svm.coef_[0]).reshape(original_features.shape[1:]).to(self.device)
        logger.info("Successfully learned CAV")
        return cav

    def apply_cav(self, content_image, style_image, cav, strength=1.0):
        """
        Applies the learned CAV during style transfer with enhanced debugging and control.
        This version includes detailed analysis of feature modifications and scale management.
        
        Args:
            content_image: Input content image tensor
            style_image: Input style image tensor
            cav: Learned concept activation vector
            strength: Scaling factor for CAV application
            
        Returns:
            Tuple of (modified image, transformation matrix)
        """
        with torch.no_grad():
            # Convert inputs to float32 and log their statistics
            content_image = content_image.float()
            style_image = style_image.float()
            
            logger.info(f"\nCAV Application Debug:")
            logger.info(f"Content image range: [{content_image.min():.3f}, {content_image.max():.3f}]")
            logger.info(f"Style image range: [{style_image.min():.3f}, {style_image.max():.3f}]")
            
            # Extract and analyze features
            content_features = self.vgg(content_image)[self.layer_name]
            style_features = self.vgg(style_image)[self.layer_name]
            
            logger.info(f"Content features range: [{content_features.min():.3f}, {content_features.max():.3f}]")
            logger.info(f"Style features range: [{style_features.min():.3f}, {style_features.max():.3f}]")
            
            # Get transformed features and analyze
            transformed_features, matrix = self.matrix(content_features, style_features)
            transformed_features = transformed_features.float()
            
            logger.info(f"Transformed features stats:")
            logger.info(f"  Range: [{transformed_features.min():.3f}, {transformed_features.max():.3f}]")
            logger.info(f"  Mean: {transformed_features.mean():.3f}")
            logger.info(f"  Std: {transformed_features.std():.3f}")
            
            # Analyze CAV before application
            cav = cav.float()
            cav_norm = torch.norm(cav)
            feature_norm = torch.norm(transformed_features)
            
            logger.info(f"\nCAV analysis:")
            logger.info(f"CAV norm: {cav_norm:.3f}")
            logger.info(f"Feature norm: {feature_norm:.3f}")
            
            # Scale CAV to match feature magnitude
            scale_factor = feature_norm / cav_norm
            adaptive_strength = strength * scale_factor * 0.1  # Start with 10% of the feature magnitude
            
            # Expand CAV and apply with adaptive strength
            cav_expanded = cav.unsqueeze(0)
            modified_features = transformed_features + (adaptive_strength * cav_expanded).float()
            
            # Analyze the effect of CAV application
            feature_diff = modified_features - transformed_features
            relative_change = torch.norm(feature_diff) / feature_norm
            
            logger.info(f"\nCAV modification analysis:")
            logger.info(f"Applied strength: {adaptive_strength:.3f}")
            logger.info(f"Relative feature change: {relative_change:.3f}")
            logger.info(f"Modified features range: [{modified_features.min():.3f}, {modified_features.max():.3f}]")
            
            # Generate and analyze final image
            result = self.decoder(modified_features)
            
            logger.info(f"\nFinal image stats:")
            logger.info(f"Output range: [{result.min():.3f}, {result.max():.3f}]")
            logger.info(f"Output mean: {result.mean():.3f}")
            
            return result.clamp(0, 1), matrix


def parse_args():
    """Sets up command line argument parsing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data",
                      help='base directory containing styled_outputs')
    parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                      help='pre-trained encoder path')
    parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                      help='pre-trained decoder path')
    parser.add_argument("--matrixPath", default='models/r41.pth',
                      help='pre-trained model path')
    parser.add_argument("--stylePath", default="data/style/",
                      help='path to style image')
    parser.add_argument("--contentPath", default="data/content/",
                      help='path to frames')
    parser.add_argument("--outf", default="Artistic/blur/",
                      help='path to output images')
    parser.add_argument("--matrixOutf", default="Matrices/",
                      help='path to save transformation matrices')
    parser.add_argument("--batchSize", type=int, default=1,
                      help='batch size')
    parser.add_argument('--loadSize', type=int, default=256,
                      help='scale image size')
    parser.add_argument('--fineSize', type=int, default=256,
                      help='crop image size')
    parser.add_argument("--layer", default="r41",
                      help='which features to transfer, either r31 or r41')
    parser.add_argument("--cav_base_strength", type=float, default=0.1,
                      help='base strength multiplier for CAV application')
    parser.add_argument("--cav_scale_mode", choices=['adaptive', 'fixed'], default='adaptive',
                      help='how to scale CAV effects')
    parser.add_argument("--cav_strengths", type=float, nargs='+',
                      default=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
                      help='strengths at which to apply CAV')
    return parser.parse_args()

def main():
    """Main function implementing CAV-controlled style transfer."""
    opt = parse_args()
    opt.cuda = torch.cuda.is_available()
    print_options(opt)
    
    # Create output directories
    os.makedirs(opt.outf, exist_ok=True)
    os.makedirs(opt.matrixOutf, exist_ok=True)
    cudnn.benchmark = True
    
    # Initialize data loaders
    content_dataset = Dataset(opt.contentPath, opt.loadSize, opt.fineSize)
    content_loader = torch.utils.data.DataLoader(dataset=content_dataset,
                                               batch_size=opt.batchSize,
                                               shuffle=False,
                                               num_workers=0)
    
    style_dataset = Dataset(opt.stylePath, opt.loadSize, opt.fineSize)
    style_loader = torch.utils.data.DataLoader(dataset=style_dataset,
                                             batch_size=opt.batchSize,
                                             shuffle=False,
                                             num_workers=0)
    
    # Initialize models
    vgg = encoder4() if opt.layer == 'r41' else encoder3()
    dec = decoder4() if opt.layer == 'r41' else decoder3()
    matrix = MulLayer(opt.layer)
    
    # Load model weights
    vgg.load_state_dict(torch.load(opt.vgg_dir))
    dec.load_state_dict(torch.load(opt.decoder_dir))
    matrix.load_state_dict(torch.load(opt.matrixPath))
    
    # Move models to GPU if available
    if opt.cuda:
        vgg.cuda()
        dec.cuda()
        matrix.cuda()
    
    # Initialize CAV controller
    cav_controller = StyleCAVController(vgg, matrix, dec, opt.layer)
    
    # Process each content-style pair
def main():
    """
    Main function implementing CAV-controlled style transfer with corrected file naming.
    This version properly handles content and style image names for output files.
    """
    opt = parse_args()
    opt.cuda = torch.cuda.is_available()
    print_options(opt)
    
    # Create output directories
    os.makedirs(opt.outf, exist_ok=True)
    os.makedirs(opt.matrixOutf, exist_ok=True)
    cudnn.benchmark = True
    
    # Initialize data loaders
    content_dataset = Dataset(opt.contentPath, opt.loadSize, opt.fineSize)
    content_loader = torch.utils.data.DataLoader(dataset=content_dataset,
                                               batch_size=opt.batchSize,
                                               shuffle=False,
                                               num_workers=0)
    
    style_dataset = Dataset(opt.stylePath, opt.loadSize, opt.fineSize)
    style_loader = torch.utils.data.DataLoader(dataset=style_dataset,
                                             batch_size=opt.batchSize,
                                             shuffle=False,
                                             num_workers=0)
    
    # Initialize models
    vgg = encoder4() if opt.layer == 'r41' else encoder3()
    dec = decoder4() if opt.layer == 'r41' else decoder3()
    matrix = MulLayer(opt.layer)
    
    # Load model weights
    vgg.load_state_dict(torch.load(opt.vgg_dir))
    dec.load_state_dict(torch.load(opt.decoder_dir))
    matrix.load_state_dict(torch.load(opt.matrixPath))
    
    if opt.cuda:
        vgg.cuda()
        dec.cuda()
        matrix.cuda()
    
    cav_controller = StyleCAVController(vgg, matrix, dec, opt.layer)
    comparison_dir = os.path.join(opt.outf, 'comparisons')
    os.makedirs(comparison_dir, exist_ok=True)
    visualizer = CAVVisualizer(len(opt.cav_strengths))
    
    # Process each content-style pair
    for content_idx, (content, contentName) in enumerate(content_loader):
            # Convert content tensor name to string and get the base filename
            if isinstance(contentName, (list, tuple)):
                content_name = contentName[0]  # Extract from list/tuple if needed
            else:
                content_name = contentName
            
            content_num = content_idx + 1
            contentV = content.cuda() if opt.cuda else content
            
            for style_idx, (style, styleName) in enumerate(style_loader):
                # Convert style tensor name to string and extract style number
                if isinstance(styleName, (list, tuple)):
                    style_name = styleName[0]  # Extract from list/tuple if needed
                else:
                    style_name = styleName
                
                styleV = style.cuda() if opt.cuda else style
                
                # Get style number ensuring it starts from 1
                style_num = max(1, style_idx + 1)
                style_output_dir = os.path.join(opt.data_dir, 'styled_outputs', f'style_{style_num:02d}')
                
                logger.info(f"Processing content {content_num:02d} with style {style_num:02d}")
                
                try:
                    # Learn CAV from existing blur variations
                    blur_cav = cav_controller.learn_cav_from_directory(style_output_dir)
                    
                    transfers = []
                    # Create new variations with different CAV strengths
                    for strength in opt.cav_strengths:
                        # Apply CAV modification
                        transfer, matrix = cav_controller.apply_cav(
                            contentV, styleV, blur_cav, strength
                        )
                        
                        transfers.append(transfer)
                        # Create descriptive filename using indices
                        output_filename = (
                            f'cav_content{content_num:02d}_'
                            f'style_{style_num:02d}_'
                            f'strength_{strength:.1f}.png'
                        )
                        output_path = os.path.join(opt.outf, output_filename)
                        
                        # Create corresponding matrix filename
                        matrix_filename = (
                        f'cav_content{content_idx:02d}_'
                        f'style_{style_num:02d}_'
                        f'strength_{strength:.1f}_matrix.pth'
                    )
                        matrix_path = os.path.join(opt.matrixOutf, matrix_filename)
                        torch.save(matrix, matrix_path)
                    
                    # Create comparison plot for this content-style pair
                    visualizer.create_comparison_plot(
                        transfers,
                        opt.cav_strengths,
                        contentV,
                        styleV,
                        comparison_dir,
                        content_idx,
                        style_num
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing content {content_idx} with style {style_num}: {str(e)}")
                    continue

if __name__ == "__main__":
    main()