import os
import torch
import argparse
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import timm
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.svm import LinearSVC
from pathlib import Path
import logging
import traceback
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Optional, Union
import torch.nn.functional as F

# Import existing style transfer components
from libs.Loader import Dataset
from libs.Matrix import MulLayer
from libs.utils import print_options
from libs.models import encoder3, encoder4, decoder3, decoder4

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('style_transfer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_device(gpu_id: int = 0) -> torch.device:
    """
    Set up and return the appropriate device for computation.
    
    Args:
        gpu_id: The GPU ID to use if CUDA is available
        
    Returns:
        torch.device: The configured device
    """
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)
        cudnn.benchmark = True
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device

def load_model(model, weights_path: str, device: torch.device) -> torch.nn.Module:
    """
    Safely load a model with proper error handling.
    
    Args:
        model: The model instance to load weights into
        weights_path: Path to the model weights
        device: Device to load the model on
        
    Returns:
        torch.nn.Module: The loaded model
    """
    try:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
        model = model.to(device)
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    except Exception as e:
        logger.error(f"Failed to load model from {weights_path}: {str(e)}")
        raise

class CAVVisualizer:
    """Creates and saves comparison plots of different CAV strength applications."""
    
    def __init__(self, num_strengths: int):
        self.fig_size = (15, 8)
        self.num_strengths = num_strengths
        logger.info(f"Initialized CAVVisualizer with {num_strengths} strength levels")
    
    @staticmethod
    def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
        """
        Safely convert a tensor to a numpy image array.
        
        Args:
            tensor: Input tensor to convert
            
        Returns:
            np.ndarray: The converted image array
        """
        try:
            # Ensure tensor is detached and on CPU
            tensor = tensor.detach().cpu()
            
            # Handle different tensor dimensions
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            
            # Convert to numpy and ensure proper range
            img = tensor.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            return img
        
        except Exception as e:
            logger.error(f"Error converting tensor to image: {str(e)}")
            raise
    
    def create_comparison_plot(
        self,
        transfers: List[torch.Tensor],
        strengths: List[float],
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        save_dir: str,
        content_name: str,
        style_num: int
    ) -> str:
        """
        Create and save a comparison plot of different CAV strengths.
        
        Args:
            transfers: List of stylized image tensors
            strengths: List of CAV strength values used
            content_img: Original content image tensor
            style_img: Original style image tensor
            save_dir: Directory to save the plot
            content_name: Content image identifier
            style_num: Style image identifier
            
        Returns:
            str: Path to the saved plot
        """
        try:
            fig = plt.figure(figsize=self.fig_size, constrained_layout=True)
            gs = GridSpec(2, self.num_strengths + 2, figure=fig)
            
            # Plot content and style
            ax_content = fig.add_subplot(gs[0, 0])
            ax_content.imshow(self.tensor_to_image(content_img))
            ax_content.set_title('Content Image')
            ax_content.axis('off')
            
            ax_style = fig.add_subplot(gs[1, 0])
            ax_style.imshow(self.tensor_to_image(style_img))
            ax_style.set_title('Style Image')
            ax_style.axis('off')
            
            # Plot transfers with different strengths
            for idx, (transfer, strength) in enumerate(zip(transfers, strengths)):
                ax = fig.add_subplot(gs[:, idx + 1])
                ax.imshow(self.tensor_to_image(transfer))
                ax.set_title(f'CAV Strength: {strength:.1f}')
                ax.axis('off')
            
            # Add colorbar
            norm = plt.Normalize(min(strengths), max(strengths))
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=norm)
            cbar_ax = fig.add_subplot(gs[:, -1])
            plt.colorbar(sm, cax=cbar_ax, label='CAV Strength')
            
            # Save plot
            os.makedirs(save_dir, exist_ok=True)
            plot_path = os.path.join(
                save_dir,
                f'comparison_content{content_name}_style_{style_num:02d}.png'
            )
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            logger.info(f"Saved comparison plot to {plot_path}")
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating comparison plot: {str(e)}")
            raise

def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(description="Style Transfer with CAV Control")
    
    parser.add_argument("--data_dir", default="data",
                      help='Base directory containing styled outputs')
    parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                      help='Pre-trained encoder path')
    parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                      help='Pre-trained decoder path')
    parser.add_argument("--matrixPath", default='models/r41.pth',
                      help='Pre-trained matrix path')
    parser.add_argument("--stylePath", default="data/style/",
                      help='Path to style images')
    parser.add_argument("--contentPath", default="data/content/",
                      help='Path to content images')
    parser.add_argument("--outf", default="Artistic/blur/",
                      help='Path to output images')
    parser.add_argument("--matrixOutf", default="Matrices/",
                      help='Path to save transformation matrices')
    parser.add_argument("--batchSize", type=int, default=1,
                      help='Batch size')
    parser.add_argument('--loadSize', type=int, default=256,
                      help='Scale image size')
    parser.add_argument('--fineSize', type=int, default=256,
                      help='Crop image size')
    parser.add_argument("--layer", default="r41",
                      help='Features to transfer (r31 or r41)')
    parser.add_argument("--gpu", type=int, default=0,
                      help='GPU ID to use')
    parser.add_argument("--cav_strengths", type=float, nargs='+',
                      default=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
                      help='CAV strength values to test')
    
    args = parser.parse_args()
    
    # Validate paths
    required_paths = [
        ('VGG weights', args.vgg_dir),
        ('Decoder weights', args.decoder_dir),
        ('Matrix weights', args.matrixPath),
        ('Style directory', args.stylePath),
        ('Content directory', args.contentPath)
    ]
    
    for name, path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found at: {path}")
    
    return args

def process_batch(
    vgg: torch.nn.Module,
    decoder: torch.nn.Module,
    matrix: torch.nn.Module,
    content: torch.Tensor,
    style: torch.Tensor,
    layer: str,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process a single content-style pair with error handling.
    
    Args:
        vgg: Encoder model
        decoder: Decoder model
        matrix: Transform matrix model
        content: Content image tensor
        style: Style image tensor
        layer: Layer to use for feature extraction
        device: Computation device
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (Stylized image, transformation matrix)
    """
    try:
        with torch.no_grad():
            # Extract features
            content_features = vgg(content)[layer]
            style_features = vgg(style)[layer]
            
            # Apply style transfer
            transformed_features, transform_matrix = matrix(content_features, style_features)
            stylized_image = decoder(transformed_features)
            
            # Ensure output is in valid range
            stylized_image = torch.clamp(stylized_image, 0, 1)
            
            return stylized_image, transform_matrix
            
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise

def main():
    """Main function implementing CAV-controlled style transfer."""
    try:
        # Parse arguments and set up
        args = parse_args()
        device = setup_device(args.gpu)
        
        # Create output directories
        os.makedirs(args.outf, exist_ok=True)
        os.makedirs(args.matrixOutf, exist_ok=True)
        comparison_dir = os.path.join(args.outf, 'comparisons')
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Initialize models
        vgg = encoder4() if args.layer == 'r41' else encoder3()
        vgg = load_model(vgg, args.vgg_dir, device)
        
        decoder = decoder4() if args.layer == 'r41' else decoder3()
        decoder = load_model(decoder, args.decoder_dir, device)
        
        matrix = MulLayer(args.layer)
        matrix = load_model(matrix, args.matrixPath, device)
        
        # Initialize data loaders
        content_dataset = Dataset(args.contentPath, args.loadSize, args.fineSize)
        content_loader = torch.utils.data.DataLoader(
            dataset=content_dataset,
            batch_size=args.batchSize,
            shuffle=False,
            num_workers=0
        )
        
        style_dataset = Dataset(args.stylePath, args.loadSize, args.fineSize)
        style_loader = torch.utils.data.DataLoader(
            dataset=style_dataset,
            batch_size=args.batchSize,
            shuffle=False,
            num_workers=0
        )
        
        # Initialize visualization
        visualizer = CAVVisualizer(len(args.cav_strengths))
        
        # Process each content-style pair
        for content_idx, (content, _) in enumerate(content_loader):
            content_num = content_idx + 1
            content = content.to(device)
            
            for style_idx, (style, _) in enumerate(style_loader):
                style_num = style_idx + 1
                style = style.to(device)
                
                try:
                    logger.info(f"Processing content {content_num} with style {style_num}")
                    
                    # Process with different CAV strengths
                    transfers = []
                    for strength in args.cav_strengths:
                        # Apply style transfer
                        transfer, transform_matrix = process_batch(
                            vgg, decoder, matrix,
                            content, style,
                            args.layer, device
                        )
                        transfers.append(transfer)
                        
                        # Save stylized image
                        output_filename = (
                            f'content{content_num:02d}_'
                            f'style_{style_num:02d}_'
                            f'strength_{strength:.1f}.png'
                        )
                        output_path = os.path.join(args.outf, output_filename)
                        vutils.save_image(transfer.detach(), output_path)
                        
                        # Save transformation matrix
                        matrix_filename = (
                            f'content{content_num:02d}_'
                            f'style_{style_num:02d}_'
                            f'strength_{strength:.1f}_matrix.pth'
                        )
                        matrix_path = os.path.join(args.matrixOutf, matrix_filename)
                        torch.save(transform_matrix.detach().cpu(), matrix_path)
                    
                    # Create comparison visualization
                    visualizer.create_comparison_plot(
                        transfers,
                        args.cav_strengths,
                        content,
                        style,
                        comparison_dir,
                        content_num,
                        style_num
                    )
                    
                except Exception as e:
                    logger.error(
                        f"Error processing content {content_num} with style {style_num}: {str(e)}\n"
                        f"Traceback: {traceback.format_exc()}"
                    )
                    continue
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in main process: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()