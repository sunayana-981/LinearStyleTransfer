import os
import torch
import argparse
import logging
from PIL import Image
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# Import from existing script
from libs.Loader import Dataset
from libs.Matrix import MulLayer
from libs.utils import print_options
from libs.models import encoder3, encoder4, decoder3, decoder4

# Import CAV-related components from the original script
from CAV_blur import StyleCAVController, CAVVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DualStyleCAVController:
    """
    Controls style transfer with two different styles and their respective CAVs.
    Enables style interpolation and per-style CAV strength control.
    """
    def __init__(self, vgg, matrix, decoder, layer_name='r41', device='cuda'):
        self.base_controller = StyleCAVController(vgg, matrix, decoder, layer_name, device)
        self.device = device
        self.layer_name = layer_name
    
    def interpolate_features(self, features1, features2, alpha):
        """
        Interpolates between two sets of features based on alpha.
        
        Args:
            features1: First set of features
            features2: Second set of features
            alpha: Interpolation factor (0 to 1), where 0 is pure style1 and 1 is pure style2
        """
        return (1 - alpha) * features1 + alpha * features2
    
    def apply_dual_style_cav(self, content_image, style1_image, style2_image, 
                            cav1, cav2, style_alpha=0.5, cav1_strength=1.0, cav2_strength=1.0):
        """
        Applies two styles with their respective CAVs, with controllable interpolation.
        
        Args:
            content_image: Input content image tensor
            style1_image: First style image tensor
            style2_image: Second style image tensor
            cav1: CAV for first style
            cav2: CAV for second style
            style_alpha: Interpolation factor between styles (0 to 1)
            cav1_strength: Strength of first CAV
            cav2_strength: Strength of second CAV
        
        Returns:
            Tuple of (modified image, transformation matrices)
        """
        with torch.no_grad():
            # Extract features
            content_features = self.base_controller.vgg(content_image)[self.layer_name]
            style1_features = self.base_controller.vgg(style1_image)[self.layer_name]
            style2_features = self.base_controller.vgg(style2_image)[self.layer_name]
            
            # Transform features with both styles
            transformed1, matrix1 = self.base_controller.matrix(content_features, style1_features)
            transformed2, matrix2 = self.base_controller.matrix(content_features, style2_features)
            
            # Interpolate between transformed features
            interpolated = self.interpolate_features(transformed1, transformed2, style_alpha)
            
            # Apply CAVs with respective strengths
            if cav1_strength != 0:
                cav1_effect = self.base_controller.apply_cav(
                    content_image, style1_image, cav1, cav1_strength)[0]
            else:
                cav1_effect = transformed1
                
            if cav2_strength != 0:
                cav2_effect = self.base_controller.apply_cav(
                    content_image, style2_image, cav2, cav2_strength)[0]
            else:
                cav2_effect = transformed2
            
            # Interpolate between CAV effects
            final_features = self.interpolate_features(cav1_effect, cav2_effect, style_alpha)
            
            # Generate final image
            result = self.base_controller.decoder(final_features)
            
            return result.clamp(0, 1), (matrix1, matrix2)

class DualStyleVisualizer(CAVVisualizer):
    """
    Extended visualizer for dual style transfer results.
    """
    def create_dual_comparison_plot(self, transfers, style_alphas, cav_strengths,
                                  content_img, style1_img, style2_img, 
                                  save_dir, content_name, style1_num, style2_num):
        """
        Creates a grid visualization showing style interpolation and CAV effects.
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(len(cav_strengths) + 2, len(style_alphas) + 2)
        
        # Plot original images
        self._plot_original_images(fig, gs, content_img, style1_img, style2_img)
        
        # Plot variations
        for i, strength in enumerate(cav_strengths):
            for j, alpha in enumerate(style_alphas):
                ax = fig.add_subplot(gs[i + 2, j + 1])
                ax.imshow(self.tensor_to_image(transfers[i][j]))
                ax.set_title(f'α={alpha:.1f}, CAV={strength:.1f}')
                ax.axis('off')
        
        # Save plot
        plot_filename = f'dual_comparison_content{content_name}_styles_{style1_num:02d}_{style2_num:02d}.png'
        plot_path = os.path.join(save_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()


def parse_args():
    """Sets up command line argument parsing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--style1Path", default="data/style/",
                      help='path to first style images')
    parser.add_argument("--style2Path", default="data/style1/",
                      help='path to second style images')
    parser.add_argument("--style_alphas", type=float, nargs='+',
                      default=[0.0, 0.25, 0.5, 0.75, 1.0],
                      help='style interpolation factors')
    parser.add_argument("--cav1_strengths", type=float, nargs='+',
                      default=[-1.0, 0.0, 1.0],
                      help='CAV strengths for first style')
    parser.add_argument("--cav2_strengths", type=float, nargs='+',
                      default=[-1.0, 0.0, 1.0],
                      help='CAV strengths for second style')
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

def load_and_preprocess_image(image_path, device='cuda'):
    """
    Loads and preprocesses an image for the style transfer network.
    
    Args:
        image_path: Path to the image file
        device: Device to load the tensor to
    Returns:
        Preprocessed image tensor
    """
    try:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.multiply(255))  # Scale to 0-255 range for VGG
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor.to(device)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        raise

def main():
    """Main function implementing dual-style transfer with CAV control."""
    opt = parse_args()
    opt.cuda = torch.cuda.is_available()
    print_options(opt)

    # Create output directories
    os.makedirs(opt.outf, exist_ok=True)
    os.makedirs(opt.matrixOutf, exist_ok=True)
    cudnn.benchmark = True

    #Initialize dataloaders
    content_dataset = Dataset(opt.contentPath, opt.loadSize, opt.fineSize)
    content_loader = torch.utils.data.DataLoader(dataset=content_dataset,
                                               batch_size=opt.batchSize,
                                               shuffle=False,
                                               num_workers=0)
    
    style_dataset = Dataset(opt.stylePath, opt.loadSize, opt.fineSize)
    style1_loader = torch.utils.data.DataLoader(dataset=style_dataset,
                                             batch_size=opt.batchSize,
                                             shuffle=False,
                                             num_workers=0)
    
    style_dataset = Dataset(opt.stylePath, opt.loadSize, opt.fineSize)
    style2_loader = torch.utils.data.DataLoader(dataset=style_dataset,
                                                batch_size=opt.batchSize,
                                                shuffle=False,
                                                num_workers=0)
    
    # Initialize models and controllers
    vgg = encoder4() if opt.layer == 'r41' else encoder3()
    dec = decoder4() if opt.layer == 'r41' else decoder3()
    matrix = MulLayer(opt.layer)
    
    # Load weights and move to GPU if available
    vgg.load_state_dict(torch.load(opt.vgg_dir))
    dec.load_state_dict(torch.load(opt.decoder_dir))
    matrix.load_state_dict(torch.load(opt.matrixPath))
    
    if opt.cuda:
        vgg.cuda()
        dec.cuda()
        matrix.cuda()
    
    # Initialize dual style controller
    dual_controller = DualStyleCAVController(vgg, matrix, dec, opt.layer)
    visualizer = DualStyleVisualizer(max(len(opt.style_alphas), len(opt.cav1_strengths)))


    
    # Process each content image
    for content_idx, (content, content_name) in enumerate(content_loader):
        contentV = content.cuda() if opt.cuda else content
        
        # Process each pair of styles
        for style1_idx, (style1, style1_name) in enumerate(style1_loader):
            style1V = style1.cuda() if opt.cuda else style1
            
            for style2_idx, (style2, style2_name) in enumerate(style2_loader):
                style2V = style2.cuda() if opt.cuda else style2
                
                try:
                    # Learn CAVs for both styles
                    style1_dir = os.path.join(opt.data_dir, 'styled_outputs', f'style_{style1_idx+1:02d}')
                    style2_dir = os.path.join(opt.data_dir, 'styled_outputs', f'style_{style2_idx+1:02d}')
                    
                    cav1 = dual_controller.base_controller.learn_cav_from_directory(style1_dir)
                    cav2 = dual_controller.base_controller.learn_cav_from_directory(style2_dir)
                    
                    # Generate variations with different style interpolations and CAV strengths
                    transfers = []
                    for cav1_strength in opt.cav1_strengths:
                        style_variations = []
                        for alpha in opt.style_alphas:
                            result, matrices = dual_controller.apply_dual_style_cav(
                                contentV, style1V, style2V, cav1, cav2,
                                style_alpha=alpha,
                                cav1_strength=cav1_strength,
                                cav2_strength=opt.cav2_strengths[0]  # Use first strength for simplicity
                            )
                            style_variations.append(result)
                            
                            # Save individual result
                            output_filename = (
                                f'dual_content{content_idx:02d}_'
                                f'style1_{style1_idx:02d}_'
                                f'style2_{style2_idx:02d}_'
                                f'alpha_{alpha:.2f}_'
                                f'cav1_{cav1_strength:.1f}_'
                                f'cav2_{opt.cav2_strengths[0]:.1f}.png'
                            )
                            save_path = os.path.join(opt.outf, output_filename)
                            vutils.save_image(result, save_path)
                        
                        transfers.append(style_variations)
                    
                    # Create comparison visualization
                    visualizer.create_dual_comparison_plot(
                        transfers, opt.style_alphas, opt.cav1_strengths,
                        contentV, style1V, style2V,
                        opt.outf, content_idx, style1_idx+1, style2_idx+1
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing content {content_idx} with styles {style1_idx}, {style2_idx}: {str(e)}")
                    continue

if __name__ == "__main__":
    main()