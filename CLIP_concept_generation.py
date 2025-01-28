import os
import torch
import argparse
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from torchvision import transforms
from PIL import Image
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
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

class StyleTransferCAV:
    """Controls the learning and application of CAVs for style transfer modification."""
    def __init__(self, vgg, matrix, decoder, layer_name='r41', device='cuda'):
        self.vgg = vgg
        self.matrix = matrix
        self.decoder = decoder
        self.layer_name = layer_name
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Configure logging
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def get_features(self, content_img, style_img):
        """Extracts and transforms features from content and style images."""
        try:
            with torch.no_grad():
                content_features = self.vgg(content_img)[self.layer_name]
                style_features = self.vgg(style_img)[self.layer_name]
                transformed_features, _ = self.matrix(content_features, style_features)
                
                self.logger.debug(f"Feature shape: {transformed_features.shape}")
                return transformed_features
                
        except Exception as e:
            self.logger.error(f"Error in feature extraction: {str(e)}")
            raise

    def learn_concept_cav(self, positive_pairs, negative_pairs):
        """
        Learns a concept activation vector (CAV) from positive and negative style transfer pairs.
        
        Args:
            positive_pairs: List of tuples (content_img, style_img) for positive examples
            negative_pairs: List of tuples (content_img, style_img) for negative examples
        """
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.feature_selection import VarianceThreshold

            self.logger.info("Extracting features for CAV learning...")
            
            # Extract features from positive and negative pairs
            pos_features_list = []
            neg_features_list = []
            
            for content, style in positive_pairs:
                features = self.get_features(content, style)
                pos_features_list.append(features.view(features.size(0), -1).cpu().numpy())
                
            for content, style in negative_pairs:
                features = self.get_features(content, style)
                neg_features_list.append(features.view(features.size(0), -1).cpu().numpy())

            pos_features = np.concatenate(pos_features_list, axis=0)
            neg_features = np.concatenate(neg_features_list, axis=0)
            
            self.logger.info(f"Positive features shape: {pos_features.shape}")
            self.logger.info(f"Negative features shape: {neg_features.shape}")

            # Standardize features
            scaler = StandardScaler()
            X = np.concatenate([pos_features, neg_features])
            X_scaled = scaler.fit_transform(X)
            
            # Handle NaN and Inf values
            self.logger.info(f"Feature statistics - Min: {np.min(X_scaled)}, Max: {np.max(X_scaled)}")
            self.logger.info(f"NaN count: {np.isnan(X_scaled).sum()}, Inf count: {np.isinf(X_scaled).sum()}")
            X_scaled = np.nan_to_num(X_scaled)

            # Feature selection
            selector = VarianceThreshold(threshold=1e-5)
            X_selected = selector.fit_transform(X_scaled)
            
            # PCA
            n_samples, n_features = X_selected.shape
            n_components = min(50, n_features, n_samples)
            self.logger.info(f"Using {n_components} components for PCA")
            
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_selected)

            # Train SVM
            self.logger.info("Training SVM...")
            labels = np.concatenate([np.ones(len(pos_features)), np.zeros(len(neg_features))])
            svm = LinearSVC(C=1.0, dual=False, max_iter=1000)
            svm.fit(X_pca, labels)

            # Get CAV and transform back
            cav_pca = svm.coef_[0]
            cav_full = pca.inverse_transform(cav_pca)
            
            # Get original feature shape from first positive example
            orig_features = self.get_features(positive_pairs[0][0], positive_pairs[0][1])
            orig_shape = orig_features.shape
            
            # Reshape CAV to match feature dimensions
            cav_tensor = torch.tensor(cav_full, dtype=torch.float32, device=self.device)
            cav_reshaped = cav_tensor.view(orig_shape[1], orig_shape[2], orig_shape[3])
            
            # Average across spatial dimensions
            cav_channel = torch.mean(cav_reshaped, dim=(1, 2)).view(-1, 1, 1)
            
            # Normalize
            cav_normalized = cav_channel / (torch.norm(cav_channel) + 1e-8)
            
            self.logger.info(f"Final CAV shape: {cav_normalized.shape}")
            return cav_normalized

        except Exception as e:
            self.logger.error(f"Error in CAV learning: {str(e)}")
            raise

    def apply_cav(self, content_image, style_image, cav, strength=1.0):
        """Applies the learned CAV during style transfer."""
        try:
            with torch.no_grad():
                # Get transformed features
                content_features = self.vgg(content_image)[self.layer_name]
                style_features = self.vgg(style_image)[self.layer_name]
                transformed_features, transform_matrix = self.matrix(content_features, style_features)
                
                self.logger.debug(f"Feature statistics:")
                self.logger.debug(f"Range: [{transformed_features.min():.3f}, {transformed_features.max():.3f}]")
                self.logger.debug(f"Mean: {transformed_features.mean():.3f}")
                
                # Prepare CAV
                cav = cav.to(transformed_features.device)
                cav_expanded = cav.expand(-1, transformed_features.shape[2], transformed_features.shape[3])
                
                # Calculate adaptive strength
                feature_norms = torch.norm(transformed_features, dim=1, keepdim=True)
                cav_norm = torch.norm(cav_expanded, dim=1, keepdim=True) + 1e-8
                scale_factor = feature_norms / cav_norm
                
                # Apply CAV
                modified_features = transformed_features + (strength * scale_factor * cav_expanded)
                
                # Generate output
                output = self.decoder(modified_features)
                output = torch.clamp(output, 0, 1)
                
                return output, transform_matrix

        except Exception as e:
            self.logger.error(f"Error applying CAV: {str(e)}")
            raise

class CAVVisualizer:
    """Creates comparison visualizations for different CAV strength applications."""
    def __init__(self, num_strengths):
        self.fig_size = (15, 8)
        self.num_strengths = num_strengths
        
    def create_comparison_plot(self, transfers, strengths, content_img, style_img, 
                             save_dir, content_idx, style_idx):
        """Creates and saves a comprehensive comparison plot."""
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
            ax.set_title(f'CAV Strength: {strength:.1f}')
            ax.axis('off')
        
        # Add strength scale
        norm = plt.Normalize(min(strengths), max(strengths))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=norm)
        cbar_ax = fig.add_subplot(gs[:, -1])
        plt.colorbar(sm, cax=cbar_ax, label='CAV Strength')
        
        # Save
        plot_filename = f'comparison_content{content_idx:02d}_style_{style_idx:02d}.png'
        plot_path = os.path.join(save_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Style Transfer CAV Training')
    parser.add_argument('--content_dir', default='data/content', help='Content image directory')
    parser.add_argument('--style_dir', default='data/style', help='Style image directory')
    parser.add_argument('--output_dir', default='output/cav_results', help='Output directory')
    parser.add_argument('--vgg_dir', default='models/vgg_r41.pth', help='VGG model path')
    parser.add_argument('--decoder_dir', default='models/dec_r41.pth', help='Decoder model path')
    parser.add_argument('--matrix_path', default='models/r41.pth', help='Matrix model path')
    parser.add_argument('--cav_strengths', type=float, nargs='+', 
                        default=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
                        help='CAV strength values')
    opt = parser.parse_args()

    # Setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    os.makedirs(opt.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    vgg = encoder4().to(device)
    dec = decoder4().to(device)
    matrix = MulLayer('r41').to(device)
    
    vgg.load_state_dict(torch.load(opt.vgg_dir))
    dec.load_state_dict(torch.load(opt.decoder_dir))
    matrix.load_state_dict(torch.load(opt.matrix_path))
    
    # Initialize controller
    cav_controller = StyleTransferCAV(vgg, matrix, dec, 'r41', device)
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    
    # Load content and style images
    def load_images(directory):
        images = []
        for img_path in Path(directory).glob('*.jpg'):
            img = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
            images.append((img, img_path.stem))
        return images
    
    content_images = load_images(opt.content_dir)
    style_images = load_images(opt.style_dir)
    
    # Process each content-style pair
    visualizer = CAVVisualizer(len(opt.cav_strengths))
    
    for content_idx, (content_img, content_name) in enumerate(content_images):
        for style_idx, (style_img, style_name) in enumerate(style_images):
            logger.info(f"Processing content {content_name} with style {style_name}")
            
            try:
                # Create positive and negative pairs
                positive_pairs = [(content_img, style_img)]
                negative_pairs = []
                for other_style_img, _ in style_images:
                    if other_style_img is not style_img:
                        negative_pairs.append((content_img, other_style_img))
                
                # Learn CAV
                cav = cav_controller.learn_concept_cav(positive_pairs, negative_pairs)
                
                # Apply CAV with different strengths
                transfers = []
                for strength in opt.cav_strengths:
                    transfer, _ = cav_controller.apply_cav(content_img, style_img, cav, strength)
                    transfers.append(transfer)
                    
                    # Save individual results
                    output_filename = f'cav_content{content_idx:02d}_style_{style_idx:02d}_strength_{strength:.1f}.png'
                    output_path = os.path.join(opt.output_dir, output_filename)
                    vutils.save_image(transfer, output_path)
                
                # Create comparison visualization
                visualizer.create_comparison_plot(
                    transfers, 
                    opt.cav_strengths,
                    content_img,
                    style_img,
                    opt.output_dir,
                    content_idx,
                    style_idx
                )
                
            except Exception as e:
                logger.error(f"Error processing {content_name} with {style_name}: {str(e)}")
                continue

if __name__ == '__main__':
    main()