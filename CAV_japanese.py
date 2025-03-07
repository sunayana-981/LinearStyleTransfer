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

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])

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

    def apply_class_direction(self, content_img, style_img, direction, strength=1.0):
        """Apply learned class direction during style transfer"""
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
    
class CAVAnalyzer:
    def __init__(self, style_controller):
        self.style_controller = style_controller
        self.device = next(style_controller.decoder.parameters()).device
    
    def analyze_cav_components(self, cav, n_components=10):
        """Analyze the most significant components of the CAV"""
        # Flatten CAV for analysis
        flat_cav = cav.view(-1)
        values, indices = torch.sort(torch.abs(flat_cav), descending=True)
        
        # Get top N components
        top_values = values[:n_components]
        top_indices = indices[:n_components]
        original_values = flat_cav[top_indices]
        
        # Get spatial locations
        spatial_locations = []
        for idx in top_indices:
            # Convert flat index to feature map coordinates
            c = (idx // (cav.shape[1] * cav.shape[2])).item()
            h = ((idx % (cav.shape[1] * cav.shape[2])) // cav.shape[2]).item()
            w = (idx % cav.shape[2]).item()
            spatial_locations.append((c, h, w))
        
        return {
            'magnitudes': top_values.cpu().numpy(),
            'values': original_values.cpu().numpy(),
            'locations': spatial_locations
        }
    
    def measure_style_impact(self, content_img, style_img, cav, feature_idx, strength_range=(-2, 2, 5)):
        """Measure impact of specific CAV feature on style transfer"""
        strengths = np.linspace(*strength_range)
        impacts = []

        with torch.no_grad():
            base_features = self.style_controller.get_style_transfer_features(content_img, style_img)
            # Ensure cav and mask are of the same dtype as base_features
            cav = cav.to(base_features.dtype)
            mask = torch.zeros_like(cav, dtype=base_features.dtype)
            c, h, w = feature_idx
            mask[c, h, w] = 1.0

            for strength in strengths:
                modified_features = base_features + strength * (cav * mask).unsqueeze(0)
                result = self.style_controller.decoder(modified_features)
                diff = torch.norm(result - self.style_controller.decoder(base_features)).item()
                impacts.append(diff)

        return strengths, impacts
  
    def get_channel_activations(self, content_img, style_img, cav):
        """Analyze per-channel activation statistics"""
        with torch.no_grad():
            features = self.style_controller.get_style_transfer_features(content_img, style_img)
            
            # Get channel-wise statistics
            channel_stats = []
            for c in range(cav.shape[1]):
                # Make tensors contiguous and flatten them
                channel_cav = cav[:, c:c+1].contiguous().flatten()
                channel_feat = features[:, c:c+1].contiguous().flatten()
                
                # Calculate correlation
                try:
                    corr = torch.corrcoef(torch.stack([channel_cav, channel_feat]))[0, 1].item()
                except RuntimeError:
                    # Handle case where correlation can't be computed
                    corr = float('nan')
                
                # Calculate impact measures
                magnitude = torch.norm(channel_cav).item()
                activation = torch.norm(channel_feat).item()
                
                channel_stats.append({
                    'channel': c,
                    'correlation': corr,
                    'cav_magnitude': magnitude,
                    'feature_activation': activation,
                    'mean_cav': channel_cav.mean().item(),
                    'std_cav': channel_cav.std().item()
                })
                
                # Free memory
                del channel_cav, channel_feat
                torch.cuda.empty_cache()
        
        return channel_stats
    
    def visualize_feature_importance(self, channel_stats, save_path=None):
        """Visualize channel-wise feature importance"""
        figure_dir = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        
        # 1. Basic Statistics Plot
        plt.figure(figsize=(15, 5))
        
        # Plot correlations
        plt.subplot(131)
        correlations = [stat['correlation'] for stat in channel_stats if not np.isnan(stat['correlation'])]
        plt.hist(correlations, bins=30)
        plt.title('CAV-Feature Correlations')
        plt.xlabel('Correlation')
        plt.ylabel('Count')
        
        # Plot magnitudes
        plt.subplot(132)
        magnitudes = [stat['cav_magnitude'] for stat in channel_stats]
        plt.hist(magnitudes, bins=30)
        plt.title('CAV Channel Magnitudes')
        plt.xlabel('Magnitude')
        plt.ylabel('Count')
        
        # Plot activation distribution
        plt.subplot(133)
        activations = [stat['feature_activation'] for stat in channel_stats]
        plt.hist(activations, bins=30)
        plt.title('Feature Activations')
        plt.xlabel('Activation')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, f'{base_name}_basic.png'))
        plt.close()
        
        # 2. Channel-wise Statistics Plot
        plt.figure(figsize=(15, 8))
        
        channels = range(len(channel_stats))
        mean_values = [stat['mean_cav'] for stat in channel_stats]
        std_values = [stat['std_cav'] for stat in channel_stats]
        
        plt.subplot(211)
        plt.bar(channels, mean_values)
        plt.title('Channel-wise Mean CAV Values')
        plt.xlabel('Channel')
        plt.ylabel('Mean Value')
        
        plt.subplot(212)
        plt.bar(channels, std_values)
        plt.title('Channel-wise Standard Deviation')
        plt.xlabel('Channel')
        plt.ylabel('Standard Deviation')
        
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, f'{base_name}_channels.png'))
        plt.close()
        
        # 3. Correlation Analysis Plot
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot of magnitude vs activation
        magnitudes = [stat['cav_magnitude'] for stat in channel_stats]
        activations = [stat['feature_activation'] for stat in channel_stats]
        
        plt.scatter(magnitudes, activations, alpha=0.5)
        plt.title('CAV Magnitude vs Feature Activation')
        plt.xlabel('CAV Magnitude')
        plt.ylabel('Feature Activation')
        
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, f'{base_name}_correlation.png'))
        plt.close()
    
    def analyze_spatial_patterns(self, cav):
        """Analyze spatial patterns in the CAV"""
        # Calculate spatial importance maps
        spatial_importance = torch.norm(cav, dim=1).squeeze()
        
        # Get regions of high activation
        threshold = spatial_importance.mean() + spatial_importance.std()
        important_regions = (spatial_importance > threshold).float()
        
        # Calculate spatial statistics
        stats = {
            'center_bias': self._calculate_center_bias(spatial_importance),
            'important_region_ratio': important_regions.mean().item(),
            'spatial_variance': spatial_importance.var().item()
        }
        
        return stats, spatial_importance
    
    def _calculate_center_bias(self, spatial_map):
        device = spatial_map.device
        h, w = spatial_map.shape
        y = torch.linspace(-1, 1, h, device=device).view(-1, 1).expand(-1, w)
        x = torch.linspace(-1, 1, w, device=device).view(1, -1).expand(h, -1)

        # Calculate distance from center
        dist_from_center = torch.sqrt(x**2 + y**2)

        # Calculate correlation with distance
        flat_map = spatial_map.view(-1)
        flat_dist = dist_from_center.view(-1)

        return torch.corrcoef(torch.stack([flat_map, flat_dist]))[0, 1].item()

def analyze_cav(cav_analyzer, content_img, style_img, cav, output_dir):
    """Run comprehensive CAV analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Analyze top components
    top_components = cav_analyzer.analyze_cav_components(cav)
    
    # Save component analysis
    with open(os.path.join(output_dir, 'top_components.txt'), 'w') as f:
        f.write('Top CAV Components:\n')
        for i, (mag, val, loc) in enumerate(zip(
            top_components['magnitudes'],
            top_components['values'],
            top_components['locations']
        )):
            f.write(f"{i+1}. Magnitude: {mag:.4f}, Value: {val:.4f}, Location: {loc}\n")
    
    # 2. Analyze channel statistics
    channel_stats = cav_analyzer.get_channel_activations(content_img, style_img, cav)
    
    # Visualize channel statistics
    cav_analyzer.visualize_feature_importance(
        channel_stats,
        save_path=os.path.join(output_dir, 'channel_analysis.png')
    )
    
    # 3. Analyze spatial patterns
    spatial_stats, importance_map = cav_analyzer.analyze_spatial_patterns(cav)
    
    # Save spatial analysis
    with open(os.path.join(output_dir, 'spatial_analysis.txt'), 'w') as f:
        f.write('Spatial Pattern Analysis:\n')
        for key, value in spatial_stats.items():
            f.write(f"{key}: {value:.4f}\n")
    
    # Visualize spatial importance
    plt.figure(figsize=(8, 8))
    plt.imshow(importance_map.cpu(), cmap='viridis')
    plt.colorbar(label='Importance')
    plt.title('CAV Spatial Importance Map')
    plt.savefig(os.path.join(output_dir, 'spatial_importance.png'))
    plt.close()
    
    # 4. Measure feature impact for top components
    for i, loc in enumerate(top_components['locations'][:3]):  # Test top 3 components
        strengths, impacts = cav_analyzer.measure_style_impact(
            content_img, style_img, cav, loc
        )
        
        plt.figure(figsize=(8, 4))
        plt.plot(strengths, impacts)
        plt.title(f'Impact of Component {i+1} at {loc}')
        plt.xlabel('Strength')
        plt.ylabel('Change Magnitude')
        plt.savefig(os.path.join(output_dir, f'component_{i+1}_impact.png'))
        plt.close()

class CAVBasisAnalyzer:
    def __init__(self, style_controller):
        self.style_controller = style_controller
        self.device = next(style_controller.decoder.parameters()).device

    def find_basis_vectors(self, cavs, n_components=5):
        """Find basis vectors that can represent multiple CAVs"""
        # Stack and reshape CAVs for analysis
        cav_tensors = []
        for style_name, cav in cavs.items():
            flat_cav = cav.cpu().view(-1).numpy()
            cav_tensors.append(flat_cav)
        
        cav_matrix = np.stack(cav_tensors)
        
        # Perform PCA to find principal components
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(n_components, len(cav_tensors)))
        pca_result = pca.fit_transform(cav_matrix)
        
        # Reshape components back to CAV shape
        original_shape = next(iter(cavs.values())).shape
        basis_vectors = []
        for component in pca.components_:
            dtype = next(self.style_controller.decoder.parameters()).dtype
            basis_vector = torch.tensor(component.reshape(original_shape), device=self.device, dtype=dtype)
            basis_vectors.append(basis_vector)
        
        # Calculate reconstruction quality
        reconstruction_errors = {}
        for style_name, original_cav in cavs.items():
            reconstructed_cav = self.reconstruct_cav(original_cav, basis_vectors)
            error = torch.norm(original_cav - reconstructed_cav).item()
            reconstruction_errors[style_name] = error
        
        return {
            'basis_vectors': basis_vectors,
            'explained_variance': pca.explained_variance_ratio_,
            'reconstruction_errors': reconstruction_errors
        }

    def reconstruct_cav(self, original_cav, basis_vectors):
        """Reconstruct CAV from basis vectors"""
        reconstruction = torch.zeros_like(original_cav)
        original_flat = original_cav.cpu().view(-1)
        
        for basis_vector in basis_vectors:
            basis_flat = basis_vector.cpu().view(-1)
            # Project onto basis vector
            projection = torch.dot(original_flat, basis_flat) / torch.dot(basis_flat, basis_flat)
            reconstruction += projection * basis_vector
        
        return reconstruction

    def analyze_basis_components(self, basis_result, output_dir):
        """Analyze and visualize basis vectors"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot explained variance
        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(basis_result['explained_variance']), marker='o')
        plt.title('Cumulative Explained Variance by Basis Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'explained_variance.png'))
        plt.close()
        
        # Visualize basis vectors
        n_basis = len(basis_result['basis_vectors'])
        plt.figure(figsize=(4*n_basis, 4))
        for idx, basis_vector in enumerate(basis_result['basis_vectors']):
            plt.subplot(1, n_basis, idx+1)
            spatial_importance = torch.norm(basis_vector, dim=1).squeeze().cpu()
            plt.imshow(spatial_importance, cmap='viridis')
            plt.title(f'Basis {idx+1}\nVar: {basis_result["explained_variance"][idx]:.3f}')
            plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'basis_vectors.png'))
        plt.close()
        
        # Save analysis results
        with open(os.path.join(output_dir, 'basis_analysis.txt'), 'w') as f:
            f.write("Basis Vector Analysis:\n\n")
            f.write("Explained Variance by Component:\n")
            for idx, var in enumerate(basis_result['explained_variance']):
                f.write(f"Basis {idx+1}: {var:.4f}\n")
            
            f.write("\nReconstruction Errors:\n")
            for style, error in basis_result['reconstruction_errors'].items():
                f.write(f"{style}: {error:.4f}\n")

    def interpolate_styles(self, content_img, style_img, basis_vectors, coefficients):
        """Create new style transfer using basis vector combination"""
        combined_cav = torch.zeros_like(basis_vectors[0])
        for basis_vector, coeff in zip(basis_vectors, coefficients):
            combined_cav += coeff * basis_vector
        
        # Apply combined CAV
        result, _ = self.style_controller.apply_class_direction(
            content_img, style_img, combined_cav, strength=1.0
        )
        return result

def analyze_style_basis(content_img, cavs, style_controller, output_dir, n_components=5):
    """Main function to find and analyze basis vectors"""
    analyzer = CAVBasisAnalyzer(style_controller)
    
    # Find basis vectors
    basis_result = analyzer.find_basis_vectors(cavs, n_components)
    
    # Analyze and visualize results
    analyzer.analyze_basis_components(basis_result, output_dir)
    
    return basis_result



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
    parser.add_argument("--outf", default="results/style_class_cav1/",
                      help='path to output images')
    parser.add_argument("--batch_size", type=int, default=10,
                      help='batch size for processing images')
    parser.add_argument("--num_images", type=int, default=100,
                      help='number of images to use from each class')
    parser.add_argument("--strengths", type=float, nargs='+',
                      default=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    
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
            transfer, _ = style_controller.apply_class_direction(
                content_img,
                base_style,
                class_direction,
                strength
            )
            transfers.append(transfer)
            
            # Save individual result
            output_filename = f'content_base_style_{style_idx}_strength_{strength:.1f}.png'
            output_path = os.path.join(opt.outf, output_filename)
            vutils.save_image(transfer, output_path)
        
        # Create comparison plot
        target_class_name = os.path.basename(opt.target_style_class.rstrip('/'))
        visualizer.create_comparison_plot(
            transfers,
            opt.strengths,
            content_img,
            base_style,
            comparison_dir,
            'base',
            f'{target_class_name}_{style_idx}'
        )

            # After learning class direction:
        # analyzer = CAVAnalyzer(style_controller)
        # analyze_cav(
        #     analyzer,
        #     content_img,
        #     target_styles[0],  # Use first style image as reference
        #     class_direction,
        #     os.path.join(opt.outf, 'cav_analysis')
        # )

    # def analyze_style_basis(content_img, cavs, style_controller, output_dir, n_components=5):
    #     """Main function to find and analyze basis vectors"""
    #     analyzer = CAVBasisAnalyzer(style_controller)
        
    #     # Find basis vectors
    #     basis_result = analyzer.find_basis_vectors(cavs, n_components)
        
    #     # Analyze and visualize results
    #     analyzer.analyze_basis_components(basis_result, output_dir)
        
    #     return basis_result

    # basis_result = analyze_style_basis(
    #     content_img,
    #     cavs,  # Your learned CAVs
    #     style_controller,
    #     os.path.join(opt.outf, 'basis_analysis'),
    #     n_components=5
    # )

    # # Example of using basis vectors for interpolation
    # coefficients = [0.5, 0.3, 0.2, 0.0, 0.0]  # Example weights
    # interpolated = analyzer.interpolate_styles(
    #     content_img,
    #     style_classes['Realism'][0],  # Use first realism image as base
    #     basis_result['basis_vectors'],
    #     coefficients
    # )

if __name__ == "__main__":
    main()