import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from libs.Loader import Dataset
from libs.Matrix import MulLayer
from libs.models import encoder4, decoder4
from PIL import Image
from torchvision import transforms
from typing import List, Tuple
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vgg_dir", default='models/vgg_r41.pth', help='pre-trained encoder path')
    parser.add_argument("--decoder_dir", default='models/dec_r41.pth', help='pre-trained decoder path')
    parser.add_argument("--matrix_file", required=True, help='path to specific transformation matrix')
    parser.add_argument("--content_image", required=True, help='path to content image')
    parser.add_argument("--style_image", required=True, help='path to style image')
    parser.add_argument("--output_dir", default="noise_visualization/", help='output directory for visualizations')
    parser.add_argument("--loadSize", type=int, default=256, help='scale image size')
    parser.add_argument("--layer", default="r41", help='which features to transfer')
    parser.add_argument("--num_noise_levels", type=int, default=5, help='number of noise levels to visualize')
    parser.add_argument("--max_sigma", type=float, default=50, help='maximum noise standard deviation')
    parser.add_argument("--seed", type=int, default=42, help='random seed for reproducibility')
    return parser.parse_args()

def set_all_seeds(seed):
    """Set seeds for all random number generators."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NoiseVisualizer:
    def __init__(self, vgg, decoder, matrix, device, seed=42):
        self.vgg = vgg.to(device)
        self.decoder = decoder.to(device)
        self.matrix = matrix.to(device)
        self.device = device
        self.seed = seed
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)

    def reset_seed(self):
        """Reset the random number generator to initial seed."""
        self.rng.manual_seed(self.seed)

    @torch.no_grad()
    def generate_stylized_image(self, content: torch.Tensor, style: torch.Tensor, 
                              transform_matrix: torch.Tensor, layer: str) -> torch.Tensor:
        """Generate stylized image using the given transformation matrix."""
        # Extract features
        content_features = self.vgg(content)
        style_features = self.vgg(style)
        
        # Transform features
        cf = content_features[layer]
        sf = style_features[layer]
        
        # Get dimensions
        batch_size, channels, height, width = cf.size()
        
        # Initial feature transformation using MulLayer
        transformed_features, _ = self.matrix(cf, sf)
        
        # Reshape features for matrix multiplication
        # Instead of using all 512 channels, we'll use the matrix to transform the compressed representation
        compressed_features = self.matrix.compress(transformed_features)
        reshaped_features = compressed_features.view(batch_size, self.matrix.matrixSize, -1)
        
        # Apply transformation matrix
        # transform_matrix shape: [1, 32, 32]
        # reshaped_features shape: [batch_size, 32, height*width]
        trans_features = torch.bmm(transform_matrix.expand(batch_size, -1, -1), 
                                 reshaped_features)
        
        # Reshape back to spatial dimensions
        trans_features = trans_features.view(batch_size, self.matrix.matrixSize, height, width)
        
        # Decompress features
        trans_features = self.matrix.unzip(trans_features)
        
        # Generate stylized image
        stylized = self.decoder(trans_features)
        return torch.clamp(stylized, 0, 1)

    def add_noise_to_matrix(self, matrix: torch.Tensor, sigma: float) -> torch.Tensor:
        """Add reproducible Gaussian noise to the transformation matrix."""
        noise = torch.randn(matrix.size(), generator=self.rng, device=self.device) * sigma
        return matrix + noise

def main():
    opt = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(opt.output_dir, exist_ok=True)
    
    # Load models with weights_only=True to address the FutureWarning
    vgg = encoder4()
    decoder = decoder4()
    matrix = MulLayer(opt.layer)
    
    vgg.load_state_dict(torch.load(opt.vgg_dir, map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(opt.decoder_dir, map_location=device, weights_only=True))
    
    visualizer = NoiseVisualizer(vgg, decoder, matrix, device, seed=opt.seed)
    
    # Load images
    transform = transforms.Compose([
        transforms.Resize(opt.loadSize),
        transforms.CenterCrop(opt.loadSize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    content = transform(Image.open(opt.content_image).convert('RGB')).unsqueeze(0).to(device)
    style = transform(Image.open(opt.style_image).convert('RGB')).unsqueeze(0).to(device)
    
    # Load transformation matrix with weights_only=True
    transform_matrix = torch.load(opt.matrix_file, map_location=device, weights_only=True)
    
    # Print shapes and stats for debugging
    print(f"Transform matrix shape: {transform_matrix.shape}")
    cf = visualizer.vgg(content)[opt.layer]
    print(f"Content features shape: {cf.shape}")
    print(f"Matrix stats - min: {transform_matrix.min():.4f}, max: {transform_matrix.max():.4f}")
    
    # Generate images with different noise levels
    sigmas = np.linspace(0, opt.max_sigma, opt.num_noise_levels)
    results = []
    
    print(f"Generating stylized images with seed {opt.seed}...")
    visualizer.reset_seed()
    
    for sigma in tqdm(sigmas):
        noisy_matrix = visualizer.add_noise_to_matrix(transform_matrix, sigma)
        stylized = visualizer.generate_stylized_image(content, style, noisy_matrix, opt.layer)
        results.append((sigma, stylized))
    
    # Plot results
    fig, axes = plt.subplots(1, len(results) + 2, figsize=((len(results) + 2) * 4, 4))
    
    # Denormalize images for visualization
    denorm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    content_show = denorm(content.cpu().squeeze())
    style_show = denorm(style.cpu().squeeze())
    
    axes[0].imshow(content_show.permute(1, 2, 0).clamp(0, 1))
    axes[0].set_title('Content')
    axes[0].axis('off')
    
    axes[1].imshow(style_show.permute(1, 2, 0).clamp(0, 1))
    axes[1].set_title('Style')
    axes[1].axis('off')
    
    for idx, (sigma, img) in enumerate(results):
        img_show = denorm(img.cpu().squeeze())
        axes[idx + 2].imshow(img_show.permute(1, 2, 0).clamp(0, 1))
        axes[idx + 2].set_title(f'σ = {sigma:.3f}')
        axes[idx + 2].axis('off')
    
    output_filename = f"noise_visualization_seed{opt.seed}_{os.path.splitext(os.path.basename(opt.matrix_file))[0]}.png"
    plt.savefig(os.path.join(opt.output_dir, output_filename))
    print(f"Results saved to {opt.output_dir}/{output_filename}")

if __name__ == '__main__':
    main()