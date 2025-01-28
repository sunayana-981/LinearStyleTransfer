import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt

class DINOLocalFeatureAnalyzer:
    def __init__(self, pretrained_model: str = 'vit_base_patch16_224', use_decoder: bool = True):
        """
        Initialize the DINO local feature analyzer.
        
        Args:
            pretrained_model: Name of the pretrained ViT model to use
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model
        self.model = timm.create_model(pretrained_model, pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Get patch size
        patch_size = self.model.patch_embed.patch_size
        self.patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        
        # Store activations
        self.activations = {}
        self.hooks = []
        
        # Register hooks for early layers (focusing on local features)
        self._register_hooks()
        
        # Initialize decoder for image reconstruction if requested
        if use_decoder:
            self.decoder = nn.Sequential(
                nn.Linear(self.model.embed_dim, 3 * self.patch_size * self.patch_size),
                nn.Unflatten(-1, (3, self.patch_size, self.patch_size))
            ).to(self.device)
            
            # Initialize decoder weights using a simple statistical approach
            with torch.no_grad():
                self.decoder[0].weight.normal_(0, 0.02)
                self.decoder[0].bias.zero_()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # Patch size and dimensions
        patch_size = self.model.patch_embed.patch_size
        # Handle both tuple and integer patch sizes
        self.patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        self.grid_size = 224 // self.patch_size  # Assuming 224x224 input
        
    def _register_hooks(self):
        """Register forward hooks for the first few transformer blocks."""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output
            return hook
        
        # Register hooks for first 3 blocks (local features)
        for i in range(3):
            self.hooks.append(
                self.model.blocks[i].register_forward_hook(
                    get_activation(f'block_{i}')
                )
            )
    
    def extract_local_features(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """
        Extract local features from early transformer blocks.
        
        Args:
            image: Input PIL image
            
        Returns:
            Dictionary containing local feature maps for each early block
        """
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Forward pass
            _ = self.model(img_tensor)
            
            # Process local features from each early block
            local_features = {}
            for i in range(3):
                # Get block output
                block_output = self.activations[f'block_{i}']  # [1, num_patches+1, hidden_dim]
                
                # Remove CLS token and reshape to spatial grid
                patch_features = block_output[:, 1:, :]  # [1, num_patches, hidden_dim]
                spatial_features = patch_features.reshape(
                    1, self.grid_size, self.grid_size, -1
                )  # [1, H, W, C]
                
                # Store features
                local_features[f'block_{i}'] = spatial_features
                
            return local_features
    
    def amplify_local_patterns(
        self, 
        features: Dict[str, torch.Tensor],
        amplification_factor: float = 2.0,
        method: str = 'contrast'
    ) -> Dict[str, torch.Tensor]:
        """
        Amplify local patterns in the feature maps.
        
        Args:
            features: Dictionary of local features from each block
            amplification_factor: Factor to amplify features by
            method: Amplification method ('contrast' or 'magnitude')
            
        Returns:
            Dictionary containing amplified feature maps
        """
        amplified_features = {}
        
        for block_name, feature_map in features.items():
            if method == 'contrast':
                # Increase contrast by centering and scaling
                mean = feature_map.mean(dim=-1, keepdim=True)
                std = feature_map.std(dim=-1, keepdim=True)
                normalized = (feature_map - mean) / (std + 1e-6)
                amplified = normalized * amplification_factor
                
            elif method == 'magnitude':
                # Direct magnitude amplification
                magnitude = torch.norm(feature_map, dim=-1, keepdim=True)
                direction = feature_map / (magnitude + 1e-6)
                amplified = direction * (magnitude ** amplification_factor)
            
            else:
                raise ValueError(f"Unknown amplification method: {method}")
            
            amplified_features[block_name] = amplified
            
        return amplified_features
    
    def compute_local_attention(
        self,
        features: Dict[str, torch.Tensor],
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute attention maps highlighting local patterns.
        
        Args:
            features: Dictionary of feature maps
            temperature: Temperature for softmax attention
            
        Returns:
            Dictionary containing attention maps for each block
        """
        attention_maps = {}
        
        for block_name, feature_map in features.items():
            # Compute self-attention between local patches
            Q = feature_map.reshape(1, -1, feature_map.shape[-1])  # [1, HW, C]
            K = Q  # Use same features for keys
            
            # Compute attention scores
            attention = torch.matmul(Q, K.transpose(-2, -1))  # [1, HW, HW]
            attention = attention / (feature_map.shape[-1] ** 0.5)  # Scale by dimension
            attention = F.softmax(attention / temperature, dim=-1)
            
            # Reshape to spatial grid
            attention_spatial = attention.reshape(
                1, self.grid_size, self.grid_size,
                self.grid_size, self.grid_size
            )
            
            attention_maps[block_name] = attention_spatial
            
        return attention_maps
    
    def reconstruct_image(self, features: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct image from features using the improved decoder.
        
        Args:
            features: Feature tensor of shape [1, H, W, C]
            
        Returns:
            Reconstructed image tensor of shape [1, 3, H*patch_size, W*patch_size]
        """
        B, H, W, C = features.shape
        
        # Normalize features before decoding
        features_flat = features.reshape(B * H * W, C)
        features_norm = F.layer_norm(features_flat, (C,))
        
        # Apply decoder
        patches = self.decoder(features_norm)  # [B*H*W, 3, patch_size, patch_size]
        
        # Reshape into image
        patches = patches.reshape(B, H, W, 3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        image = patches.reshape(B, 3, H * self.patch_size, W * self.patch_size)
        
        return image

    def save_reconstructed_images(
        self,
        original_recon: torch.Tensor,
        amplified_recon: torch.Tensor,
        save_dir: str
    ):
        """
        Save reconstructed images.
        
        Args:
            original_recon: Original reconstructed image tensor
            amplified_recon: Amplified reconstructed image tensor
            save_dir: Directory to save images
        """
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert tensors to images and save
        to_pil = transforms.ToPILImage()
        
        original_img = to_pil(original_recon[0].cpu().clamp(0, 1))
        amplified_img = to_pil(amplified_recon[0].cpu().clamp(0, 1))
        
        original_img.save(os.path.join(save_dir, 'original_reconstruction.png'))
        amplified_img.save(os.path.join(save_dir, 'amplified_reconstruction.png'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple

class DINOLocalFeatureAnalyzer:
    # ... [previous initialization code remains the same until visualize_local_patterns] ...

    def visualize_local_patterns(
        self,
        image: Image.Image,
        save_path: Optional[str] = None,
        save_dir: Optional[str] = None
    ):
        """
        Extract, amplify, and visualize local patterns in the image.
        
        Args:
            image: Input PIL image
            save_path: Optional path to save visualization
            save_dir: Optional directory to save reconstructed images
        """
        # Extract features
        local_features = self.extract_local_features(image)
        
        # Amplify features
        amplified_features = self.amplify_local_patterns(
            local_features,
            amplification_factor=2.0
        )
        
        # Compute attention maps
        attention_maps = self.compute_local_attention(local_features)
        
        # Reconstruct images
        original_recon = self.reconstruct_image(local_features['block_0'])
        amplified_recon = self.reconstruct_image(amplified_features['block_0'])
        
        # Save reconstructed images if directory is provided
        if save_dir:
            self.save_reconstructed_images(original_recon, amplified_recon, save_dir)
        
        # Create visualization with proper figure size and DPI
        num_blocks = len(local_features)
        plt.figure(figsize=(20, 20), dpi=100)
        fig, axes = plt.subplots(4, num_blocks + 1, figsize=(20, 20))
        
        # Original image and reconstructions
        # Convert PIL image to numpy array for proper display
        img_array = np.array(image)
        axes[0, 0].imshow(img_array)
        axes[0, 0].set_title('Original Image', fontsize=12)
        axes[0, 0].axis('off')
        
        # Ensure proper normalization for reconstructions
        orig_recon_np = original_recon[0].permute(1, 2, 0).detach().cpu().numpy()
        orig_recon_np = np.clip((orig_recon_np - orig_recon_np.min()) / 
                               (orig_recon_np.max() - orig_recon_np.min()), 0, 1)
        axes[1, 0].imshow(orig_recon_np)
        axes[1, 0].set_title('Reconstructed', fontsize=12)
        axes[1, 0].axis('off')
        
        amp_recon_np = amplified_recon[0].permute(1, 2, 0).detach().cpu().numpy()
        amp_recon_np = np.clip((amp_recon_np - amp_recon_np.min()) / 
                              (amp_recon_np.max() - amp_recon_np.min()), 0, 1)
        axes[2, 0].imshow(amp_recon_np)
        axes[2, 0].set_title('Amplified', fontsize=12)
        axes[2, 0].axis('off')
        
        # Feature visualizations for each block
        for i, block_name in enumerate(local_features.keys()):
            # Feature magnitude with proper normalization
            feat_mag = torch.norm(local_features[block_name], dim=-1)[0]
            feat_mag = feat_mag.detach().cpu().numpy()
            feat_mag = (feat_mag - feat_mag.min()) / (feat_mag.max() - feat_mag.min())
            
            im = axes[0, i+1].imshow(feat_mag, cmap='viridis')
            axes[0, i+1].set_title(f'{block_name} Features', fontsize=12)
            axes[0, i+1].axis('off')
            plt.colorbar(im, ax=axes[0, i+1])
            
            # Original reconstruction features
            recon = self.reconstruct_image(local_features[block_name])
            recon_np = recon[0].permute(1, 2, 0).detach().cpu().numpy()
            recon_np = np.clip((recon_np - recon_np.min()) / 
                              (recon_np.max() - recon_np.min()), 0, 1)
            axes[1, i+1].imshow(recon_np)
            axes[1, i+1].set_title(f'{block_name} Reconstructed', fontsize=12)
            axes[1, i+1].axis('off')
            
            # Amplified features
            amp_mag = torch.norm(amplified_features[block_name], dim=-1)[0]
            amp_mag = amp_mag.detach().cpu().numpy()
            amp_mag = (amp_mag - amp_mag.min()) / (amp_mag.max() - amp_mag.min())
            
            im = axes[2, i+1].imshow(amp_mag, cmap='viridis')
            axes[2, i+1].set_title(f'{block_name} Amplified Features', fontsize=12)
            axes[2, i+1].axis('off')
            plt.colorbar(im, ax=axes[2, i+1])
            
            # Attention visualization with proper normalization
            att_map = attention_maps[block_name][0].mean(dim=(2, 3))
            att_map = att_map.cpu().numpy()
            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())
            
            im = axes[3, i+1].imshow(att_map, cmap='viridis')
            axes[3, i+1].set_title(f'{block_name} Attention', fontsize=12)
            axes[3, i+1].axis('off')
            plt.colorbar(im, ax=axes[3, i+1])
        
        axes[3, 0].axis('off')
        
        # Adjust layout and spacing
        plt.tight_layout(pad=3.0)
        
        # Save or display with high quality
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300, format='png')
            plt.close()
        else:
            plt.show()

    def __del__(self):
        """Clean up hooks when object is deleted."""
        for hook in self.hooks:
            hook.remove()


# Example usage
if __name__ == "__main__":
    import argparse
    import os

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze local features in an image using DINO')
    parser.add_argument('--image_path', type=str, default='data/style/01.jpg',
                      help='Path to the input image')
    parser.add_argument('--output_dir', type=str, default='output_DINO/',
                      help='Directory to save outputs')
    args = parser.parse_args()

    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Check if image exists
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"Image file not found: {args.image_path}")

        # Initialize analyzer
        print("Initializing DINO feature analyzer...")
        analyzer = DINOLocalFeatureAnalyzer()
        
        # Load and analyze image
        print(f"Loading image from {args.image_path}")
        image = Image.open(args.image_path).convert('RGB')
        
        # Print debug information
        print(f"Image size: {image.size}")
        print(f"Patch size: {analyzer.patch_size}")
        print(f"Grid size: {analyzer.grid_size}")
        
        # Extract and visualize local patterns
        print("Analyzing local patterns...")
        analyzer.visualize_local_patterns(
            image,
            save_path=args.output_dir + 'local_patterns.png',
        )
        print(f"Visualization saved to {args.output_dir}local_patterns.png")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise e
    
