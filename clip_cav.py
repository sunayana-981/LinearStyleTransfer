import torch
import clip
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.svm import LinearSVC
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeightedStyleCAVController:
    def __init__(self, vgg, matrix, decoder, clip_model, layer_name='r41', device='cuda'):
        self.vgg = vgg
        self.matrix = matrix
        self.decoder = decoder
        self.clip_model = clip_model
        self.layer_name = layer_name
        self.device = device
        
        # Progress bar for initialization
        with tqdm(total=1, desc="Initializing controller", position=0) as pbar:
            # Define weighted concepts
            self.weighted_concepts = {
                'color': [
                    ("an image with muted colors", 0.5305),
                    ("an image with warm colors", 0.3250),
                    ("an image with vibrant colors", 0.1445)
                ],
                'texture': [
                    ("an image with rough texture", 0.5141),
                    ("an image with regular patterns", 0.3025),
                    ("an image with irregular patterns", 0.1834)
                ],
                'composition': [
                    ("an image with complex composition", 0.5463),
                    ("an image with balanced composition", 0.3263),
                    ("an image with dynamic composition", 0.1274)
                ]
            }
            
            self.activations = []
            self._register_hooks()
            pbar.update(1)
    
    def _register_hooks(self):
        """Register hooks to capture feature activations."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.activations.append(output[0].detach())
            else:
                self.activations.append(output.detach())
        
        self.matrix.register_forward_hook(hook_fn)

    def get_style_features(self, image_path: str) -> torch.Tensor:
        """Extract features using style transfer model."""
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
                content_features = features[self.layer_name]
                style_features = features[self.layer_name]
                
                transformed_features, _ = self.matrix(content_features, style_features)
            
            if not self.activations:
                raise ValueError("No activations captured during forward pass")
            
            return self.activations[-1]
        
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise

    def learn_category_cav(self, image_dir: Path, category: str) -> torch.Tensor:
        """Learn CAV for a specific concept category using style transfer features."""
        # Get concept pairs
        concepts = self.weighted_concepts[category]
        
        # Split concepts based on weights
        with tqdm(total=1, desc=f"Preparing {category} concepts", position=1, leave=False) as pbar:
            positive_concepts = []
            negative_concepts = []
            median_weight = np.median([weight for _, weight in concepts])
            
            for concept, weight in concepts:
                if weight > median_weight:
                    positive_concepts.append(concept)
                else:
                    negative_concepts.append(concept)
            pbar.update(1)
        
        # Collect image paths
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(list(image_dir.glob(ext)))
        
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        # Extract features with progress bar
        features_list = []
        labels = []
        
        for path in tqdm(image_paths, 
                        desc=f"Processing {category} images",
                        position=1,
                        leave=False):
            try:
                features = self.get_style_features(str(path))
                flat_features = features.view(features.size(0), -1).cpu().numpy()
                features_list.append(flat_features)
                
                # TODO: Implement proper concept matching
                label = np.random.choice([0, 1])
                labels.extend([label] * flat_features.shape[0])
                
            except Exception as e:
                logger.warning(f"Error processing {path}: {str(e)}")
                continue
        
        if not features_list:
            raise ValueError("No valid features extracted")
        
        # Train SVM with progress
        with tqdm(total=1, desc=f"Training {category} SVM", position=1, leave=False) as pbar:
            X = np.vstack(features_list)
            y = np.array(labels)
            
            svm = LinearSVC(C=1.0, dual="auto")
            svm.fit(X, y)
            
            cav = torch.tensor(svm.coef_[0]).reshape(features.shape[1:]).to(self.device)
            pbar.update(1)
        
        return cav

    def apply_weighted_cav(self, 
                          content_image: torch.Tensor,
                          style_image: torch.Tensor,
                          cavs: Dict[str, torch.Tensor],
                          category_weights: Dict[str, float] = None,
                          base_strength: float = 1.0):
        """Apply learned weighted CAVs with progress tracking."""
        if category_weights is None:
            category_weights = {cat: 1.0 for cat in self.weighted_concepts.keys()}
        
        with torch.no_grad():
            with tqdm(total=3, desc="Applying CAVs", position=1, leave=False) as pbar:
                # Get initial features
                content_features = self.vgg(content_image)[self.layer_name]
                style_features = self.vgg(style_image)[self.layer_name]
                pbar.update(1)
                
                # Get transformed features
                transformed_features, matrix = self.matrix(content_features, style_features)
                transformed_features = transformed_features.float()
                pbar.update(1)
                
                # Apply weighted CAVs
                modified_features = transformed_features.clone()
                feature_norm = torch.norm(transformed_features)
                
                for category, cav in cavs.items():
                    if category in category_weights:
                        weight = category_weights[category]
                        logger.info(f"Applying {category} CAV with weight {weight}")
                        
                        cav = cav.unsqueeze(0)
                        scale_factor = feature_norm / torch.norm(cav)
                        strength = base_strength * weight * scale_factor * 0.1
                        
                        modified_features += (strength * cav).float()
                
                # Generate final image
                result = self.decoder(modified_features)
                pbar.update(1)
                
                return result.clamp(0, 1), matrix

def create_comparison_plot(variations, strengths, content_img, style_img, output_path):
    """Create comparison visualization."""
    num_variations = len(variations)
    fig, axes = plt.subplots(1, num_variations + 2, figsize=(4 * (num_variations + 2), 4))
    
    def tensor_to_image(tensor):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        return tensor.cpu().permute(1, 2, 0).numpy()
    
    # Plot original images
    axes[0].imshow(tensor_to_image(content_img))
    axes[0].set_title('Content')
    axes[0].axis('off')
    
    axes[1].imshow(tensor_to_image(style_img))
    axes[1].set_title('Style')
    axes[1].axis('off')
    
    # Plot variations
    for idx, (variation, strength) in enumerate(zip(variations, strengths)):
        axes[idx + 2].imshow(tensor_to_image(variation))
        axes[idx + 2].set_title(f'Strength: {strength:.1f}')
        axes[idx + 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    import argparse
    from libs.models import encoder4, decoder4
    from libs.Matrix import MulLayer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", type=str, required=True)
    parser.add_argument("--style", type=str, required=True)
    parser.add_argument("--style_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--vgg_model", default="models/vgg_r41.pth")
    parser.add_argument("--decoder_model", default="models/dec_r41.pth")
    parser.add_argument("--matrix_model", default="models/r41.pth")
    
    args = parser.parse_args()
    
    # Initialize with progress
    with tqdm(total=4, desc="Setting up models", position=0) as pbar:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vgg = encoder4().to(device)
        pbar.update(1)
        
        dec = decoder4().to(device)
        pbar.update(1)
        
        matrix = MulLayer('r41').to(device)
        pbar.update(1)
        
        clip_model, _ = clip.load("ViT-B/32", device=device)
        pbar.update(1)
    
    # Load weights with progress
    with tqdm(total=3, desc="Loading model weights", position=0) as pbar:
        vgg.load_state_dict(torch.load(args.vgg_model))
        pbar.update(1)
        
        dec.load_state_dict(torch.load(args.decoder_model))
        pbar.update(1)
        
        matrix.load_state_dict(torch.load(args.matrix_model))
        pbar.update(1)
    
    # Process images
    controller = WeightedStyleCAVController(vgg, matrix, dec, clip_model)
    style_dir = Path(args.style_dir)
    
    # Learn CAVs for each category
    cavs = {}
    for category in tqdm(controller.weighted_concepts.keys(), 
                        desc="Learning CAVs for categories",
                        position=0):
        cavs[category] = controller.learn_category_cav(style_dir, category)
    
    # Load images with progress
    with tqdm(total=2, desc="Loading images", position=0) as pbar:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor()
        ])
        
        content_img = transform(Image.open(args.content).convert('RGB')).unsqueeze(0)
        pbar.update(1)
        
        style_img = transform(Image.open(args.style).convert('RGB')).unsqueeze(0)
        pbar.update(1)
    
    if torch.cuda.is_available():
        content_img = content_img.cuda()
        style_img = style_img.cuda()
    
    # Create variations with progress
    strengths = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
    variations = []
    for strength in tqdm(strengths, 
                        desc="Generating variations",
                        position=0):
        result, _ = controller.apply_weighted_cav(
            content_img,
            style_img,
            cavs,
            base_strength=strength
        )
        variations.append(result)
    
    # Create visualization
    with tqdm(total=1, desc="Creating visualization", position=0) as pbar:
        create_comparison_plot(
            variations,
            strengths,
            content_img,
            style_img,
            args.output
        )
        pbar.update(1)