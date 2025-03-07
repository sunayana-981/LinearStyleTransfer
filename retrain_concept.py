import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from pathlib import Path
import wandb  # For experiment tracking
from sklearn.svm import LinearSVC

# Increase PIL's DecompressionBomb limit
Image.MAX_IMAGE_PIXELS = None  # or set to a higher value like 200000000
# Import style transfer components
from libs.models import encoder4, decoder4
from libs.Matrix import MulLayer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("wikiart_cav_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def sanitize_filename(filename):
    """Remove invalid characters from filenames for Windows compatibility."""
    # Replace problematic characters with safe alternatives
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        filename = filename.replace(char, '-')
    return filename

class ConceptAwareStyleTransfer(nn.Module):
    """
    Enhanced style transfer network with concept-aware feature transformation
    that works across the WikiArt dataset.
    """
    def __init__(self, encoder, decoder, matrix, num_concepts=3, layer='r41', device='cuda'):
        super(ConceptAwareStyleTransfer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.matrix = matrix
        self.layer = layer
        self.device = device
        self.num_concepts = num_concepts
        
        # Initialize feature dimensions
        self._initialize_feature_dimensions()
        
        # Create learnable concept directions
        self.concept_directions = nn.ParameterList([
            nn.Parameter(torch.randn(self.feature_dim, *self.spatial_dims).to(device))
            for _ in range(num_concepts)
        ])
        
        # Initialize concept names
        self.concept_names = ["edges", "sharpness", "brightness"]
        
        # Register hooks to capture activations for SVM-based CAV learning
        self.activations = []
        self._register_hooks()
        
        # Normalize concept directions
        self._normalize_concept_directions()
    
    def _initialize_feature_dimensions(self):
        """Get feature dimensions using a dummy forward pass."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256).to(self.device)
            features = self.encoder(dummy_input)
            self.feature_dim = features[self.layer].shape[1]
            self.spatial_dims = features[self.layer].shape[2:]
            logger.info(f"Feature dimensions: {self.feature_dim}x{self.spatial_dims}")
    
    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.activations.append(output[0].detach())
            else:
                self.activations.append(output.detach())
        
        self.matrix.register_forward_hook(hook_fn)
    
    def _normalize_concept_directions(self):
        """Normalize concept directions to unit norm."""
        with torch.no_grad():
            for i in range(self.num_concepts):
                flat_direction = self.concept_directions[i].view(-1)
                norm = torch.norm(flat_direction)
                if norm > 0:
                    self.concept_directions[i].data = self.concept_directions[i].data / norm
                    logger.debug(f"Normalized concept {i} ({self.concept_names[i]}) to unit norm")
    
    def extract_features(self, image_paths, batch_size=8):
        """
        Extract features for a list of images.
        
        Args:
            image_paths: List of paths to images
            batch_size: Batch size for processing
            
        Returns:
            numpy.ndarray of features
        """
        # Create dataset transform
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor()
        ])
        
        features_list = []
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(self.device)
                    batch_images.append(img_tensor)
                except Exception as e:
                    logger.error(f"Error loading image {path}: {str(e)}")
                    continue
            
            if not batch_images:
                continue
                
            # Stack images into a batch
            batch_tensor = torch.cat(batch_images, dim=0)
            
            # Extract features
            self.activations = []
            with torch.no_grad():
                content_features = self.encoder(batch_tensor)[self.layer]
                style_features = self.encoder(batch_tensor)[self.layer]
                transformed_features, _ = self.matrix(content_features, style_features)
            
            if not self.activations:
                raise ValueError("No activations captured during forward pass")
            
            # Get features from hooks and flatten
            batch_features = self.activations[-1].view(len(batch_images), -1).cpu().numpy()
            features_list.append(batch_features)
        
        if not features_list:
            raise ValueError("No features could be extracted from images")
            
        # Concatenate all features
        return np.concatenate(features_list, axis=0)
    
    def learn_cav_with_svm(self, concept_image_paths, random_image_paths, concept_idx=0):
        """
        Learn a CAV using Linear SVM on concept vs. random images.
        
        Args:
            concept_image_paths: Paths to concept images
            random_image_paths: Paths to random/negative images
            concept_idx: Index to assign the learned CAV to
            
        Returns:
            The learned CAV
        """
        logger.info(f"Learning CAV from {len(concept_image_paths)} concept images and {len(random_image_paths)} random images")
        
        # Extract features for concept and random images
        concept_features = self.extract_features(concept_image_paths)
        random_features = self.extract_features(random_image_paths)
        
        logger.info(f"Extracted features of shape: {concept_features.shape}")
        
        # Prepare data for SVM
        X = np.concatenate([concept_features, random_features], axis=0)
        y = np.concatenate([np.ones(len(concept_features)), 
                           np.zeros(len(random_features))])
        
        # Train SVM
        svm = LinearSVC(C=0.01, max_iter=1000, dual="auto")
        svm.fit(X, y)
        
        # Get CAV direction (perpendicular to decision boundary)
        cav_direction = svm.coef_[0]
        
        # Normalize
        cav_norm = np.linalg.norm(cav_direction)
        if cav_norm > 0:
            cav_direction = cav_direction / cav_norm
        
        # Convert to tensor and reshape to match feature dimensions
        cav = torch.tensor(cav_direction, device=self.device)
        
        # Reshape CAV to match feature dimensions
        cav = cav.reshape(self.concept_directions[0].shape)
        
        # Update the corresponding concept direction
        with torch.no_grad():
            self.concept_directions[concept_idx].data = cav.data
        
        logger.info(f"Successfully learned CAV for concept {concept_idx}")
        return cav
    
    def concept_loss(self, transformed_features, cav, desensitize=True):
        """
        Calculate the concept loss as described in the Concept Distillation paper.
        
        Args:
            transformed_features: The transformed features from the style transfer model
            cav: The concept activation vector
            desensitize: Whether to desensitize (True) or sensitize (False) to the concept
            
        Returns:
            Loss value that can be minimized during training
        """
        # Flatten features for cosine similarity calculation
        batch_size = transformed_features.shape[0]
        flattened_features = transformed_features.view(batch_size, -1)
        
        # Flatten CAV and normalize if not already normalized
        cav_flat = cav.view(-1)
        cav_norm = torch.norm(cav_flat)
        if cav_norm > 0:
            cav_flat = cav_flat / cav_norm
        
        # Calculate cosine similarity between features and CAV
        cos_sim = F.cosine_similarity(
            flattened_features, 
            cav_flat.unsqueeze(0).expand(batch_size, -1), 
            dim=1
        )
        
        # Take absolute value to minimize angular alignment regardless of direction
        cos_sim_abs = torch.abs(cos_sim)
        
        if desensitize:
            # For desensitizing: minimize the absolute cosine similarity
            return cos_sim_abs.mean()
        else:
            # For sensitizing: maximize the absolute cosine similarity
            return -cos_sim_abs.mean()
    
    def finetune_with_cav(self, dataloader, cav_idx=0, desensitize=True, 
                        num_epochs=5, learning_rate=1e-4, concept_weight=0.1, 
                        style_weight=1.0, content_weight=1.0):
        """
        Fine-tune the model using CAV-based concept loss.
        
        Args:
            dataloader: DataLoader with content-style pairs
            cav_idx: Index of the CAV to use
            desensitize: Whether to desensitize (True) or sensitize (False)
            num_epochs: Number of epochs
            learning_rate: Learning rate
            concept_weight: Weight for concept loss
            style_weight: Weight for style loss
            content_weight: Weight for content loss
        """
        logger.info(f"Fine-tuning with CAV {cav_idx}, desensitize={desensitize}")
        
        # Get the target CAV
        cav = self.concept_directions[cav_idx]
        
        # Create optimizer for matrix and decoder
        optimizer = optim.Adam(
            list(self.matrix.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate
        )
        
        # Track metrics
        running_loss = 0.0
        running_content_loss = 0.0
        running_style_loss = 0.0
        running_concept_loss = 0.0
        
        # Training loop
        self.train()
        for epoch in range(num_epochs):
            # Reset metrics for each epoch
            epoch_loss = 0.0
            epoch_content_loss = 0.0
            epoch_style_loss = 0.0
            epoch_concept_loss = 0.0
            
            # Process batches
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                # Get content and style images
                content_images = batch['content'].to(self.device)
                style_images = batch['style'].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                content_features = self.encoder(content_images)
                style_features = self.encoder(style_images)
                
                # Apply style transformation
                self.activations = []  # Clear activations
                transformed_features, matrix = self.matrix(
                    content_features[self.layer], style_features[self.layer]
                )
                
                # Compute stylized output
                stylized_images = self.decoder(transformed_features)
                
                # Compute content loss (simplified)
                stylized_features = self.encoder(stylized_images)
                content_loss = F.mse_loss(
                    stylized_features[self.layer], 
                    content_features[self.layer]
                )
                
                # Compute style loss (simplified)
                style_loss = F.mse_loss(matrix, matrix)  # Placeholder
                
                # Compute concept loss
                c_loss = self.concept_loss(transformed_features, cav, desensitize)
                
                # Compute total loss
                total_loss = (
                    content_weight * content_loss + 
                    style_weight * style_loss + 
                    concept_weight * c_loss
                )
                
                # Backward pass and optimization
                total_loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += total_loss.item()
                epoch_content_loss += content_loss.item()
                epoch_style_loss += style_loss.item()
                epoch_concept_loss += c_loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss.item(),
                    'content': content_loss.item(),
                    'concept': c_loss.item(),
                })
            
            # Compute epoch averages
            num_batches = len(dataloader)
            epoch_loss /= num_batches
            epoch_content_loss /= num_batches
            epoch_style_loss /= num_batches
            epoch_concept_loss /= num_batches
            
            # Log epoch metrics
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Loss={epoch_loss:.4f}, "
                       f"Content={epoch_content_loss:.4f}, "
                       f"Style={epoch_style_loss:.4f}, "
                       f"Concept={epoch_concept_loss:.4f}")
        
        logger.info("Fine-tuning complete")
    
    def forward(self, content_img, style_img, concept_strengths=None):
        """
        Forward pass with optional concept manipulation.
        
        Args:
            content_img: Content image tensor (B, 3, H, W)
            style_img: Style image tensor (B, 3, H, W)
            concept_strengths: List of strength values for each concept or tensor (B, num_concepts)
                              (None means no concept manipulation)
        
        Returns:
            Stylized image tensor (B, 3, H, W)
        """
        # Extract content and style features
        content_features = self.encoder(content_img)
        style_features = self.encoder(style_img)
        
        # Apply style matrix transformation
        transformed_features, _ = self.matrix(content_features[self.layer], style_features[self.layer])
        
        # Apply concept manipulation if strengths are provided
        if concept_strengths is not None:
            batch_size = transformed_features.shape[0]
            
            # Convert concept_strengths to proper tensor if it's a list
            if isinstance(concept_strengths, list):
                concept_strengths = torch.tensor([concept_strengths] * batch_size).to(self.device)
            elif len(concept_strengths.shape) == 1:
                concept_strengths = concept_strengths.unsqueeze(0).repeat(batch_size, 1)
            
            # Apply each concept direction with its corresponding strength
            for i in range(min(self.num_concepts, concept_strengths.shape[1])):
                # Get feature norm for adaptive scaling
                feature_norms = torch.norm(transformed_features.view(batch_size, -1), dim=1)
                concept_norm = torch.norm(self.concept_directions[i].view(-1))
                
                # Apply concept direction with adaptive strength for each sample in batch
                if concept_norm > 0:
                    for b in range(batch_size):
                        strength = concept_strengths[b, i].item()
                        if strength != 0:
                            scale_factor = feature_norms[b] / concept_norm
                            adaptive_strength = strength * scale_factor * 0.1
                            
                            # Apply the concept direction
                            transformed_features[b] = transformed_features[b] + (
                                adaptive_strength * self.concept_directions[i]
                            )
        
        # Decode to get stylized image
        stylized_img = self.decoder(transformed_features)
        
        return stylized_img.clamp(0, 1)
    
    def apply_concepts(self, content_img, style_img, concept_dict):
        """
        Apply concepts using a dictionary mapping concept names to strengths.
        
        Args:
            content_img: Content image tensor
            style_img: Style image tensor
            concept_dict: Dictionary mapping concept names to strength values
                         e.g. {'edges': 1.5, 'sharpness': -0.5}
        
        Returns:
            Stylized image tensor
        """
        # Create concept strengths array
        concept_strengths = torch.zeros(self.num_concepts)
        
        # Fill in provided concept strengths
        for name, strength in concept_dict.items():
            if name in self.concept_names:
                idx = self.concept_names.index(name)
                concept_strengths[idx] = strength
        
        # Apply concepts
        return self.forward(content_img, style_img, concept_strengths)


class WikiArtDataset(Dataset):
    """
    Dataset for WikiArt images with style categories.
    """
    def __init__(self, root_dir, transform=None, style_categories=None, max_images_per_category=None):
        self.root_dir = root_dir
        
        self.transform = transform
        self.max_images_per_category = max_images_per_category
        
        # Default to all styles if none specified
        self.style_categories = style_categories or self._get_all_style_categories()
        
        # Collect image paths
        self.images = self._collect_images()
        logger.info(f"Loaded WikiArt dataset with {len(self.images)} images from {len(self.style_categories)} styles")
    
    def _get_all_style_categories(self):
        """Get all style directories in root_dir."""
        style_dirs = [d for d in os.listdir(self.root_dir) 
                     if os.path.isdir(os.path.join(self.root_dir, d))]
        return style_dirs
    
    def _collect_images(self):
        """Collect all image paths."""
        images = []
        
        for style in self.style_categories:
            style_dir = os.path.join(self.root_dir, style)
            if not os.path.exists(style_dir):
                logger.warning(f"Style directory not found: {style_dir}")
                continue
            
            # Get all image files
            img_files = [f for f in os.listdir(style_dir) 
                        if f.lower().endswith(('jpg', 'jpeg', 'png'))]
            
            # Limit images per category if specified
            if self.max_images_per_category:
                img_files = img_files[:self.max_images_per_category]
            
            # Add image info
            for img_file in img_files:
                img_path = os.path.join(style_dir, img_file)
                images.append({
                    'path': img_path,
                    'style': style,
                    'filename': img_file
                })
        
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        
        try:
            image = Image.open(img_info['path']).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return {
                'image': image,
                'style': img_info['style'],
                'path': img_info['path']
            }
        except Exception as e:
            logger.error(f"Error loading image {img_info['path']}: {str(e)}")
            # Return a placeholder on error
            placeholder = torch.zeros(3, 256, 256)
            return {
                'image': placeholder,
                'style': img_info['style'],
                'path': img_info['path']
            }


class ConceptDataGenerator:
    """
    Generates variations of images with different concept intensities on-the-fly.
    Supports multiple concept types like edges, sharpness, and brightness.
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.concept_fns = {
            'edges': self.apply_edges,
            'sharpness': self.apply_sharpness,
            'brightness': self.apply_brightness
        }
    
    def generate_variations(self, images, concept_name, intensities=None):
        """
        Generate variations of images with different concept intensities.
        
        Args:
            images: Tensor of images (B, 3, H, W)
            concept_name: Name of concept to apply
            intensities: List/tensor of intensity values (if None, random values are used)
            
        Returns:
            Tuple of (modified_images, intensities)
        """
        batch_size = images.shape[0]
        
        # Generate random intensities if not provided
        if intensities is None:
            # Generate values between -1 and 1
            intensities = torch.rand(batch_size).to(self.device) * 2 - 1
        
        # Apply concept function
        if concept_name in self.concept_fns:
            modified_images = self.concept_fns[concept_name](images, intensities)
            return modified_images, intensities
        else:
            logger.warning(f"Unknown concept: {concept_name}")
            return images, torch.zeros(batch_size).to(self.device)
    
    def apply_edges(self, images, intensities):
        """Apply edge enhancement/reduction with different intensities."""
        batch_size = images.shape[0]
        modified = []
        
        # Process each image individually
        for i in range(batch_size):
            img = images[i].cpu()
            intensity = intensities[i].item()
            
            # Convert to PIL for edge processing
            pil_img = transforms.ToPILImage()(img)
            
            # Apply edge enhancement or reduction based on intensity
            if intensity > 0:  # Enhance edges
                # Apply edge enhancement filter
                enhanced = pil_img.filter(ImageFilter.EDGE_ENHANCE)
                # Apply multiple times for stronger effect
                for _ in range(int(intensity * 3)):
                    enhanced = enhanced.filter(ImageFilter.EDGE_ENHANCE)
                result = enhanced
            else:  # Reduce edges (blur)
                # Apply gaussian blur with radius based on intensity
                blur_radius = abs(intensity) * 2
                result = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Convert back to tensor
            tensor_result = transforms.ToTensor()(result).to(self.device)
            modified.append(tensor_result)
        
        return torch.stack(modified)
    
    def apply_sharpness(self, images, intensities):
        """Apply sharpness enhancement/reduction with different intensities."""
        batch_size = images.shape[0]
        modified = []
        
        for i in range(batch_size):
            img = images[i].cpu()
            intensity = intensities[i].item()
            
            # Convert to PIL
            pil_img = transforms.ToPILImage()(img)
            
            # Calculate sharpness factor (0 to 2, with 1 being original)
            # Map intensity from [-1, 1] to [0, 2]
            sharpness_factor = intensity + 1
            
            # Apply sharpness adjustment
            enhancer = ImageEnhance.Sharpness(pil_img)
            result = enhancer.enhance(sharpness_factor)
            
            # Convert back to tensor
            tensor_result = transforms.ToTensor()(result).to(self.device)
            modified.append(tensor_result)
        
        return torch.stack(modified)
    
    def apply_brightness(self, images, intensities):
        """Apply brightness adjustment with different intensities."""
        batch_size = images.shape[0]
        modified = []
        
        for i in range(batch_size):
            img = images[i].cpu()
            intensity = intensities[i].item()
            
            # Convert to PIL
            pil_img = transforms.ToPILImage()(img)
            
            # Map intensity from [-1, 1] to [0.5, 1.5]
            brightness_factor = intensity * 0.5 + 1.0
            
            # Apply brightness adjustment
            enhancer = ImageEnhance.Brightness(pil_img)
            result = enhancer.enhance(brightness_factor)
            
            # Convert back to tensor
            tensor_result = transforms.ToTensor()(result).to(self.device)
            modified.append(tensor_result)
        
        return torch.stack(modified)


class StyleTransferDataset(Dataset):
    """
    Dataset for style transfer training with content and style images.
    """
    def __init__(self, content_dir, style_dir, transform=None, max_content=None, max_styles=None):
        self.transform = transform
        
        # Load content images
        self.content_images = self._load_directory(content_dir, max_content)
        logger.info(f"Loaded {len(self.content_images)} content images")
        
        # Load style images
        self.style_images = self._load_directory(style_dir, max_styles)
        logger.info(f"Loaded {len(self.style_images)} style images")
        
        # Create pairs for training
        self.pairs = self._create_pairs()
    
    def _load_directory(self, directory, max_images=None):
        """Load images from a directory with size checking."""
        images = []
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return images
        
        # Get all image files
        img_files = [f for f in os.listdir(directory) 
                    if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        
        # Limit if specified
        if max_images:
            img_files = img_files[:max_images]
        
        # Add image info
        for img_file in img_files:
            img_path = os.path.join(directory, img_file)
            try:
                # Check image size before fully loading
                with Image.open(img_path) as img:
                    width, height = img.size
                    # Skip extremely large images
                    if width * height > 50000000:  # 50 million pixels
                        logger.warning(f"Skipping oversized image: {img_path} ({width}x{height})")
                        continue
                
                images.append({
                    'path': img_path,
                    'filename': img_file
                })
            except Exception as e:
                logger.error(f"Error checking image {img_path}: {str(e)}")
        
        return images
    
    def _create_pairs(self):
        """Create content-style pairs for training."""
        pairs = []
        
        # For simplicity, create all possible pairs
        # In practice, you might want to sample or limit the number of pairs
        for content in self.content_images:
            for style in self.style_images:
                pairs.append({
                    'content': content,
                    'style': style
                })
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        try:
            # Load content image
            content_img = Image.open(pair['content']['path']).convert('RGB')
            if self.transform:
                content_img = self.transform(content_img)
            
            # Load style image
            style_img = Image.open(pair['style']['path']).convert('RGB')
            if self.transform:
                style_img = self.transform(style_img)
            
            return {
                'content': content_img,
                'style': style_img,
                'content_path': pair['content']['path'],
                'style_path': pair['style']['path']
            }
        except Exception as e:
            logger.error(f"Error loading pair: {str(e)}")
            # Return placeholders on error
            placeholder = torch.zeros(3, 256, 256)
            return {
                'content': placeholder,
                'style': placeholder,
                'content_path': pair['content']['path'],
                'style_path': pair['style']['path']
            }


def train_concept_model(model, dataset, concept_generator, num_epochs=10, batch_size=8,
                       learning_rate=1e-4, device='cuda', save_dir='models',
                       log_interval=10, save_interval=1, use_wandb=False):
    """
    Train the concept-aware style transfer model.
    
    Args:
        model: ConceptAwareStyleTransfer model
        dataset: WikiArtDataset instance
        concept_generator: ConceptDataGenerator instance
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
        log_interval: Steps between logging
        save_interval: Epochs between saving checkpoints
        use_wandb: Whether to use Weights & Biases for logging
    """
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Only train concept directions
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.concept_directions:
        param.requires_grad = True
    
    # Create optimizer
    optimizer = optim.Adam([p for p in model.concept_directions], lr=learning_rate)
    
    # Create loss function
    criterion = nn.MSELoss()
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Track best loss
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        concept_losses = {name: 0.0 for name in concept_generator.concept_fns.keys()}
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, batch in enumerate(pbar):
            # Get images
            images = batch['image'].to(device)
            
            # Randomly select a concept for this batch
            concept_name = random.choice(list(concept_generator.concept_fns.keys()))
            concept_idx = model.concept_names.index(concept_name)
            
            # Create concept variations
            intensities = (torch.rand(images.shape[0]) * 2 - 1).to(device)  # -1 to 1
            modified_images, _ = concept_generator.generate_variations(images, concept_name, intensities)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            # Create concept strengths tensor (batch_size, num_concepts)
            concept_strengths = torch.zeros(images.shape[0], model.num_concepts).to(device)
            concept_strengths[:, concept_idx] = intensities
            
            # Generate stylized images with concept applied
            outputs = model(images, images, concept_strengths)
            
            # Compute loss
            loss = criterion(outputs, modified_images)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            concept_losses[concept_name] += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (i + 1),
                f'{concept_name}_loss': concept_losses[concept_name] / (i + 1)
            })
            
            # Log to wandb
            if use_wandb and i % log_interval == 0:
                wandb.log({
                    'epoch': epoch,
                    'step': epoch * len(dataloader) + i,
                    'loss': loss.item(),
                    f'{concept_name}_loss': loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0]
                })
        
        # End of epoch
        epoch_loss = running_loss / len(dataloader)
        
        # Log epoch stats
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
        
        # Log to wandb
        if use_wandb:
            concept_avg_losses = {f'{name}_avg_loss': loss / len(dataloader) 
                                for name, loss in concept_losses.items()}
            wandb.log({
                'epoch': epoch,
                'epoch_loss': epoch_loss,
                **concept_avg_losses
            })
        
        # Save checkpoint if improved
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Normalize concept directions
        model._normalize_concept_directions()
        
        # Update scheduler
        scheduler.step()
    
    # Save final model
    final_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'loss': epoch_loss,
    }, final_path)
    logger.info(f"Saved final model to {final_path}")
    
    return model


def evaluate_concept_model(model, content_dir, style_dir, output_dir,
                         concept_strengths=None, device='cuda'):
    """
    Evaluate the concept model by applying it to various content-style pairs
    and visualizing the results with different concept strengths.
    
    Args:
        model: Trained ConceptAwareStyleTransfer model
        content_dir: Directory with content images
        style_dir: Directory with style images
        output_dir: Directory to save outputs
        concept_strengths: Dictionary mapping concept names to list of strengths
        device: Device to run on
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    # Define default concept strengths if not provided
    if concept_strengths is None:
        concept_strengths = {
            'edges': [-2.0, -1.0, 0.0, 1.0, 2.0],
            'sharpness': [-2.0, -1.0, 0.0, 1.0, 2.0],
            'brightness': [-2.0, -1.0, 0.0, 1.0, 2.0]
        }
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    
    # Get images
    content_files = [f for f in os.listdir(content_dir) 
                    if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    style_files = [f for f in os.listdir(style_dir) 
                  if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    
    # Limit to a reasonable number for evaluation
    content_files = content_files[:5]
    style_files = style_files[:5]
    
    # For each content-style pair
    for content_file in content_files:
        for style_file in style_files:
            content_path = os.path.join(content_dir, content_file)
            style_path = os.path.join(style_dir, style_file)
            
            # Load images
            content_img = Image.open(content_path).convert('RGB')
            style_img = Image.open(style_path).convert('RGB')
            
            # Apply transform
            content_tensor = transform(content_img).unsqueeze(0).to(device)
            style_tensor = transform(style_img).unsqueeze(0).to(device)
            
            # Create output directory for this pair
            pair_dir = os.path.join(output_dir, f"{content_file.split('.')[0]}_{style_file.split('.')[0]}")
            os.makedirs(pair_dir, exist_ok=True)
            
            with torch.no_grad():
                # Base style transfer (no concepts)
                base_output = model(content_tensor, style_tensor, None)
                utils.save_image(
                    base_output,
                    os.path.join(pair_dir, 'base.png')
                )
                
                # Evaluate each concept
                for concept_name, strengths in concept_strengths.items():
                    concept_idx = model.concept_names.index(concept_name)
                    concept_outputs = []
                    
                    # Apply different strengths
                    for strength in strengths:
                        # Create concept strengths tensor
                        concept_tensor = torch.zeros(1, model.num_concepts).to(device)
                        concept_tensor[0, concept_idx] = strength
                        
                        # Apply concept
                        output = model(content_tensor, style_tensor, concept_tensor)
                        concept_outputs.append(output)
                        
                        # Save individual image
                        utils.save_image(
                            output,
                            os.path.join(pair_dir, f"{concept_name}_{strength:.1f}.png")
                        )
                    
                    # Save grid of outputs for this concept
                    utils.save_image(
                        torch.cat(concept_outputs, dim=0),
                        os.path.join(pair_dir, f"{concept_name}_grid.png"),
                        nrow=len(strengths)
                    )
                
                # Test combining concepts
                combined_outputs = []
                combined_labels = []
                
                # Try a few combinations
                combinations = [
                    {'edges': 1.0, 'sharpness': 1.0},
                    {'edges': -1.0, 'sharpness': 1.0},
                    {'edges': 1.0, 'sharpness': -1.0},
                    {'edges': 1.0, 'brightness': 1.0},
                    {'sharpness': 1.0, 'brightness': 1.0},
                    {'edges': 1.0, 'sharpness': 1.0, 'brightness': 1.0}
                ]
                
                for combo in combinations:
                    # Create concept tensor
                    concept_tensor = torch.zeros(1, model.num_concepts).to(device)
                    label_parts = []
                    
                    for name, strength in combo.items():
                        idx = model.concept_names.index(name)
                        concept_tensor[0, idx] = strength
                        label_parts.append(f"{name}:{strength:.1f}")
                    
                    # Apply combined concepts
                    output = model(content_tensor, style_tensor, concept_tensor)
                    combined_outputs.append(output)
                    combined_labels.append("_".join(label_parts))
                    
                    # Save individual image
                    utils.save_image(
                        output,
                        os.path.join(pair_dir, sanitize_filename(f"combined_{'_'.join(label_parts)}.png"))
                    )
                
                # Save combined grid
                utils.save_image(
                    torch.cat(combined_outputs, dim=0),
                    os.path.join(pair_dir, "combined_grid.png"),
                    nrow=3
                )
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")



def main():
    """Main function to run the WikiArt CAV training."""
    parser = argparse.ArgumentParser(description="Train concept-aware style transfer on WikiArt")
    
    # Data paths
    parser.add_argument("--wikiart_dir", default="datasets/wikiArt/wikiart",
                      help='WikiArt dataset directory')
    parser.add_argument("--content_dir", default="data/content",
                      help='Content images directory')
    parser.add_argument("--style_dir", default="data/style",
                      help='Style images directory')
    parser.add_argument("--output_dir", default="outputs/concept_transfer",
                      help='Output directory for results')
    
    # Model paths
    parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                      help='Pre-trained encoder path')
    parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                      help='Pre-trained decoder path')
    parser.add_argument("--matrix_dir", default='models/r41.pth',
                      help='Pre-trained matrix path')
    parser.add_argument("--model_dir", default='models/concept_models',
                      help='Directory to save trained models')
    parser.add_argument("--load_model", default=None,
                      help='Path to load pre-trained concept model')
    
    # Training parameters
    parser.add_argument("--num_concepts", type=int, default=3,
                      help='Number of concept dimensions to learn')
    parser.add_argument("--batch_size", type=int, default=8,
                      help='Batch size for training')
    parser.add_argument("--num_epochs", type=int, default=10,
                      help='Number of training epochs')
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument("--max_images", type=int, default=1000,
                      help='Maximum number of images to use per style')
    
    # Execution mode
    parser.add_argument("--mode", choices=['train', 'evaluate', 'apply'],
                      default='train', help='Execution mode')
    parser.add_argument("--concept", default=None,
                      help='Specific concept to train/evaluate (None for all)')
    
    # Logging and visualization
    parser.add_argument("--log_interval", type=int, default=10,
                      help='Steps between logging')
    parser.add_argument("--save_interval", type=int, default=1,
                      help='Epochs between saving checkpoints')
    parser.add_argument("--use_wandb", action='store_true',
                      help='Use Weights & Biases for logging')
    parser.add_argument("--project_name", default='concept-transfer',
                      help='Project name for wandb')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project=args.project_name)
        wandb.config.update(args)
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    
    # Initialize models
    logger.info("Initializing models...")
    encoder = encoder4()
    decoder = decoder4()
    matrix = MulLayer('r41')
    
    # Load pre-trained weights
    encoder.load_state_dict(torch.load(args.vgg_dir, weights_only=True))
    decoder.load_state_dict(torch.load(args.decoder_dir, weights_only=True))
    matrix.load_state_dict(torch.load(args.matrix_dir, weights_only=True))
    
    # Move models to device
    encoder.to(device)
    decoder.to(device)
    matrix.to(device)
    
    # Create concept-aware model
    model = ConceptAwareStyleTransfer(
        encoder=encoder,
        decoder=decoder,
        matrix=matrix,
        num_concepts=args.num_concepts,
        device=device
    ).to(device)
    
    # Load pre-trained concept model if specified
    if args.load_model:
        logger.info(f"Loading pre-trained concept model from {args.load_model}")
        checkpoint = torch.load(args.load_model)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Execute based on mode
    if args.mode == 'train':
        logger.info("Initializing training...")
        
        # Initialize dataset
        wikiart_dataset = WikiArtDataset(
            root_dir=args.wikiart_dir,
            transform=transform,
            max_images_per_category=args.max_images
        )
        
        # Initialize concept generator
        concept_generator = ConceptDataGenerator(device=device)
        
        # Train model
        train_concept_model(
            model=model,
            dataset=wikiart_dataset,
            concept_generator=concept_generator,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            save_dir=args.model_dir,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            use_wandb=args.use_wandb
        )
        
    elif args.mode == 'evaluate':
        logger.info("Evaluating concept model...")
        
        # Evaluate model
        evaluate_concept_model(
            model=model,
            content_dir=args.content_dir,
            style_dir=args.style_dir,
            output_dir=args.output_dir,
            device=device
        )
        
    elif args.mode == 'apply':
        logger.info("Applying concept to new images...")
        
        # Load test images
        content_path = args.content_dir
        style_path = args.style_dir
        
        # Get all image files
        content_files = [f for f in os.listdir(content_path) 
                        if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        style_files = [f for f in os.listdir(style_path) 
                      if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        
        # Limit to a reasonable number
        content_files = content_files[:5]
        style_files = style_files[:5]
        
        # Create output directory
        apply_dir = os.path.join(args.output_dir, 'applications')
        os.makedirs(apply_dir, exist_ok=True)
        
        # Process each content-style pair
        with torch.no_grad():
            for content_file in content_files:
                content_img = Image.open(os.path.join(content_path, content_file)).convert('RGB')
                content_tensor = transform(content_img).unsqueeze(0).to(device)
                
                for style_file in style_files:
                    style_img = Image.open(os.path.join(style_path, style_file)).convert('RGB')
                    style_tensor = transform(style_img).unsqueeze(0).to(device)
                    
                    # Apply concepts
                    concept_name = args.concept or 'edges'  # Default to edges if none specified
                    concept_idx = model.concept_names.index(concept_name)
                    
                    # Create output directory for this pair
                    pair_dir = os.path.join(apply_dir, f"{content_file.split('.')[0]}_{style_file.split('.')[0]}")
                    os.makedirs(pair_dir, exist_ok=True)
                    
                    # Save original images
                    utils.save_image(content_tensor, os.path.join(pair_dir, 'content.png'))
                    utils.save_image(style_tensor, os.path.join(pair_dir, 'style.png'))
                    
                    # Base style transfer (no concept)
                    base_output = model(content_tensor, style_tensor, None)
                    utils.save_image(base_output, os.path.join(pair_dir, 'base_transfer.png'))
                    
                    # Apply concept with different strengths
                    strengths = [-5.0, -4.0,-2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 5.0]
                    concept_outputs = []
                    
                    for strength in strengths:
                        concept_tensor = torch.zeros(1, model.num_concepts).to(device)
                        concept_tensor[0, concept_idx] = strength
                        
                        output = model(content_tensor, style_tensor, concept_tensor)
                        concept_outputs.append(output)
                        
                        utils.save_image(output, os.path.join(pair_dir, f"{concept_name}_{strength:.1f}.png"))
                    
                    # Save grid
                    utils.save_image(
                        output,
                        os.path.join(pair_dir, f"{concept_name}_{strength:.1f}.png")
                    )
        
        logger.info(f"Applied concept {concept_name} to {len(content_files) * len(style_files)} pairs")

    logger.info("Done!")


if __name__ == "__main__":
    main()