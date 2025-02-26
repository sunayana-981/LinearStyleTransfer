import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import clip
import json
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StyleImageDataset(Dataset):
    def __init__(self, style_dir, transform=None, max_images=100):
        """
        Args:
            style_dir: Path to style directory
            transform: Optional transform to be applied
            max_images: Maximum number of images to use
        """
        self.image_paths = []
        style_dir = Path(style_dir)
        
        # Collect all image paths
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        for ext in valid_extensions:
            self.image_paths.extend(list(style_dir.glob(f'**/*{ext}')))
        
        # Randomly sample if we have more images than max_images
        if len(self.image_paths) > max_images:
            self.image_paths = random.sample(self.image_paths, max_images)
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                               (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return a blank image in case of error
            return torch.zeros(3, 224, 224)

class ConceptAnalyzer:
    def __init__(self, style_analysis_data: dict, clip_model, device='cuda'):
        self.style_data = style_analysis_data
        self.clip_model = clip_model
        self.device = device
        self.concept_categories = ['color', 'texture', 'composition']

    def analyze_style_images(self, style_dir: str, batch_size=16):
        """Analyze actual style images using CLIP."""
        dataset = StyleImageDataset(style_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        image_features = []
        logger.info("Extracting features from style images...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = batch.to(self.device)
                features = self.clip_model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True)
                image_features.append(features)
        
        return torch.cat(image_features)

    def generate_concept_embeddings(self, style_name: str, category: str):
        """Generate CLIP embeddings for concepts with variations."""
        concepts = self.style_data[style_name][category]['per_prompt_variance'].keys()
        all_embeddings = []
        concept_labels = []

        for concept in concepts:
            # Generate varied prompts
            prompts = [
                f"a {style_name} artwork with {concept}",
                f"an artistic piece showing {concept}",
                f"a painting with {concept}",
                f"{style_name} art demonstrating {concept}",
                f"visual artwork featuring {concept}",
                f"an example of {style_name} with {concept}",
                f"{concept} in {style_name} style",
                f"artistic representation of {concept}",
                f"{style_name} piece emphasizing {concept}"
            ]

            tokens = clip.tokenize(prompts).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_embeddings.append(text_features)
                concept_labels.extend([concept] * len(prompts))

        embeddings = torch.cat(all_embeddings)
        return embeddings, concept_labels

    def visualize_concept_comparison(self, 
                                   image_features: torch.Tensor,
                                   concept_features: torch.Tensor,
                                   concept_labels: list,
                                   title: str,
                                   output_path: str):
        """Visualize comparison between actual images and concept embeddings."""
        # Convert to numpy and combine features
        image_features_np = image_features.cpu().numpy()
        concept_features_np = concept_features.cpu().numpy()
        combined_features = np.vstack([image_features_np, concept_features_np])
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(combined_features)
        
        # Split back into image and concept features
        n_images = len(image_features_np)
        image_scaled = features_scaled[:n_images]
        concept_scaled = features_scaled[n_images:]
        
        # Get top variable dimensions
        variances = np.var(features_scaled, axis=0)
        top_dims = np.argsort(variances)[-2:]
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot image points
        plt.scatter(image_scaled[:, top_dims[0]], 
                   image_scaled[:, top_dims[1]],
                   alpha=0.3, 
                   label='Style Images',
                   color='gray')
        
        # Plot concept points with different colors
        unique_concepts = list(set(concept_labels))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_concepts)))
        
        for concept, color in zip(unique_concepts, colors):
            mask = np.array(concept_labels) == concept
            concept_points = concept_scaled[mask]
            plt.scatter(concept_points[:, top_dims[0]], 
                       concept_points[:, top_dims[1]],
                       label=concept,
                       color=color,
                       alpha=0.6)
        
        plt.title(title)
        plt.xlabel(f'Feature Dimension {top_dims[0]}')
        plt.ylabel(f'Feature Dimension {top_dims[1]}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

def analyze_style(style_name: str, style_dir: str, clip_model, style_data: dict, output_dir: str):
    """Run complete analysis for a style using both image and concept data."""
    analyzer = ConceptAnalyzer(style_data, clip_model)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get image features
    image_features = analyzer.analyze_style_images(style_dir)

    # Analyze each category
    for category in analyzer.concept_categories:
        logger.info(f"Analyzing {category} concepts for {style_name}")
        
        # Generate concept embeddings
        concept_features, concept_labels = analyzer.generate_concept_embeddings(style_name, category)
        
        # Create visualization comparing images and concepts
        viz_path = output_dir / f"{style_name}_{category}_concept_comparison.png"
        analyzer.visualize_concept_comparison(
            image_features,
            concept_features,
            concept_labels,
            f"{style_name} {category} Concept Space",
            str(viz_path)
        )

if __name__ == "__main__":
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Paths
    style_name = "Impressionism"
    style_dir = Path("datasets/wikiArt/wikiart/Impressionism")
    analysis_path = Path("analysis_results/style_consistency.json")
    output_dir = Path("output/concept_analysis")
    
    # Load style analysis data
    with open(analysis_path, 'r') as f:
        style_data = json.load(f)
    
    logger.info(f"Starting concept analysis for {style_name}")
    analyze_style(style_name, style_dir, model, style_data, output_dir)
    logger.info("Analysis complete")