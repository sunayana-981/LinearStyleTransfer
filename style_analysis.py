import torch
import clip
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from tqdm import tqdm 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StyleUnderstandingAnalyzer:
    """
    Analyzes how well VLMs understand artistic styles through 
    various experimental validations.
    """
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Define style-related text prompts
        self.style_attributes = {
            'color': [
                "an image with warm colors",
                "an image with cool colors",
                "an image with vibrant colors",
                "an image with muted colors"
            ],
            'texture': [
                "an image with rough texture",
                "an image with smooth texture",
                "an image with regular patterns",
                "an image with irregular patterns"
            ],
            'composition': [
                "an image with balanced composition",
                "an image with dynamic composition",
                "an image with minimal composition",
                "an image with complex composition"
            ]
        }
        
        # Art movement prompts
        self.art_movements = [
            "impressionist painting",
            "cubist artwork",
            "abstract expressionist painting",
            "pop art style",
            "minimalist artwork",
            "baroque painting"
        ]

    def analyze_style_image(self, image_path):
        """
        Comprehensive analysis of how CLIP perceives a style image.
        Returns similarity scores for different style attributes.
        """
        # Load and preprocess image
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Get image embedding
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        results = {}
        
        # Analyze style attributes
        for category, prompts in self.style_attributes.items():
            # Encode text prompts
            text_tokens = clip.tokenize(prompts).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarities
            similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            results[category] = {
                prompt: score.item() 
                for prompt, score in zip(prompts, similarities[0])
            }
            
        # Analyze art movement classification
        text_tokens = clip.tokenize(self.art_movements).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        movement_similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        results['art_movement'] = {
            movement: score.item()
            for movement, score in zip(self.art_movements, movement_similarities[0])
        }
        
        return results

    def analyze_style_dataset(self, dataset_path):
        """
        Analyzes a dataset of style images to understand clustering in CLIP's embedding space.
        
        Args:
            dataset_path: Either a path to the dataset directory or a list of image paths
        """
        # Handle both directory path and list of paths
        if isinstance(dataset_path, (str, Path)):
            # If given a directory, collect image paths
            path_obj = Path(dataset_path)
            image_paths = list(path_obj.glob("**/*.jpg")) + list(path_obj.glob("**/*.png"))
        else:
            # If given a list of paths, use them directly
            image_paths = [Path(p) for p in dataset_path]
        
        # Collect embeddings for all images
        embeddings = []
        labels = []
        
        for path in tqdm(image_paths, desc="Processing images"):
            try:
                # Load and preprocess image
                image = Image.open(path)
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                
                # Get embedding
                with torch.no_grad():
                    features = self.model.encode_image(image_input)
                    features = features / features.norm(dim=-1, keepdim=True)
                    embeddings.append(features.cpu().numpy())
                    
                # Use parent folder name as label
                labels.append(path.parent.name)
            except Exception as e:
                logger.warning(f"Error processing image {path}: {str(e)}")
                continue
        
        if not embeddings:
            raise ValueError("No valid images found in dataset")
            
        embeddings = np.vstack(embeddings)
        
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        return embeddings_2d, labels

    def visualize_style_space(self, embeddings_2d, labels):
        """
        Creates a visualization of the style embedding space.
        """
        plt.figure(figsize=(12, 8))
        
        # Convert labels to numeric
        unique_labels = list(set(labels))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = [label_to_id[label] for label in labels]
        
        # Create scatter plot
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=numeric_labels,
            cmap='tab20',
            alpha=0.6
        )
        
        # Add legend
        plt.legend(
            handles=scatter.legend_elements()[0],
            labels=unique_labels,
            title="Style Categories",
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
        
        plt.title("CLIP Style Embedding Space")
        plt.tight_layout()
        
        return plt.gcf()

    def evaluate_style_consistency(self, image_paths):
        """
        Evaluates how consistently CLIP perceives style across different
        images of the same style category.
        """
        results = []
        
        for path in image_paths:
            # Analyze individual image
            scores = self.analyze_style_image(path)
            results.append(scores)
            
        # Calculate consistency metrics
        consistency = {}
        for category in self.style_attributes:
            category_scores = [r[category] for r in results]
            
            # Calculate variance in predictions
            variances = {}
            for prompt in self.style_attributes[category]:
                scores = [scores[prompt] for scores in category_scores]
                variances[prompt] = np.var(scores)
                
            consistency[category] = {
                'mean_variance': np.mean(list(variances.values())),
                'per_prompt_variance': variances
            }
            
        return consistency

def run_style_analysis(dataset_path, output_path):
    """
    Runs a complete analysis of style understanding and saves results.
    """
    analyzer = StyleUnderstandingAnalyzer()
    
    # Analyze embedding space
    embeddings_2d, labels = analyzer.analyze_style_dataset(dataset_path)
    
    # Create visualization
    fig = analyzer.visualize_style_space(embeddings_2d, labels)
    fig.savefig(str(Path(output_path) / "style_space.png"))
    
    # Analyze consistency
    image_paths = list(Path(dataset_path).glob("**/*.jpg"))
    consistency = analyzer.evaluate_style_consistency(image_paths)
    
    # Save numerical results
    np.save(
        str(Path(output_path) / "embeddings.npy"),
        embeddings_2d
    )
    
    return embeddings_2d, labels, consistency