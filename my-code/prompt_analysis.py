import torch
import clip
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple

class StylePromptAnalyzer:
    def __init__(self, model_name: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Define different prompting strategies
        self.prompt_strategies = {
            'basic': [
                "a painting"
            ],
            'descriptive': [
                "an artwork with {aspect}",
                "a painting showing {aspect}",
                "an artistic image with {aspect}"
            ],
            'analytical': [
                "analyze the {aspect} in this painting",
                "examine how {aspect} is used in this artwork",
                "study the {aspect} elements in this piece"
            ],
            'art_historical': [
                "examine how {aspect} reflects the artistic style",
                "analyze how {aspect} contributes to the period's characteristics",
                "study how {aspect} represents the artistic movement"
            ]
        }
        
        # Define aspects to analyze
        self.aspects = {
            'color': [
                "warm colors", "cool colors",
                "vibrant palette", "muted tones",
                "light and dark contrast"
            ],
            'texture': [
                "brushwork", "surface texture",
                "paint application", "material quality",
                "artistic technique"
            ],
            'composition': [
                "spatial arrangement", "visual balance",
                "geometric structure", "form organization",
                "compositional harmony"
            ]
        }

    def extract_embeddings_with_strategy(
        self, 
        image: Image.Image, 
        strategy: str,
        aspect: str
    ) -> torch.Tensor:
        """Extract embeddings using a specific prompting strategy."""
        all_embeddings = []
        
        # Get prompts for this strategy
        prompt_templates = self.prompt_strategies[strategy]
        aspect_terms = self.aspects[aspect]
        
        # Generate all combinations of prompts
        for template in prompt_templates:
            for term in aspect_terms:
                prompt = template.format(aspect=term)
                
                with torch.no_grad():
                    # Encode text prompt
                    text = clip.tokenize([prompt]).to(self.device)
                    text_features = self.model.encode_text(text)
                    
                    # Encode image
                    image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                    image_features = self.model.encode_image(image_input)
                    
                    # Normalize features
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Combine features (flattened)
                    combined = torch.cat([image_features, text_features], dim=-1).flatten()
                    all_embeddings.append(combined)
        
        # Average all embeddings
        return torch.mean(torch.stack(all_embeddings), dim=0)

    def analyze_prompting_effectiveness(
        self, 
        dataset_path: str,
        num_samples: int = 100
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Analyze how different prompting strategies affect concept separation."""
        dataset_path = Path(dataset_path)
        results = {strategy: {aspect: [] for aspect in self.aspects} 
                for strategy in self.prompt_strategies}
        
        print(f"Looking for images in {dataset_path}")
        
        # Collect image paths
        image_paths = []
        for ext in ('*.jpg', '*.png', '*.jpeg'):
            image_paths.extend(dataset_path.rglob(ext))
        
        print(f"Found {len(image_paths)} images")
        
        # Sample images
        sampled_paths = np.random.choice(image_paths, num_samples, replace=False)
        
        print(f"Processing {num_samples} sampled images...")
        for img_path in tqdm(sampled_paths):
            try:
                image = Image.open(img_path).convert('RGB')
                style_label = img_path.parent.name
                
                # Get embeddings for each strategy and aspect
                for strategy in self.prompt_strategies:
                    for aspect in self.aspects:
                        embedding = self.extract_embeddings_with_strategy(
                            image, strategy, aspect
                        )
                        results[strategy][aspect].append({
                            'embedding': embedding.cpu().numpy(),
                            'style': style_label
                        })
                        
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
        
        # Verify embedding shapes
        for strategy in results:
            for aspect in results[strategy]:
                embeddings = results[strategy][aspect]
                if embeddings:
                    print(f"Shape for {strategy}-{aspect}: {embeddings[0]['embedding'].shape}")
                    
        return results

    def visualize_concept_separation(self, results: Dict, output_path: str):
        """Visualize how different strategies separate concepts."""
        fig, axes = plt.subplots(len(self.prompt_strategies), 
                            len(self.aspects), 
                            figsize=(15, 20))
        
        for i, (strategy, aspects) in enumerate(results.items()):
            for j, (aspect, embeddings) in enumerate(aspects.items()):
                # Extract embeddings and labels
                X = np.stack([e['embedding'] for e in embeddings])
                labels = [e['style'] for e in embeddings]
                
                # Reshape the embeddings to 2D
                X_reshaped = X.reshape(X.shape[0], -1)  # Flatten all dimensions after the first
                
                # Reduce dimensionality for visualization
                tsne = TSNE(n_components=2, random_state=42)
                X_2d = tsne.fit_transform(X_reshaped)
                
                # Plot
                ax = axes[i][j]
                scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], 
                                c=[hash(l) % 20 for l in labels],  # Modulo to ensure color index is in range
                                cmap='tab20', alpha=0.6)
                
                ax.set_title(f"{strategy} - {aspect}")
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = StylePromptAnalyzer()
    
    # Analyze dataset
    results = analyzer.analyze_prompting_effectiveness(
        "datasets/wikiArt/wikiart",
        num_samples=100
    )
    
    # Visualize results
    analyzer.visualize_concept_separation(
        results,
        "prompt_analysis.png"
    )