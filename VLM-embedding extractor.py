import torch
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForImageTextRetrieval,
    LlavaProcessor, LlavaForConditionalGeneration
)
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple
import os
from tqdm import tqdm

class VLMEmbeddingAnalyzer:
    def __init__(self):
        # Initialize models and processors
        self.models = {}
        self.processors = {}
        
        # CLIP
        print("Loading CLIP...")
        self.models['clip'] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processors['clip'] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # BLIP
        print("Loading BLIP...")
        self.models['blip'] = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        self.processors['blip'] = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        
        # Note: LLaVA and Flamingo require additional setup/dependencies
        # Add them here if needed
        
    def extract_embeddings(self, image_path: str, model_name: str) -> np.ndarray:
        """Extract embeddings for a single image using specified model."""
        image = Image.open(image_path).convert('RGB')
        
        if model_name == 'clip':
            inputs = self.processors['clip'](images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.models['clip'].get_image_features(**inputs)
            embeddings = outputs.numpy()
            
        elif model_name == 'blip':
            inputs = self.processors['blip'](images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.models['blip'].get_image_features(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            
        # Add other models here
            
        return embeddings
    
    def process_dataset(self, 
                       image_dir: str,
                       labels: Dict[str, int]) -> Dict[str, Dict[str, np.ndarray]]:
        """Process entire dataset and return embeddings for each model."""
        embeddings = {model: {'features': [], 'labels': []} for model in self.models.keys()}
        
        for image_name in tqdm(os.listdir(image_dir)):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(image_dir, image_name)
            label = labels.get(image_name)
            
            if label is None:
                continue
                
            for model_name in self.models.keys():
                emb = self.extract_embeddings(image_path, model_name)
                embeddings[model_name]['features'].append(emb.flatten())
                embeddings[model_name]['labels'].append(label)
        
        # Convert to numpy arrays
        for model_name in embeddings:
            embeddings[model_name]['features'] = np.stack(embeddings[model_name]['features'])
            embeddings[model_name]['labels'] = np.array(embeddings[model_name]['labels'])
            
        return embeddings
    
    def cluster_and_evaluate(self,
                           embeddings: Dict[str, Dict[str, np.ndarray]],
                           n_clusters: int) -> Dict[str, Dict[str, float]]:
        """Perform clustering and evaluate results for each model."""
        results = {}
        
        for model_name, data in embeddings.items():
            features = data['features']
            true_labels = data['labels']
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            pred_labels = kmeans.fit_predict(features)
            
            # Calculate metrics
            results[model_name] = {
                'silhouette': silhouette_score(features, pred_labels),
                'ari': adjusted_rand_score(true_labels, pred_labels),
                'nmi': normalized_mutual_info_score(true_labels, pred_labels)
            }
            
        return results
    
    def visualize_clusters(self,
                         embeddings: Dict[str, Dict[str, np.ndarray]],
                         results: Dict[str, Dict[str, float]]):
        """Visualize clustering results using PCA."""
        fig, axes = plt.subplots(1, len(embeddings), figsize=(6*len(embeddings), 5))
        
        for idx, (model_name, data) in enumerate(embeddings.items()):
            # Reduce dimensionality for visualization
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(data['features'])
            
            # Plot
            ax = axes[idx] if len(embeddings) > 1 else axes
            scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                               c=data['labels'], cmap='tab10')
            ax.set_title(f'{model_name.upper()}\nSilhouette: {results[model_name]["silhouette"]:.3f}\n'
                        f'ARI: {results[model_name]["ari"]:.3f}\n'
                        f'NMI: {results[model_name]["nmi"]:.3f}')
            
        plt.tight_layout()
        plt.show()

# Example usage
def main():
    # Initialize analyzer
    analyzer = VLMEmbeddingAnalyzer()
    
    # Example data structure for labels
    labels = {
        'image1.jpg': 0,
        'image2.jpg': 1,
        # ... more image labels
    }
    
    # Process dataset
    embeddings = analyzer.process_dataset(
        image_dir='data/style/',
        labels=labels
    )
    
    # Perform clustering and evaluation
    results = analyzer.cluster_and_evaluate(
        embeddings=embeddings,
        n_clusters=len(set(labels.values()))
    )
    
    # Visualize results
    analyzer.visualize_clusters(embeddings, results)
    
    # Print detailed results
    print("\nDetailed Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.3f}")

if __name__ == "__main__":
    main()