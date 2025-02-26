import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from collections import defaultdict 

class SDStyleClusterAnalyzer:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load SD components
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        
        self.vae.eval()
        self.text_encoder.eval()

        # Define style-specific feature extractors
        self.style_features = {
            'composition': [
                "analyze the composition and layout",
                "study the spatial arrangement",
                "examine the structural elements"
            ],
            'color': [
                "focus on the color palette",
                "analyze the tonal relationships",
                "examine the color harmony"
            ],
            'technique': [
                "analyze the brushwork technique",
                "examine the artistic method",
                "study the painting style"
            ]
        }

    @torch.no_grad()
    def extract_style_features(self, image: Image) -> torch.Tensor:
        """Extract comprehensive style features from image"""
        # Normalize and encode image
        image = image.convert("RGB").resize((512, 512))
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Get VAE latents
        latents = self.vae.encode(image).latent_dist.mean
        
        # Extract multi-scale features
        features = []
        
        # 1. Global features (mean and std across spatial dimensions)
        global_features = torch.cat([
            latents.mean(dim=[2, 3]),
            latents.std(dim=[2, 3])
        ], dim=1)
        features.append(global_features)
        
        # 2. Local features (grid-based statistics)
        grid_size = 4
        patches = F.unfold(latents, kernel_size=latents.shape[-1]//grid_size)
        local_features = torch.cat([
            patches.mean(dim=2),
            patches.std(dim=2)
        ], dim=1)
        features.append(local_features)
        
        return torch.cat(features, dim=1)

    @torch.no_grad()
    def get_style_embeddings(self, style_name: str) -> torch.Tensor:
        """Get style-specific text embeddings"""
        all_embeddings = []
        
        for feature_type, prompts in self.style_features.items():
            formatted_prompts = [f"{p} in {style_name} style" for p in prompts]
            
            text_input = self.tokenizer(
                formatted_prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            embeddings = self.text_encoder(text_input.input_ids)[0]
            # Average across sequence length
            avg_embedding = embeddings.mean(dim=1)
            all_embeddings.append(avg_embedding)
        
        return torch.cat(all_embeddings, dim=1)

    def analyze_dataset(self, dataset_path: str, num_samples: int = 100):
        dataset_path = Path(dataset_path)
        all_features = []
        all_labels = []
        style_clusters = defaultdict(list)
        
        # Collect and sample images
        image_paths = []
        for ext in ('*.jpg', '*.png', '*.jpeg'):
            image_paths.extend(dataset_path.rglob(ext))
        
        sampled_paths = np.random.choice(image_paths, num_samples, replace=False)
        
        for img_path in tqdm(sampled_paths):
            try:
                image = Image.open(img_path).convert('RGB')
                style_label = img_path.parent.name
                
                # Extract image features
                image_features = self.extract_style_features(image)
                # Get style embeddings
                style_embeddings = self.get_style_embeddings(style_label)
                
                # Combine features
                combined = torch.cat([
                    image_features.flatten(),
                    style_embeddings.mean(dim=0)
                ])
                
                # Store features and label
                style_clusters[style_label].append(combined.cpu().numpy())
                all_features.append(combined.cpu().numpy())
                all_labels.append(style_label)
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
        
        return np.array(all_features), np.array(all_labels), style_clusters

    def cluster_styles(self, features: np.ndarray, n_clusters: int = None):
        """Perform hierarchical clustering on style features"""
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler
        
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
        
        # If n_clusters not specified, use square root of sample size
        if n_clusters is None:
            n_clusters = int(np.sqrt(len(features)))
        
        # Make sure n_clusters is less than number of samples
        n_clusters = min(n_clusters, len(features) - 1)
        
        print(f"Clustering into {n_clusters} clusters")
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        
        return clustering.fit_predict(normalized_features)

    def visualize_clusters(self, features: np.ndarray, labels: np.ndarray, clusters: np.ndarray, 
                         output_path: str):
        """Visualize clustered styles"""
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot clusters
        scatter = plt.scatter(
            features_2d[:, 0], 
            features_2d[:, 1],
            c=clusters,
            cmap='tab20',
            alpha=0.6
        )
        
        # Add labels
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            center = features_2d[mask].mean(axis=0)
            plt.annotate(
                label,
                center,
                fontsize=8,
                alpha=0.7
            )
        
        plt.title("Style Clusters")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    analyzer = SDStyleClusterAnalyzer()
    
    # Process dataset
    features, labels, style_clusters = analyzer.analyze_dataset(
        "datasets/wikiArt/wikiart", 
        num_samples=500
    )
    
    # Get clusters with specific number of clusters
    n_clusters = min(30, len(np.unique(labels)))  # or any reasonable number
    clusters = analyzer.cluster_styles(features, n_clusters=n_clusters)
    
    # Visualize
    analyzer.visualize_clusters(features, labels, clusters, "style_clusters.png")