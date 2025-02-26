import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F
import timm
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from collections import defaultdict

class DINOStyleAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load ViT model using timm
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize hooks
        self.activation = {}
        self.hooks = []
        
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        
        # Add hooks to blocks
        for i, block in enumerate(self.model.blocks):
            self.hooks.append(block.register_forward_hook(get_activation(f'block_{i}')))
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # Define feature categories
        self.feature_categories = {
            'local_patterns': [1, 2, 3],  # Early blocks for texture and local details
            'mid_level': [4, 5, 6],  # Mid blocks for brushwork and medium-scale patterns
            'global_structure': [7, 8, 9],  # Later blocks for composition
        }

    def __del__(self):
        if hasattr(self, 'hooks'):
            for hook in self.hooks:
                hook.remove()

    @torch.no_grad()
    def get_intermediate_features(self, image: Image.Image):
        """Extract features from different layers of DINO"""
        try:
            img = self.transform(image).unsqueeze(0).to(self.device)
            
            # Forward pass to trigger hooks
            _ = self.model(img)
            
            # Process features for each category
            features = {}
            for category, blocks in self.feature_categories.items():
                category_features = []
                for block_idx in blocks:
                    # Get activation for this block
                    block_output = self.activation[f'block_{block_idx}']  # [1, num_patches+1, hidden_dim]
                    
                    if category == 'local_patterns':
                        # Use patch tokens directly for local patterns
                        patch_features = block_output[:, 1:, :]  # Skip CLS token
                        # Compute statistics over patches
                        mean_features = patch_features.mean(dim=1)  # [1, hidden_dim]
                        std_features = patch_features.std(dim=1)   # [1, hidden_dim]
                        features_combined = torch.cat([mean_features, std_features], dim=1)
                        category_features.append(features_combined.flatten())
                    
                    elif category == 'mid_level':
                        # Use relationships between neighboring patches
                        patch_features = block_output[:, 1:, :]  # Skip CLS token
                        # Reshape to 2D grid (assuming square image)
                        grid_size = int(np.sqrt(patch_features.size(1)))
                        grid_features = patch_features.view(1, grid_size, grid_size, -1)
                        # Compute local correlations
                        spatial_features = F.avg_pool2d(
                            grid_features.permute(0, 3, 1, 2), 
                            kernel_size=2, 
                            stride=1
                        )
                        category_features.append(spatial_features.flatten())
                    
                    else:  # global_structure
                        # Use CLS token and its relationship to patches
                        cls_token = block_output[:, 0, :]  # [1, hidden_dim]
                        patch_tokens = block_output[:, 1:, :]  # [1, num_patches, hidden_dim]
                        # Compute attention-like scores
                        similarity = torch.matmul(cls_token.unsqueeze(1), patch_tokens.transpose(1, 2))
                        category_features.append(similarity.flatten())
                
                # Combine features for this category
                features[category] = torch.cat(category_features, dim=0)
            
            return features
            
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            return None

    def analyze_dataset(self, dataset_path: str, num_samples: int = 100):
        dataset_path = Path(dataset_path)
        results = {category: [] for category in self.feature_categories.keys()}
        
        try:
            # Collect image paths
            image_paths = []
            for ext in ('*.jpg', '*.png', '*.jpeg'):
                image_paths.extend(list(dataset_path.rglob(ext)))
            
            if not image_paths:
                raise ValueError(f"No images found in {dataset_path}")
            
            print(f"Found {len(image_paths)} images")
            
            # Sample images
            num_samples = min(num_samples, len(image_paths))
            sampled_paths = np.random.choice(image_paths, num_samples, replace=False)
            
            valid_results = False
            for img_path in tqdm(sampled_paths):
                try:
                    image = Image.open(img_path).convert('RGB')
                    style_label = img_path.parent.name
                    
                    # Get features for all categories
                    features = self.get_intermediate_features(image)
                    
                    if features is not None:
                        # Store results for each category
                        for category, feature in features.items():
                            results[category].append({
                                'embedding': feature.cpu().numpy(),
                                'style': style_label
                            })
                            valid_results = True
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
            
            if not valid_results:
                raise ValueError("No valid results were obtained from the dataset")
                    
            return results
            
        except Exception as e:
            print(f"Error in dataset analysis: {str(e)}")
            return None

    def analyze_feature_clusters(self, results, n_clusters=5):
        """Analyze how features cluster within each category"""
        if results is None:
            print("No results to analyze")
            return None
            
        cluster_analysis = {}
        
        for category, embeddings in results.items():
            if not embeddings:
                print(f"No embeddings for category {category}")
                continue
                
            try:
                # Stack all embeddings for this category
                X = np.stack([e['embedding'] for e in embeddings])
                labels = [e['style'] for e in embeddings]
                
                # Reduce dimensionality for clustering
                pca = PCA(n_components=min(50, X.shape[1]))
                X_reduced = pca.fit_transform(X)
                
                # Perform hierarchical clustering
                clustering = AgglomerativeClustering(n_clusters=n_clusters)
                cluster_labels = clustering.fit_predict(X_reduced)
                
                # Analyze cluster composition
                cluster_composition = defaultdict(lambda: defaultdict(int))
                for cluster_label, style_label in zip(cluster_labels, labels):
                    cluster_composition[cluster_label][style_label] += 1
                
                cluster_analysis[category] = {
                    'composition': dict(cluster_composition),
                    'pca_explained_variance': pca.explained_variance_ratio_.sum()
                }
                
            except Exception as e:
                print(f"Error analyzing clusters for {category}: {str(e)}")
                continue
        
        return cluster_analysis

    def visualize_features(self, results: dict, output_path: str):
        """Visualize feature embeddings using t-SNE"""
        if not results:
            print("No results to visualize")
            return
            
        num_categories = len(self.feature_categories)
        fig, axes = plt.subplots(2, (num_categories + 1) // 2, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, (category, embeddings) in enumerate(results.items()):
            if not embeddings:
                print(f"No embeddings for category {category}")
                continue
                
            if i >= len(axes):
                break
                
            try:
                # Extract data
                X = np.stack([e['embedding'] for e in embeddings])
                labels = [e['style'] for e in embeddings]
                
                # Reduce dimensionality first with PCA
                pca = PCA(n_components=min(50, X.shape[1]))
                X_reduced = pca.fit_transform(X)
                
                # Then apply t-SNE
                tsne = TSNE(n_components=2, random_state=42)
                X_2d = tsne.fit_transform(X_reduced)
                
                # Create unique colors for labels
                unique_labels = list(set(labels))
                color_map = {label: i for i, label in enumerate(unique_labels)}
                colors = [color_map[label] for label in labels]
                
                # Plot
                scatter = axes[i].scatter(X_2d[:, 0], X_2d[:, 1], 
                                        c=colors, cmap='tab20', alpha=0.6)
                axes[i].set_title(f"{category} Features")
                
                # Add legend to first plot only
                if i == 0:
                    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=plt.cm.tab20(color_map[label]/len(unique_labels)), 
                                                label=label, markersize=8)
                                     for label in unique_labels]
                    axes[i].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1))
                    
            except Exception as e:
                print(f"Error visualizing {category}: {str(e)}")
                continue
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    try:
        print("Initializing DINOStyleAnalyzer...")
        analyzer = DINOStyleAnalyzer()
        
        print("\nStarting dataset analysis...")
        dataset_path = "datasets/wikiArt/wikiart"  # Update this path
        results = analyzer.analyze_dataset(dataset_path, num_samples=10000)
        
        if results:
            print("\nAnalyzing feature clusters...")
            cluster_analysis = analyzer.analyze_feature_clusters(results)
            
            if cluster_analysis:
                print("\nCreating visualizations...")
                analyzer.visualize_features(results, "dino_style_analysis.png")
                
                print("\nAnalysis complete!")
                
                # Print cluster analysis results
                for category, analysis in cluster_analysis.items():
                    print(f"\nCategory: {category}")
                    print(f"PCA explained variance: {analysis['pca_explained_variance']:.2%}")
                    print("Cluster composition:")
                    for cluster, styles in analysis['composition'].items():
                        print(f"\nCluster {cluster}:")
                        sorted_styles = sorted(styles.items(), key=lambda x: x[1], reverse=True)
                        for style, count in sorted_styles:
                            print(f"  {style}: {count}")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")