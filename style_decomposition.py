import torch
import clip
from PIL import Image
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import warnings
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
try:
    import umap.umap_ as umap
except ImportError:
    logger.warning("UMAP not available. Will use PCA for dimensionality reduction instead.")
    umap = None
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
import torch.nn.functional as F

# Configure logging and warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)

class StyleDecompositionAnalyzer:
    def __init__(self, model_name: str = "ViT-B/32", batch_size: int = 32):
        """
        Initialize the style decomposition analyzer with CLIP model.
        
        Args:
            model_name: CLIP model variant to use
            batch_size: Batch size for processing images
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.batch_size = batch_size
        
        # Move model to GPU
        self.model = self.model.to(self.device)
        
        # Define concept spaces
        self.concept_spaces = {
            'color': [
                ("warm colors", "cool colors"),
                ("vibrant colors", "muted colors"),
                ("light tones", "dark tones"),
                ("high contrast", "low contrast")
            ],
            'texture': [
                ("rough texture", "smooth texture"),
                ("regular patterns", "irregular patterns"),
                ("detailed surface", "simple surface"),
                ("organic texture", "geometric texture")
            ],
            'composition': [
                ("minimal composition", "complex composition"),
                ("symmetric layout", "asymmetric layout"),
                ("balanced composition", "dynamic composition"),
                ("structured composition", "fluid composition")
            ]
        }
        
        # Initialize concept embeddings
        self.concept_embeddings = self._initialize_concept_embeddings()

    def _initialize_concept_embeddings(self) -> Dict[str, torch.Tensor]:
        """Initialize embeddings for all concept pairs."""
        concept_embeddings = {}
        
        for category, concept_pairs in self.concept_spaces.items():
            category_embeddings = []
            
            for concept_a, concept_b in concept_pairs:
                tokens_a = clip.tokenize([f"an artwork with {concept_a}"]).to(self.device)
                tokens_b = clip.tokenize([f"an artwork with {concept_b}"]).to(self.device)
                
                with torch.no_grad():
                    embedding_a = self.model.encode_text(tokens_a)
                    embedding_b = self.model.encode_text(tokens_b)
                    
                    # Keep embeddings on GPU
                    embedding_a = F.normalize(embedding_a, dim=-1)
                    embedding_b = F.normalize(embedding_b, dim=-1)
                    
                    concept_axis = (embedding_a - embedding_b)
                    category_embeddings.append(concept_axis)
            
            # Stack tensors on GPU
            concept_embeddings[category] = torch.cat(category_embeddings, dim=0)
        
        return concept_embeddings

    def process_batch(self, image_batch: List[Image.Image]) -> torch.Tensor:
        """Process a batch of images and return their embeddings."""
        # Preprocess all images in the batch
        inputs = torch.stack([self.preprocess(img) for img in image_batch]).to(self.device)
        
        with torch.no_grad():
            # Get embeddings
            features = self.model.encode_image(inputs)
            # Normalize on GPU
            features = F.normalize(features, dim=-1)
        
        return features

    def extract_embeddings(self, dataset_path: str) -> Tuple[torch.Tensor, List[str], List[str]]:
        """Extract CLIP embeddings for all images in the dataset."""
        dataset_path = Path(dataset_path)
        image_paths = []
        embeddings_list = []
        labels = []
        
        # Collect all image files
        for ext in ('*.jpg', '*.png', '*.jpeg'):
            image_paths.extend(dataset_path.rglob(ext))
        
        logger.info(f"Processing {len(image_paths)} images...")
        
        # Process images in batches
        current_batch = []
        current_labels = []
        
        for img_path in tqdm(image_paths):
            try:
                image = Image.open(img_path).convert('RGB')
                current_batch.append(image)
                current_labels.append(img_path.parent.name)
                
                if len(current_batch) == self.batch_size:
                    batch_embeddings = self.process_batch(current_batch)
                    embeddings_list.append(batch_embeddings)
                    labels.extend(current_labels)
                    current_batch = []
                    current_labels = []
                    
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {str(e)}")
                continue
        
        # Process remaining images
        if current_batch:
            batch_embeddings = self.process_batch(current_batch)
            embeddings_list.append(batch_embeddings)
            labels.extend(current_labels)
        
        # Concatenate all embeddings on GPU
        embeddings = torch.cat(embeddings_list, dim=0)
        
        return embeddings, [str(p) for p in image_paths], labels
    
    def decompose_hierarchically(self, embeddings: torch.Tensor, labels: List[str]) -> Dict:
        """Hierarchical decomposition of style embeddings"""
        logger.info("Starting hierarchical decomposition...")
        
        # Initial concept projections
        logger.info("Computing initial projections...")
        initial_projections = self.decompose_concept_space(embeddings)
        
        # Create root node
        tree = {
            'embeddings': embeddings,
            'projections': initial_projections,
            'children': {}
        }

        # First level: Split by composition
        logger.info("Processing first level (composition)...")
        comp_projections = initial_projections['composition']
        comp_clusters = self._cluster_by_variance(comp_projections, self.variance_thresholds['composition'])
        logger.info(f"Found {len(comp_clusters)} composition clusters")

        for comp_idx, comp_mask in comp_clusters.items():
            logger.info(f"Processing composition cluster {comp_idx}")
            cluster_embeddings = embeddings[comp_mask]
            cluster_projections = self.decompose_concept_space(cluster_embeddings)
            
            tree['children'][f'comp_{comp_idx}'] = {
                'embeddings': cluster_embeddings,
                'projections': cluster_projections,
                'children': {}
            }

            # Second level: Split by texture
            logger.info(f"Processing texture splits for composition cluster {comp_idx}")
            tex_clusters = self._cluster_by_variance(
                cluster_projections['texture'],
                self.variance_thresholds['texture']
            )

            for tex_idx, tex_mask in tex_clusters.items():
                logger.info(f"Processing texture cluster {tex_idx}")
                sub_embeddings = cluster_embeddings[tex_mask]
                sub_projections = self.decompose_concept_space(sub_embeddings)
                
                tree['children'][f'comp_{comp_idx}']['children'][f'tex_{tex_idx}'] = {
                    'embeddings': sub_embeddings,
                    'projections': sub_projections,
                    'children': {}
                }

        logger.info("Hierarchical decomposition completed")
        return tree

    def _cluster_by_variance(self, projections: torch.Tensor, threshold: float) -> Dict[int, torch.Tensor]:
        """Cluster embeddings based on projection variance"""
        # Move to CPU for sklearn operations
        proj_np = projections.cpu().numpy()
        
        # Calculate variances along feature dimensions
        variances = np.var(proj_np, axis=1)
        
        # Determine number of clusters based on variance threshold
        n_clusters = max(2, int(np.sum(variances > threshold)))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(proj_np)
        
        # Create masks for each cluster
        cluster_masks = {}
        for i in range(n_clusters):
            # Convert back to tensor and move to same device as original projections
            mask = torch.tensor(clusters == i, device=projections.device)
            cluster_masks[i] = mask
        
        return cluster_masks

    def decompose_concept_space(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decompose embeddings into concept-specific subspaces."""
        concept_projections = {}
        
        for category, concept_axes in self.concept_embeddings.items():
            # Keep everything on GPU
            concept_axes = concept_axes.to(self.device)
            
            # Calculate projections using GPU operations
            projections = torch.mm(embeddings, concept_axes.t())
            concept_projections[category] = projections
        
        return concept_projections

    def analyze_concept_clusters(self, 
                               projections: Dict[str, torch.Tensor], 
                               labels: List[str], 
                               n_clusters: int = 5) -> Dict[str, Dict]:
        """Analyze clusters within each concept space."""
        results = {}
        
        for category, category_projections in projections.items():
            # Move to CPU for sklearn operations
            proj_np = category_projections.cpu().numpy()
            
            # Standardize projections
            scaler = StandardScaler()
            scaled_projections = scaler.fit_transform(proj_np)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_projections)
            
            # Analyze cluster composition
            cluster_composition = defaultdict(lambda: defaultdict(int))
            for style_label, cluster_label in zip(labels, cluster_labels):
                cluster_composition[f"cluster_{cluster_label}"][style_label] += 1
            
            results[category] = {
                'cluster_composition': dict(cluster_composition),
                'cluster_centers': kmeans.cluster_centers_,
                'cluster_labels': cluster_labels
            }
        
        return results

    def visualize_concept_space(self, 
                              projections: Dict[str, torch.Tensor],
                              labels: List[str],
                              category: str,
                              output_path: Optional[str] = None,
                              method: str = 'umap'):
        """
        Visualize concept space using dimensionality reduction.
        
        Args:
            projections: Dictionary of concept projections
            labels: List of style labels
            category: Concept category to visualize
            output_path: Path to save visualization
            method: Dimensionality reduction method ('umap' or 'pca')
        """
        # Move data to CPU for processing
        proj_np = projections[category].cpu().numpy()
        
        # Reduce dimensionality
        if method == 'umap' and umap is not None:
            try:
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
                embedding_2d = reducer.fit_transform(proj_np)
                method_name = "UMAP"
            except Exception as e:
                logger.warning(f"UMAP failed, falling back to PCA: {e}")
                method = 'pca'
        
        if method == 'pca':
            pca = PCA(n_components=2, random_state=42)
            embedding_2d = pca.fit_transform(proj_np)
            method_name = "PCA"
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                            c=[hash(label) for label in labels],
                            cmap='tab20', alpha=0.6)
        
        unique_labels = list(set(labels))
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=plt.cm.tab20(i/len(unique_labels)),
                                    label=label, markersize=10)
                         for i, label in enumerate(unique_labels)]
        
        plt.legend(handles=legend_elements, title="Styles",
                  bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title(f"{category.capitalize()} Concept Space ({method_name} projection)")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
        
        plt.close()

    def analyze_style_transitions(self, 
                                projections: Dict[str, torch.Tensor],
                                labels: List[str]) -> Dict[str, pd.DataFrame]:
        """Analyze style transitions in concept spaces."""
        transition_analysis = {}
        
        for category, category_projections in projections.items():
            # Move to CPU for pandas operations
            proj_np = category_projections.cpu().numpy()
            
            df = pd.DataFrame(proj_np, columns=[f"axis_{i}" for i in range(proj_np.shape[1])])
            df['style'] = labels
            
            style_centroids = df.groupby('style').mean()
            
            n_styles = len(style_centroids)
            distance_matrix = np.zeros((n_styles, n_styles))
            styles = style_centroids.index
            
            for i, style1 in enumerate(styles):
                for j, style2 in enumerate(styles):
                    if i < j:
                        dist = np.linalg.norm(style_centroids.loc[style1] - style_centroids.loc[style2])
                        distance_matrix[i, j] = dist
                        distance_matrix[j, i] = dist
            
            transition_analysis[category] = pd.DataFrame(distance_matrix, 
                                                       index=styles, 
                                                       columns=styles)
        
        return transition_analysis

def run_style_decomposition(dataset_path: str, output_path: str, batch_size: int = 32, viz_method: str = 'umap'):
    """Run complete style decomposition analysis."""
    # Initialize analyzer with batch processing
    analyzer = StyleDecompositionAnalyzer(batch_size=batch_size)
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract embeddings (now using GPU)
    embeddings, image_paths, labels = analyzer.extract_embeddings(dataset_path)
    
    # Decompose into concept spaces (GPU operations)
    concept_projections = analyzer.decompose_concept_space(embeddings)
    
    # Analyze clusters
    cluster_analysis = analyzer.analyze_concept_clusters(concept_projections, labels)
    
    # Analyze style transitions
    transition_analysis = analyzer.analyze_style_transitions(concept_projections, labels)
    
    # Generate visualizations
    for category in analyzer.concept_spaces.keys():
        viz_path = output_dir / f"{category}_concept_space.png"
        analyzer.visualize_concept_space(concept_projections, labels, category, str(viz_path), method=viz_method)
        
        # Save transition analysis
        transition_analysis[category].to_csv(output_dir / f"{category}_transitions.csv")
    
    # Save embeddings (convert to numpy for saving)
    np.save(str(output_dir / "embeddings.npy"), embeddings.cpu().numpy())
    
    return {
        'embeddings': embeddings,
        'concept_projections': concept_projections,
        'cluster_analysis': cluster_analysis,
        'transition_analysis': transition_analysis
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run style decomposition analysis")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    
    args = parser.parse_args()
    
    try:
        results = run_style_decomposition(args.dataset, args.output, args.batch_size)
        logger.info("Analysis completed successfully")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise