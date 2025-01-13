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

from style_decomposition import StyleDecompositionAnalyzer


class HierarchicalStyleDecomposer(StyleDecompositionAnalyzer):

    def __init__(self, model_name: str = "ViT-B/32", batch_size: int = 32):
        super().__init__(model_name, batch_size)
        self.variance_thresholds = {
            'color': 0.03,  # Based on your analysis data
            'texture': 0.02,
            'composition': 0.01
        }

    def decompose_hierarchically(self, embeddings: torch.Tensor, labels: List[str]) -> Dict:
        """Hierarchical decomposition of style embeddings"""
        # Initial concept projections
        initial_projections = self.decompose_concept_space(embeddings)
        
        # Create root node
        tree = {
            'embeddings': embeddings,
            'projections': initial_projections,
            'children': {}
        }

        # First level: Split by most consistent aspect (composition)
        comp_projections = initial_projections['composition']
        comp_clusters = self._cluster_by_variance(comp_projections, self.variance_thresholds['composition'])

        for comp_idx, comp_mask in comp_clusters.items():
            cluster_embeddings = embeddings[comp_mask]
            cluster_projections = self.decompose_concept_space(cluster_embeddings)
            
            # Store first level nodes
            tree['children'][f'comp_{comp_idx}'] = {
                'embeddings': cluster_embeddings,
                'projections': cluster_projections,
                'children': {}
            }

            # Second level: Split by texture
            tex_clusters = self._cluster_by_variance(
                cluster_projections['texture'],
                self.variance_thresholds['texture']
            )

            for tex_idx, tex_mask in tex_clusters.items():
                sub_embeddings = cluster_embeddings[tex_mask]
                sub_projections = self.decompose_concept_space(sub_embeddings)
                
                # Store second level nodes
                tree['children'][f'comp_{comp_idx}']['children'][f'tex_{tex_idx}'] = {
                    'embeddings': sub_embeddings,
                    'projections': sub_projections,
                    'children': {}
                }

                # Third level: Split by color
                color_clusters = self._cluster_by_variance(
                    sub_projections['color'],
                    self.variance_thresholds['color']
                )

                for color_idx, color_mask in color_clusters.items():
                    leaf_embeddings = sub_embeddings[color_mask]
                    leaf_projections = self.decompose_concept_space(leaf_embeddings)
                    
                    # Store leaf nodes
                    tree['children'][f'comp_{comp_idx}']['children'][f'tex_{tex_idx}']['children'][f'color_{color_idx}'] = {
                        'embeddings': leaf_embeddings,
                        'projections': leaf_projections
                    }

        return tree

    def visualize_decomposition_tree(self, tree: Dict, output_path: str):
        """Visualize the hierarchical decomposition"""
        plt.figure(figsize=(20, 15))
        
        def plot_node(node, level: int, pos_x: float, pos_y: float):
            if 'projections' not in node:
                return
                
            # Create subplot for this node
            plt.subplot(level, 1, level)
            
            # Plot distributions for each concept space
            for concept, proj in node['projections'].items():
                proj_np = proj.cpu().numpy()
                plt.hist(proj_np.flatten(), bins=50, alpha=0.3, label=concept)
            
            plt.legend()
            plt.title(f"Level {level} Node Distribution")
            
            # Recursively plot children
            if 'children' in node:
                num_children = len(node['children'])
                for i, (child_name, child) in enumerate(node['children'].items()):
                    new_x = pos_x + (i - num_children/2) * (1.0/level)
                    new_y = pos_y - 1
                    plot_node(child, level + 1, new_x, new_y)
        
        # Start plotting from root
        plot_node(tree, 1, 0.5, 1.0)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

if __name__=='__main__':

    try:
        decomposer = HierarchicalStyleDecomposer()
        embeddings, paths, labels = decomposer.extract_embeddings("datasets/wikiArt/wikiart")
        tree = decomposer.decompose_hierarchically(embeddings, labels)
        decomposer.visualize_decomposition_tree(tree, "decomposition_tree.png")
        logger.info("Analysis completed successfully")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise