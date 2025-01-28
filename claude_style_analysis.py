import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from anthropic import Anthropic
from collections import defaultdict
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StyleConceptAnalyzer:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", anthropic_key=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        logger.info("Loading Stable Diffusion components...")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        
        self.vae.eval()
        self.text_encoder.eval()
        
        # Setup Claude
        if anthropic_key:
            self.claude = Anthropic(api_key=anthropic_key)
        else:
            raise ValueError("Anthropic API key is required")
        
        # Initialize cache
        self.description_cache = {}
        self.cache_file = "style_descriptions_cache.json"
        self._load_cache()
        
        # Define style aspects
        self.style_aspects = ['color', 'technique', 'composition', 'distinctive']

    def _load_cache(self):
        """Load cached style descriptions"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.description_cache = json.load(f)
                logger.info(f"Loaded {len(self.description_cache)} cached style descriptions")
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
                self.description_cache = {}

    def _save_cache(self):
        """Save style descriptions to cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.description_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")

    def get_style_description(self, style_name: str) -> dict:
        """Get detailed style description using Claude"""
        if style_name in self.description_cache:
            return self.description_cache[style_name]

        prompt = f"""You are an art historian analyzing the artistic style '{style_name}'.
        Provide a structured analysis focusing on:

        1. Color: Specific color palettes and relationships
        2. Technique: Brushwork, mark-making, and artistic methods
        3. Composition: Arrangement, space use, and structural elements
        4. Distinctive: Unique identifying features of this style

        Format as JSON with these exact keys:
        {{
            "color": ["3-4 specific color phrases"],
            "technique": ["3-4 specific technique phrases"],
            "composition": ["3-4 specific composition phrases"],
            "distinctive": ["3-4 specific distinctive features"]
        }}

        Keep phrases focused and specific. Provide only the JSON structure with no additional text."""

        try:
            message = self.claude.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            description = json.loads(message.content[0].text)
            self.description_cache[style_name] = description
            self._save_cache()
            return description
            
        except Exception as e:
            logger.error(f"Error getting description for {style_name}: {e}")
            return {aspect: ["error getting description"] for aspect in self.style_aspects}

    @torch.no_grad()
    def extract_concept_embedding(self, prompt: str) -> torch.Tensor:
        """Extract concept embedding from Stable Diffusion"""
        try:
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            embeddings = self.text_encoder(text_input.input_ids)[0]
            embeddings = embeddings.mean(dim=1)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error extracting concept embedding: {e}")
            return torch.zeros((1, 768), device=self.device)  # Return zero embedding on error

    def analyze_style(self, style_name: str) -> Dict[str, torch.Tensor]:
        """Analyze style using Claude descriptions and SD concept extraction"""
        logger.info(f"Analyzing style: {style_name}")
        
        style_aspects = self.get_style_description(style_name)
        concept_embeddings = {}
        
        for aspect_type, phrases in style_aspects.items():
            embeddings = []
            for phrase in phrases:
                prompt = f"artwork with {phrase} in {style_name} style"
                embedding = self.extract_concept_embedding(prompt)
                embeddings.append(embedding)
            
            if embeddings:
                concept_embeddings[aspect_type] = torch.stack(embeddings).mean(dim=0)
            else:
                logger.warning(f"No embeddings generated for {aspect_type} in {style_name}")
                concept_embeddings[aspect_type] = torch.zeros((768,), device=self.device)
        
        return concept_embeddings

    def compute_similarity_matrix(self, embeddings_dict: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, np.ndarray]:
        """Compute similarity matrices for each aspect"""
        styles = list(embeddings_dict.keys())
        n_styles = len(styles)
        
        similarity_matrices = {}
        for aspect in self.style_aspects:
            matrix = np.zeros((n_styles, n_styles))
            
            for i, style1 in enumerate(styles):
                for j, style2 in enumerate(styles):
                    if i != j:
                        sim = torch.nn.functional.cosine_similarity(
                            embeddings_dict[style1][aspect].unsqueeze(0),
                            embeddings_dict[style2][aspect].unsqueeze(0)
                        )
                        matrix[i, j] = sim.item()
            
            similarity_matrices[aspect] = matrix
        
        return similarity_matrices, styles

    def visualize_similarities(self, matrices: Dict[str, np.ndarray], styles: List[str], output_dir: str):
        """Create visualizations for style similarities"""
        os.makedirs(output_dir, exist_ok=True)
        
        for aspect, matrix in matrices.items():
            plt.figure(figsize=(15, 12))
            sns.heatmap(
                matrix,
                xticklabels=styles,
                yticklabels=styles,
                cmap='coolwarm',
                center=0,
                annot=True,
                fmt='.2f'
            )
            plt.title(f'Style Similarity - {aspect.capitalize()}')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'similarity_{aspect}.png'), bbox_inches='tight', dpi=300)
            plt.close()

def main(args):
    analyzer = StyleConceptAnalyzer(anthropic_key=args.anthropic_key)
    
    # Process dataset
    dataset_path = Path(args.dataset)
    styles = [d.name for d in dataset_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(styles)} styles in dataset")
    
    # Analyze styles
    style_embeddings = {}
    for style in tqdm(styles, desc="Analyzing styles"):
        style_embeddings[style] = analyzer.analyze_style(style)
    
    # Compute and visualize similarities
    matrices, style_names = analyzer.compute_similarity_matrix(style_embeddings)
    analyzer.visualize_similarities(matrices, style_names, args.output)
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze artistic styles using Claude and Stable Diffusion")
    parser.add_argument("--dataset", type=str, required=True, help="Path to WikiArt dataset")
    parser.add_argument("--anthropic-key", type=str, default= , required=True, help="Anthropic API key")
    parser.add_argument("--output", type=str, default="style_analysis_output", help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise