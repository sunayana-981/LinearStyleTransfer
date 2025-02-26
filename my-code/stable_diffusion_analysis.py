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

class SDStyleAnalyzer:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load SD components separately for efficiency
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

        # Put models in eval mode
        self.vae.eval()
        self.text_encoder.eval()

        self.prompt_strategies = {
            'artistic': [
                "a painting in the style of {style}",
                "artwork inspired by {style} movement",
                "an image showing {style} characteristics"
            ],
            'technical': [
                "art technique analysis of {style}",
                "visual elements of {style}",
                "{style} artistic method"
            ],
            'contextual': [
                "historical context of {style}",
                "{style} period artwork",
                "cultural elements of {style}"
            ]
        }

    @torch.no_grad()
    def get_image_latents(self, image: Image) -> torch.Tensor:
        """Get VAE latents from image"""
        # Preprocess image
        image = image.convert("RGB")
        image = image.resize((512, 512))
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Get VAE latents
        latents = self.vae.encode(image).latent_dist.mean
        return latents

    @torch.no_grad()
    def get_text_embeddings(self, prompts: list) -> torch.Tensor:
        """Get text embeddings for prompts"""
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        text_embeddings = self.text_encoder(text_input.input_ids)[0]
        return text_embeddings

    def analyze_style_embeddings(self, dataset_path: str, num_samples: int = 100):
        dataset_path = Path(dataset_path)
        results = {strategy: [] for strategy in self.prompt_strategies}
        
        # Collect image paths
        image_paths = []
        for ext in ('*.jpg', '*.png', '*.jpeg'):
            image_paths.extend(dataset_path.rglob(ext))

        # Sample images
        sampled_paths = np.random.choice(image_paths, num_samples, replace=False)
        
        for img_path in tqdm(sampled_paths):
            try:
                image = Image.open(img_path).convert('RGB')
                style_label = img_path.parent.name
                
                # Get image latents - reshape to ensure correct dimensions
                image_latents = self.get_image_latents(image)  # [1, 4, 64, 64]
                image_features = image_latents.reshape(1, -1)  # [1, 16384]
                
                # Process each prompting strategy
                for strategy, prompts in self.prompt_strategies.items():
                    # Generate prompts with style
                    formatted_prompts = [p.format(style=style_label) for p in prompts]
                    text_embeddings = self.get_text_embeddings(formatted_prompts)  # [num_prompts, seq_len, hidden_dim]
                    
                    # Average across prompts and sequence length
                    text_features = text_embeddings.mean(dim=[0, 1]).unsqueeze(0)  # [1, hidden_dim]
                    
                    # Combine image and text features
                    combined_embedding = torch.cat([
                        image_features,
                        text_features
                    ], dim=1)  # [1, 16384 + hidden_dim]
                    
                    results[strategy].append({
                        'embedding': combined_embedding.squeeze(0).cpu().numpy(),
                        'style': style_label
                    })
                    
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue

        return results

    @torch.no_grad()
    def get_image_latents(self, image: Image) -> torch.Tensor:
        """Get VAE latents from image"""
        try:
            # Preprocess image
            image = image.convert("RGB")
            image = image.resize((512, 512))
            image = torch.from_numpy(np.array(image)).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)

            print(f"Image tensor shape: {image.shape}")
            
            # Get VAE latents
            latents = self.vae.encode(image).latent_dist.mean
            print(f"Latents shape: {latents.shape}")
            
            return latents
        except Exception as e:
            print(f"Error in get_image_latents: {str(e)}")
            raise

    @torch.no_grad()
    def get_text_embeddings(self, prompts: list) -> torch.Tensor:
        """Get text embeddings for prompts"""
        try:
            text_input = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            print(f"Tokenizer output shape: {text_input.input_ids.shape}")
            
            text_embeddings = self.text_encoder(text_input.input_ids)[0]
            print(f"Text embeddings shape: {text_embeddings.shape}")
            
            return text_embeddings
        except Exception as e:
            print(f"Error in get_text_embeddings: {str(e)}")
            raise

    def visualize_embeddings(self, results: dict, output_path: str):
        """Visualize embeddings using t-SNE"""
        fig, axes = plt.subplots(1, len(self.prompt_strategies), figsize=(20, 6))
        
        for i, (strategy, embeddings) in enumerate(results.items()):
            # Extract data
            X = np.stack([e['embedding'] for e in embeddings])
            labels = [e['style'] for e in embeddings]
            
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            X_2d = tsne.fit_transform(X)
            
            # Create unique colors for labels
            unique_labels = list(set(labels))
            color_map = {label: i for i, label in enumerate(unique_labels)}
            colors = [color_map[label] for label in labels]
            
            # Plot
            scatter = axes[i].scatter(X_2d[:, 0], X_2d[:, 1], 
                                    c=colors, cmap='tab20', alpha=0.6)
            axes[i].set_title(f"{strategy} Embeddings")
            
            # Add legend
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=plt.cm.tab20(color_map[label]/len(unique_labels)), 
                                        label=label, markersize=8)
                             for label in unique_labels]
            axes[i].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1))
            
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

# Usage
if __name__ == "__main__":
    analyzer = SDStyleAnalyzer()
    results = analyzer.analyze_style_embeddings("datasets/wikiArt/wikiart", num_samples=100)
    analyzer.visualize_embeddings(results, "sd_style_analysis.png")