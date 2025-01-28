import torch
import numpy as np
from sklearn.decomposition import PCA
import clip
from PIL import Image
import torch.nn.functional as F

class WarmDirectionFinder:
    def __init__(self, stylegan_model, device='cuda'):
        self.G = stylegan_model
        self.device = device
        
        # Load CLIP for warmth evaluation
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        
        # Initialize warmth prompts
        self.init_warmth_prompts()
    
    def init_warmth_prompts(self):
        """Initialize CLIP prompts for different types of warmth"""
        self.warmth_categories = {
            'golden': [
                "a painting with warm golden colors",
                "artwork with golden sunlight",
                "warm golden tones in art"
            ],
            'sunset': [
                "a painting with warm sunset colors",
                "artwork with warm orange and red tones",
                "sunset color palette in art"
            ],
            'autumn': [
                "a painting with warm autumn colors",
                "artwork with fall colors",
                "warm earthy tones in art"
            ]
        }
        
        # Encode all prompts
        self.encoded_prompts = {}
        for category, prompts in self.warmth_categories.items():
            with torch.no_grad():
                text = clip.tokenize(prompts).to(self.device)
                text_features = self.clip_model.encode_text(text)
                text_features = F.normalize(text_features, dim=-1)
                self.encoded_prompts[category] = text_features

    def generate_samples(self, n_samples=1000):
        """Generate random samples and evaluate their warmth"""
        z_samples = np.random.RandomState(42).randn(n_samples, self.G.z_dim)
        w_samples = []
        warmth_scores = {category: [] for category in self.warmth_categories.keys()}
        
        for z in tqdm(z_samples, desc="Generating samples"):
            # Generate image
            z_tensor = torch.from_numpy(z).unsqueeze(0).to(self.device)
            w = self.G.mapping(z_tensor, None)
            img = self.G.synthesis(w)
            
            # Evaluate warmth with CLIP
            img_processed = self.preprocess_for_clip(img)
            with torch.no_grad():
                img_features = self.clip_model.encode_image(img_processed)
                img_features = F.normalize(img_features, dim=-1)
                
                # Calculate scores for each category
                for category, text_features in self.encoded_prompts.items():
                    similarity = (100 * img_features @ text_features.T).mean()
                    warmth_scores[category].append(similarity.item())
            
            w_samples.append(w.cpu().numpy())
        
        return np.concatenate(w_samples), warmth_scores
    
    def find_warm_directions(self):
        """Find principal directions for each warmth category"""
        # Generate samples and evaluate warmth
        w_samples, warmth_scores = self.generate_samples()
        
        directions = {}
        for category in self.warmth_categories.keys():
            # Select top warm samples
            scores = np.array(warmth_scores[category])
            top_indices = np.argsort(scores)[-100:]  # Top 100 warmest samples
            warm_samples = w_samples[top_indices]
            
            # Perform PCA on warm samples
            pca = PCA(n_components=1)
            pca.fit(warm_samples.reshape(len(warm_samples), -1))
            
            # The first principal component represents the warm direction
            warm_direction = pca.components_[0].reshape(w_samples.shape[1:])
            directions[category] = warm_direction
        
        return directions
    
    def preprocess_for_clip(self, img_tensor):
        """Preprocess generated image for CLIP"""
        # Convert from StyleGAN range to [0, 1]
        img = (img_tensor + 1) / 2
        # Resize and normalize for CLIP
        return F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)

def discover_warm_directions(network_pkl, output_dir):



    
    """Main function to discover and save warm directions"""
    # Load StyleGAN
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].cuda()
    
    # Find directions
    finder = WarmDirectionFinder(G)
    directions = finder.find_warm_directions()
    
    # Save directions
    os.makedirs(output_dir, exist_ok=True)
    for category, direction in directions.items():
        np.save(os.path.join(output_dir, f'warm_{category}.npy'), direction)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Discover warm directions in StyleGAN latent space')
    parser.add_argument('--network_pkl', type=str, required=True, help='Path to StyleGAN model')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for directions')
    
    args = parser.parse_args()
    
    discover_warm_directions(args.network_pkl, args.output_dir)