import torch
import clip
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import logging
import numpy as np

class CleanWarmthController:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # Define prompts for warmth evaluation
        self.warm_prompt = "an artwork with warm colors"
        self.cool_prompt = "an artwork with cool colors"
        
        # Encode prompts
        with torch.no_grad():
            text = clip.tokenize([self.warm_prompt, self.cool_prompt]).to(device)
            self.text_features = self.model.encode_text(text)
            self.text_features = F.normalize(self.text_features, dim=-1)

    def get_warmth_score(self, image_tensor):
        """Calculate warmth score using CLIP"""
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = F.normalize(image_features, dim=-1)
            similarities = 100 * (image_features @ self.text_features.T)
            # Return difference between warm and cool scores
            return similarities[0, 0] - similarities[0, 1]

    def apply_color_transform(self, image, transform_params):
        """Apply color transformation while preserving image structure"""
        # Convert PIL to tensor
        img_tensor = TF.to_tensor(image)
        
        # Unpack transform parameters
        temperature, saturation = transform_params
        
        # Apply temperature adjustment (warm colors up, cool colors down)
        r, g, b = img_tensor.unbind(0)
        r_out = torch.clamp(r * (1 + 0.02 * temperature), 0, 1)
        b_out = torch.clamp(b * (1 - 0.01 * temperature), 0, 1)
        g_out = torch.clamp(g * (1 + 0.01 * temperature), 0, 1)  # Slight green adjustment
        
        # Adjust saturation
        img_adjusted = torch.stack([r_out, g_out, b_out])
        # Convert to HSV-like space for saturation adjustment
        max_rgb = torch.max(img_adjusted, dim=0)[0]
        min_rgb = torch.min(img_adjusted, dim=0)[0]
        delta = max_rgb - min_rgb
        
        # Adjust saturation while preserving luminance
        if saturation > 0:
            for c in range(3):
                img_adjusted[c] = torch.where(
                    max_rgb > 0,
                    max_rgb - (max_rgb - img_adjusted[c]) * (1 - saturation),
                    img_adjusted[c]
                )
        
        return TF.to_pil_image(img_adjusted)

    def optimize_warmth(self, image, num_steps=10):
        """Find optimal color transformation parameters"""
        # Initial parameters: [temperature, saturation]
        params = torch.tensor([0.0, 0.0], device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([params], lr=0.5)
        
        best_score = float('-inf')
        best_params = None
        best_image = None
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Apply current transformation
            transformed_image = self.apply_color_transform(image, params.cpu().detach().numpy())
            
            # Get CLIP score
            img_tensor = self.preprocess(transformed_image).unsqueeze(0).to(self.device)
            warmth_score = self.get_warmth_score(img_tensor)
            
            # We want to maximize warmth while keeping parameters in check
            loss = -warmth_score + 0.1 * torch.sum(params ** 2)  # Small regularization
            loss.backward()
            
            optimizer.step()
            
            # Keep track of best result
            if warmth_score > best_score:
                best_score = warmth_score
                best_params = params.clone().detach()
                best_image = transformed_image
            
            # Clip parameters to reasonable ranges
            with torch.no_grad():
                params[0].clamp_(-10, 10)  # Temperature range
                params[1].clamp_(-0.5, 0.5)  # Saturation range
            
            logging.info(f"Step {step}: Score = {warmth_score:.2f}, "
                        f"Temp = {params[0].item():.2f}, "
                        f"Sat = {params[1].item():.2f}")
        
        return best_image, best_score.item(), best_params.cpu().numpy()

    def create_warmth_variations(self, image, num_variations=5):
        """Create variations with different warmth levels"""
        # Get optimal parameters first
        _, _, best_params = self.optimize_warmth(image)
        
        variations = []
        scores = []
        
        # Create variations around the optimal parameters
        variation_scales = np.linspace(0.5, 1.5, num_variations)
        for scale in variation_scales:
            params = best_params * scale
            transformed = self.apply_color_transform(image, params)
            
            # Calculate score
            img_tensor = self.preprocess(transformed).unsqueeze(0).to(self.device)
            score = self.get_warmth_score(img_tensor)
            
            variations.append(transformed)
            scores.append(score.item())
        
        return variations, scores