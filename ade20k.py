import torch
import clip
from PIL import Image
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import shutil
import logging
import random
import json

class CLIPConceptSetGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        logging.info(f"Using device: {device}")
        
        # Load CLIP model
        logging.info("Loading CLIP model...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        logging.info("CLIP model loaded successfully")
        
    def compute_text_embedding(self, text_prompt):
        """Compute CLIP embedding for a text prompt"""
        text_token = clip.tokenize([text_prompt]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_token)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def compute_image_embedding(self, image_path):
        """Compute CLIP embedding for an image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            return image_features
            
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def compute_similarity(self, image_embedding, text_embedding):
        """Compute similarity between image and text embeddings"""
        with torch.no_grad():
            similarity = (image_embedding @ text_embedding.T).item()
        return similarity
    
    def find_segmentation_mask(self, image_path):
        """Find corresponding segmentation mask for an image path in ADE20K structure"""
        # Convert to Path object for manipulation
        img_path = Path(image_path)
        img_filename = img_path.name
        
        # First, try direct path substitution (works on both Windows and Unix paths)
        mask_path = str(img_path).replace("images", "annotations")
        if os.path.exists(mask_path):
            return mask_path
        
        # If the mask file has a different extension (e.g., .png instead of .jpg)
        if img_filename.lower().endswith('.jpg'):
            png_mask_path = mask_path[:-4] + '.png'  # Replace .jpg extension with .png
            if os.path.exists(png_mask_path):
                return png_mask_path
        
        # Try with stem only (in case of extension differences)
        img_stem = img_path.stem  # Get filename without extension
        
        # Reconstruct path components manually
        path_parts = str(img_path).replace('\\', '/').split('/')
        
        # Try to find relevant directory markers
        try:
            if 'ade20k' in path_parts and 'images' in path_parts and 'training' in path_parts:
                # Get the base structure
                ade_idx = path_parts.index('ade20k')
                base_path = '/'.join(path_parts[:ade_idx+1])  # Include up to ade20k
                
                # Check for mask with same name but .png extension
                mask_path_png = f"{base_path}/annotations/training/{img_stem}.png"
                if os.path.exists(mask_path_png):
                    return mask_path_png
                
                # Check for mask with original extension
                mask_path_orig = f"{base_path}/annotations/training/{img_filename}"
                if os.path.exists(mask_path_orig):
                    return mask_path_orig
        except Exception as e:
            logging.debug(f"Error while trying alternative mask paths: {str(e)}")
        
        # Additional debug info
        logging.debug(f"Image path: {image_path}")
        logging.debug(f"Tried mask path: {mask_path}")
        if img_filename.lower().endswith('.jpg'):
            logging.debug(f"Tried PNG mask path: {png_mask_path}")
            
        # Print the directory listing to help diagnose the issue
        try:
            # Get the parent directory of the image
            img_dir = os.path.dirname(image_path)
            # Construct the expected mask directory
            mask_dir = img_dir.replace("images", "annotations")
            
            if os.path.exists(mask_dir):
                logging.debug(f"Mask directory exists: {mask_dir}")
                # List a few files in the mask directory to verify naming pattern
                mask_files = os.listdir(mask_dir)[:5]
                logging.debug(f"Sample mask files: {mask_files}")
        except Exception as e:
            logging.debug(f"Error listing mask directory: {str(e)}")
            
        return None
    
    def generate_concept_set(self, 
                            image_dir, 
                            positive_prompt, 
                            negative_prompt,
                            output_dir,
                            num_images=15,
                            sample_size=None,
                            copy_masks=True):
        """
        Generate positive and negative concept sets based on CLIP similarity
        
        Args:
            image_dir: Directory containing input images
            positive_prompt: Text prompt for positive concept
            negative_prompt: Text prompt for negative concept
            output_dir: Directory to save concept sets
            num_images: Number of images to select for each class
            sample_size: Number of images to sample from dataset (None = use all)
            copy_masks: Whether to copy segmentation masks if available
        """
        # Create output directories
        positive_dir = os.path.join(output_dir, 'positive')
        negative_dir = os.path.join(output_dir, 'negative')
        os.makedirs(positive_dir, exist_ok=True)
        os.makedirs(negative_dir, exist_ok=True)
        
        # Create directories for masks if needed
        if copy_masks:
            positive_masks_dir = os.path.join(output_dir, 'positive_masks')
            negative_masks_dir = os.path.join(output_dir, 'negative_masks')
            os.makedirs(positive_masks_dir, exist_ok=True)
            os.makedirs(negative_masks_dir, exist_ok=True)
        
        # Compute text embeddings
        logging.info(f"Computing embeddings for prompts: '{positive_prompt}' and '{negative_prompt}'")
        positive_text_embedding = self.compute_text_embedding(positive_prompt)
        negative_text_embedding = self.compute_text_embedding(negative_prompt)
        
        # Get list of image paths
        image_paths = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            image_paths.extend(list(Path(image_dir).glob(f"**/{ext}")))
        
        logging.info(f"Found {len(image_paths)} images in {image_dir}")
        
        # Sample subset if requested
        if sample_size is not None and sample_size < len(image_paths):
            logging.info(f"Sampling {sample_size} images from dataset")
            image_paths = random.sample(image_paths, sample_size)
        
        # Compute image embeddings and similarities
        logging.info("Computing image embeddings and similarities...")
        similarities = []
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            image_embedding = self.compute_image_embedding(img_path)
            if image_embedding is not None:
                pos_similarity = self.compute_similarity(image_embedding, positive_text_embedding)
                neg_similarity = self.compute_similarity(image_embedding, negative_text_embedding)
                
                # Find segmentation mask if requested
                mask_path = None
                if copy_masks:
                    mask_path = self.find_segmentation_mask(img_path)
                
                # Store path and similarities
                similarities.append({
                    'path': str(img_path),
                    'mask_path': mask_path,
                    'positive_similarity': pos_similarity,
                    'negative_similarity': neg_similarity,
                    'contrast': pos_similarity - neg_similarity  # Higher = more aligned with positive
                })
        
        # Sort by contrast (difference between positive and negative similarity)
        similarities_sorted = sorted(similarities, key=lambda x: x['contrast'], reverse=True)
        
        # Select top images for positive class
        positive_images = similarities_sorted[:num_images]
        
        # Select bottom images for negative class
        negative_images = similarities_sorted[-num_images:]
        
        # Copy images and masks to output directories
        logging.info(f"Copying {num_images} positive and {num_images} negative images to {output_dir}")
        
        # Track mask statistics
        mask_stats = {
            'positive_masks_found': 0,
            'negative_masks_found': 0
        }
        
        for idx, img_data in enumerate(positive_images):
            # Copy image
            src_path = img_data['path']
            filename = os.path.basename(src_path)
            dst_path = os.path.join(positive_dir, f"{idx:03d}_{filename}")
            shutil.copy2(src_path, dst_path)
            
            # Copy mask if available
            if copy_masks and img_data['mask_path']:
                mask_src_path = img_data['mask_path']
                mask_filename = os.path.basename(mask_src_path)
                mask_dst_path = os.path.join(positive_masks_dir, f"{idx:03d}_{mask_filename}")
                shutil.copy2(mask_src_path, mask_dst_path)
                mask_stats['positive_masks_found'] += 1
                img_data['mask_copied'] = True
            else:
                img_data['mask_copied'] = False
        
        for idx, img_data in enumerate(negative_images):
            # Copy image
            src_path = img_data['path']
            filename = os.path.basename(src_path)
            dst_path = os.path.join(negative_dir, f"{idx:03d}_{filename}")
            shutil.copy2(src_path, dst_path)
            
            # Copy mask if available
            if copy_masks and img_data['mask_path']:
                mask_src_path = img_data['mask_path']
                mask_filename = os.path.basename(mask_src_path)
                mask_dst_path = os.path.join(negative_masks_dir, f"{idx:03d}_{mask_filename}")
                shutil.copy2(mask_src_path, mask_dst_path)
                mask_stats['negative_masks_found'] += 1
                img_data['mask_copied'] = True
            else:
                img_data['mask_copied'] = False
        
        # Save metadata
        metadata = {
            'positive_prompt': positive_prompt,
            'negative_prompt': negative_prompt,
            'positive_images': positive_images,
            'negative_images': negative_images,
            'mask_statistics': mask_stats,
            'statistics': {
                'positive_mean_similarity': np.mean([img['positive_similarity'] for img in positive_images]),
                'negative_mean_similarity': np.mean([img['negative_similarity'] for img in negative_images]),
                'positive_mean_contrast': np.mean([img['contrast'] for img in positive_images]),
                'negative_mean_contrast': np.mean([img['contrast'] for img in negative_images])
            }
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logging.info(f"Concept set generation completed: {output_dir}")
        logging.info(f"Mask statistics: Found {mask_stats['positive_masks_found']} positive masks and {mask_stats['negative_masks_found']} negative masks")
        
        return metadata

def generate_ade20k_concept_set(concept_name, positive_prompt, negative_prompt, ade20k_dir, output_base_dir='concept_sets', sample_size=500, copy_masks=True):
    """Generate a concept set from ADE20K dataset"""
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info(f"Generating concept set for '{concept_name}'")
    
    # Create generator
    generator = CLIPConceptSetGenerator()
    
    # Create output directory
    output_dir = os.path.join(output_base_dir, concept_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate concept set
    metadata = generator.generate_concept_set(
        image_dir=ade20k_dir,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        output_dir=output_dir,
        num_images=15,
        sample_size=sample_size,
        copy_masks=copy_masks
    )
    
    logging.info(f"Concept set generation for '{concept_name}' completed")
    return metadata

if __name__ == "__main__":
    # Example usage:
    generate_ade20k_concept_set(
        concept_name="day_vs_night",
        positive_prompt="a bright day with clear blue sky",
        negative_prompt="a dark night with starry sky or city lights",
        ade20k_dir="datasets/ade20k/images/training/",
        sample_size=5000
    )



# Example usage:
# generate_ade20k_concept_set(
#     concept_name="sunny_vs_cloudy",
#     positive_prompt="a sunny day with clear blue sky",
#     negative_prompt="a cloudy overcast day with gray sky",
#     ade20k_dir="/path/to/ade20k/images",
#     sample_size=500
# )