import torch
import clip
import os
from PIL import Image, ImageFile
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
from tqdm import tqdm
import shutil

# Allow truncated image loading (fixes the OSError)
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # Remove PIL image size safety limit

class ImageDataset(Dataset):
    def __init__(self, image_dir, preprocess):
        self.image_paths = [p for p in Path(image_dir).glob("*") if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')  # Ensure 3-channel image
                return self.preprocess(img), str(image_path)
        except (OSError, IOError):
            print(f"Skipping corrupted image: {image_path}")
            return None  # Return None for bad images

# Custom collate function to filter out None values from the dataloader
def collate_fn(batch):
    batch = [b for b in batch if b is not None]  # Remove None values
    if len(batch) == 0:
        return None  # Handle the case where all images in a batch are corrupted
    return tuple(zip(*batch))  # Unzip the batch

def create_concept_sets(image_dir, output_dir, batch_size=32, top_k=50, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Create concept sets from images based on CLIP embeddings and prompt similarities.
    
    Args:
        image_dir (str): Directory containing randomly sampled images
        output_dir (str): Directory to save concept sets
        batch_size (int): Batch size for processing images
        top_k (int): Number of top images to select for each concept
        device (str): Device to run CLIP on
    """
    # Initialize CLIP
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()  # Set model to evaluation mode
    
    # Define concept prompts
    concepts = {
        "color": [
            ("warm colors", "cool colors"),
            ("vibrant colors", "muted colors")
        ],
        "texture": [
            ("rough texture", "smooth texture"),
            ("regular patterns", "irregular patterns")
        ],
        "composition": [
            ("balanced composition", "dynamic composition"),
            ("minimal composition", "complex composition")
        ]
    }
    
    # Create dataset and dataloader
    dataset = ImageDataset(image_dir, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # Process all images
    image_embeddings = []
    image_paths = []
    
    print("Computing image embeddings...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batch is None:
                continue  # Skip batches where all images are corrupted
            
            batch_images, batch_paths = batch
            batch_images = torch.stack(batch_images).to(device)  # Stack tensor images
            
            batch_embeddings = model.encode_image(batch_images)
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)  # Normalize embeddings
            
            image_embeddings.append(batch_embeddings.cpu())
            image_paths.extend(batch_paths)

    image_embeddings = torch.cat(image_embeddings, dim=0)

    # Create output directories
    output_dir = Path(output_dir)
    for category in concepts.keys():
        (output_dir / category).mkdir(parents=True, exist_ok=True)
    
    # Process each concept pair
    print("\nCreating concept sets...")
    for category, concept_pairs in concepts.items():
        category_dir = output_dir / category
        
        for positive_prompt, negative_prompt in concept_pairs:
            # Create concept set directory
            concept_dir = category_dir / f"{positive_prompt.replace(' ', '_')}_vs_{negative_prompt.replace(' ', '_')}"
            concept_dir.mkdir(exist_ok=True)
            
            # Get text embeddings for both prompts
            text_inputs = clip.tokenize([f"an image with {positive_prompt}", 
                                       f"an image with {negative_prompt}"]).to(device)
            with torch.no_grad():
                text_embeddings = model.encode_text(text_inputs)
                text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
            # Compute similarities
            similarities = (100.0 * image_embeddings @ text_embeddings.cpu().T)
            
            # Get positive concept images (high similarity with positive prompt)
            positive_scores = similarities[:, 0]
            positive_indices = torch.topk(positive_scores, k=min(top_k, len(image_paths))).indices
            
            # Get negative concept images (high similarity with negative prompt)
            negative_scores = similarities[:, 1]
            negative_indices = torch.topk(negative_scores, k=min(top_k, len(image_paths))).indices
            
            # Copy images to concept directories
            for prefix, indices in [("positive", positive_indices), ("negative", negative_indices)]:
                concept_subdir = concept_dir / prefix
                concept_subdir.mkdir(exist_ok=True)
                
                for rank, idx in enumerate(indices):
                    src_path = image_paths[idx]
                    dst_path = concept_subdir / f"{prefix}_{rank:03d}{Path(src_path).suffix}"
                    shutil.copy2(src_path, dst_path)
                    
                    # Save similarity score
                    with open(concept_subdir / "similarities.txt", "a") as f:
                        score = similarities[idx][0 if prefix == "positive" else 1].item()
                        f.write(f"{dst_path.name}\t{score:.4f}\n")
            
            print(f"Created concept set for {positive_prompt} vs {negative_prompt}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create concept sets using CLIP')
    parser.add_argument('--image_dir', type=str, default="sampled_images", help='Directory containing sampled images')
    parser.add_argument('--output_dir', type=str, default="data/CLIP_concept", help='Output directory for concept sets')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--top_k', type=int, default=50, help='Number of top images to select for each concept')
    
    args = parser.parse_args()
    
    create_concept_sets(args.image_dir, args.output_dir, args.batch_size, args.top_k)
