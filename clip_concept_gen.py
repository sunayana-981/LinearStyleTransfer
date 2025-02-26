import torch
import clip
from PIL import Image
import os
from pathlib import Path
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import logging
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
import json

class CLIPConceptClassifier:
    def __init__(self, device='cuda'):
        self.device = device
        # Load CLIP model
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # Define concept categories and their descriptors
        self.concept_categories = {
            "lighting_condition": [
                "bright daylight scene",
                "dark night scene",
                "indoor artificial lighting",
                "golden hour sunset lighting",
                "soft diffused lighting"
            ],
            "weather_condition": [
                "clear sunny weather",
                "cloudy overcast scene",
                "rainy weather scene",
                "foggy misty atmosphere",
                "snowy winter scene"
            ],
            "scene_type": [
                "urban city scene",
                "natural landscape",
                "indoor room scene",
                "beach coastal scene",
                "forest woodland scene"
            ],
            "time_of_day": [
                "morning scene",
                "midday scene",
                "sunset evening scene",
                "night scene",
                "dawn early morning"
            ],
            "season": [
                "summer scene",
                "autumn fall scene",
                "winter scene",
                "spring scene"
            ]
        }
        
        # Pre-compute text embeddings
        self.text_embeddings = {}
        for category, descriptors in self.concept_categories.items():
            self.text_embeddings[category] = self._compute_text_embeddings(descriptors)
    
    def _compute_text_embeddings(self, texts):
        """Compute CLIP embeddings for text descriptions"""
        text_tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def _compute_image_embedding(self, image_path):
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
    
    def classify_image(self, image_path):
        """Classify image into different concept categories"""
        image_embedding = self._compute_image_embedding(image_path)
        if image_embedding is None:
            return None
        
        classifications = {}
        
        # Compare with each category's text embeddings
        for category, text_emb in self.text_embeddings.items():
            similarity = (100.0 * image_embedding @ text_emb.T).softmax(dim=-1)
            probs, indices = similarity[0].sort(descending=True)
            
            # Get top matches
            top_matches = []
            for prob, idx in zip(probs, indices):
                concept = self.concept_categories[category][idx]
                top_matches.append({
                    "concept": concept,
                    "confidence": prob.item()
                })
            
            classifications[category] = top_matches
        
        return classifications
    
    def batch_classify_directory(self, directory_path):
        """Classify all images in a directory"""
        image_paths = list(Path(directory_path).glob('*.jpg')) + \
                     list(Path(directory_path).glob('*.png'))
        
        results = {}
        for img_path in tqdm(image_paths, desc="Classifying images"):
            classifications = self.classify_image(img_path)
            if classifications:
                results[str(img_path)] = classifications
        
        return results
    
    def analyze_concept_distribution(self, classification_results):
        """Analyze distribution of concepts across images"""
        distribution = {category: {} for category in self.concept_categories.keys()}
        
        for img_path, classifications in classification_results.items():
            for category, matches in classifications.items():
                # Count top concept for each category
                top_concept = matches[0]["concept"]
                if top_concept not in distribution[category]:
                    distribution[category][top_concept] = 0
                distribution[category][top_concept] += 1
        
        return distribution
    
    def group_similar_images(self, directory_path, n_clusters=5):
        """Group images based on their CLIP embeddings"""
        # Collect image embeddings
        image_embeddings = []
        image_paths = []
        
        for img_path in tqdm(Path(directory_path).glob('*.*'), desc="Computing embeddings"):
            if img_path.suffix.lower() in ['.jpg', '.png']:
                embedding = self._compute_image_embedding(img_path)
                if embedding is not None:
                    image_embeddings.append(embedding.cpu().numpy().flatten())
                    image_paths.append(str(img_path))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(np.array(image_embeddings))
        
        # Group images by cluster
        grouped_images = {i: [] for i in range(n_clusters)}
        for path, cluster in zip(image_paths, clusters):
            grouped_images[cluster].append(path)
        
        return grouped_images

def process_coco_dataset():
    """Process COCO dataset and categorize images"""
    # Initialize classifier
    classifier = CLIPConceptClassifier()
    
    # Process images
    results = classifier.batch_classify_directory('datasets/ade20k/images/training/')
    
    # Analyze concept distribution
    distribution = classifier.analyze_concept_distribution(results)
    
    # Group similar images
    grouped_images = classifier.group_similar_images('datasets/ade20k/images/training/')
    
    # Save results
    output = {
        'classifications': results,
        'distribution': distribution,
        'groups': grouped_images
    }
    
    # Save to JSON
    with open('concept_analysis_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    return output

def visualize_results(results):
    """Create visualizations of the concept analysis"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Plot concept distribution
    plt.figure(figsize=(15, 10))
    for i, (category, concepts) in enumerate(results['distribution'].items()):
        plt.subplot(3, 2, i+1)
        concepts_sorted = sorted(concepts.items(), key=lambda x: x[1], reverse=True)
        labels, values = zip(*concepts_sorted)
        plt.bar(range(len(values)), values)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.title(f'{category} Distribution')
    plt.tight_layout()
    plt.savefig('concept_distribution.png')
    plt.close()
    
    return "Results visualized and saved to concept_distribution.png"


if __name__ == "__main__":
    # Process COCO dataset and save results
    results = process_coco_dataset()
    print(visualize_results(results))

