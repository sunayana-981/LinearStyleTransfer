import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torchvision.utils as vutils
from tqdm import tqdm
from libs.Matrix import MulLayer
from libs.models import encoder4, decoder4

class ConceptExtractor(nn.Module):
    def __init__(self, n_components=50):
        super().__init__()
        print("Loading DINO model from torch hub...")
        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
        self.n_components = n_components
        self.pca = None
        self.kmeans = None
        
        # Freeze DINO weights
        for param in self.dino.parameters():
            param.requires_grad = False
            
    def extract_features(self, x):
        """Extract DINO features with attention maps"""
        with torch.no_grad():
            attentions = self.dino.get_last_selfattention(x)
            features = self.dino(x)
        return features, attentions
    
    def fit_concept_space(self, dataset, n_clusters=10):
        """Fit PCA and KMeans to create concept space"""
        all_features = []
        
        # Extract features from dataset
        for img in dataset:
            features, _ = self.extract_features(img.unsqueeze(0))
            all_features.append(features.cpu().numpy())
            
        all_features = np.concatenate(all_features, axis=0)
        
        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        features_pca = self.pca.fit_transform(all_features)
        
        # Fit KMeans for concept clustering
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.kmeans.fit(features_pca)
        
        return features_pca, self.kmeans.labels_
    
    def extract_concepts(self, x, amplification_factor=2.0):
        """Extract and amplify concepts from input"""
        features, attentions = self.extract_features(x)
        
        # Project features to PCA space
        features_pca = self.pca.transform(features.cpu().numpy())
        
        # Get closest concept cluster
        cluster = self.kmeans.predict(features_pca)[0]
        
        # Get concept centroid
        concept_centroid = self.kmeans.cluster_centers_[cluster]
        
        # Amplify difference from centroid
        amplified_features = features_pca + (features_pca - concept_centroid) * amplification_factor
        
        # Project back to original space
        amplified_features = torch.tensor(
            self.pca.inverse_transform(amplified_features)
        ).float().cuda()
        
        return amplified_features, attentions, cluster

class ConceptStyleTransfer(nn.Module):
    def __init__(self, vgg_path, decoder_path, matrix_path, dino_path):
        super().__init__()
        # Load original style transfer models
        self.vgg = encoder4()
        self.decoder = decoder4()
        self.matrix = MulLayer('r41')
        
        self.vgg.load_state_dict(torch.load(vgg_path, weights_only=True))
        self.decoder.load_state_dict(torch.load(decoder_path, weights_only=True))
        self.matrix.load_state_dict(torch.load(matrix_path, weights_only=True))
        
        # Initialize concept extractor
        self.concept_extractor = ConceptExtractor(dino_path)
        
        self.eval()
        
    def forward(self, content, style, amplification_factor=2.0):
        """Perform concept-aware style transfer"""
        # Extract and amplify concepts
        content_concepts, content_attention, content_cluster = \
            self.concept_extractor.extract_concepts(content, amplification_factor)
        style_concepts, style_attention, style_cluster = \
            self.concept_extractor.extract_concepts(style, amplification_factor)
        
        # Get VGG features
        with torch.no_grad():
            cF = self.vgg(content)
            sF = self.vgg(style)
            
        # Modify features based on concept attention
        content_attention = content_attention.mean(1)  # Average attention heads
        style_attention = style_attention.mean(1)
        
        # Weight features by attention and concepts
        cF['r41'] = cF['r41'] * content_attention.unsqueeze(1)
        sF['r41'] = sF['r41'] * style_attention.unsqueeze(1)
        
        # Perform style transfer with concept-weighted features
        feature, _ = self.matrix(cF['r41'], sF['r41'])
        transfer = self.decoder(feature)
        
        return transfer.clamp(0, 1), content_cluster, style_cluster

def process_dataset_concepts(dataset_path, output_path, model, batch_size=4):
    """Process dataset to extract and visualize concepts"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    
    # Load and preprocess dataset
    dataset = []
    for img_path in sorted(os.listdir(dataset_path)):
        if img_path.endswith(('.jpg', '.png')):
            img = Image.open(os.path.join(dataset_path, img_path)).convert('RGB')
            img_tensor = transform(img).cuda()
            dataset.append(img_tensor)
    
    # Fit concept space
    features_pca, labels = model.concept_extractor.fit_concept_space(dataset)
    
    # Create visualization of concept clusters
    os.makedirs(output_path, exist_ok=True)
    
    # Process images in batches
    for i in range(0, len(dataset), batch_size):
        batch = torch.stack(dataset[i:i+batch_size])
        
        # Extract concepts and generate amplified versions
        for j, img in enumerate(batch):
            img_concepts, _, cluster = model.concept_extractor.extract_concepts(
                img.unsqueeze(0),
                amplification_factor=2.0
            )
            
            # Save original and amplified versions
            vutils.save_image(
                img,
                os.path.join(output_path, f'original_{i+j}.png'),
                normalize=True
            )
            vutils.save_image(
                img_concepts,
                os.path.join(output_path, f'amplified_{i+j}_cluster_{cluster}.png'),
                normalize=True
            )
    
    return features_pca, labels

def main():
    parser = argparse.ArgumentParser(description="Concept-Aware Neural Style Transfer with DINO Features")
    
    # Data paths
    parser.add_argument("--dataset_path", required=True,
                        help="Path to the dataset for concept extraction")
    parser.add_argument("--content_path", required=True,
                        help="Path to content images")
    parser.add_argument("--style_path", required=True,
                        help="Path to style images")
    parser.add_argument("--output_path", default="outputs",
                        help="Path to save outputs")
    
    # Model paths
    parser.add_argument("--vgg_path", default="models/vgg_r41.pth",
                        help="Path to pre-trained VGG model")
    parser.add_argument("--decoder_path", default="models/dec_r41.pth",
                        help="Path to pre-trained decoder model")
    parser.add_argument("--matrix_path", default="models/r41.pth",
                        help="Path to pre-trained matrix")
    # Removed DINO path argument as we're using pretrained model from torch hub
    
    # Concept extraction parameters
    parser.add_argument("--n_components", type=int, default=50,
                        help="Number of PCA components for concept space")
    parser.add_argument("--n_clusters", type=int, default=10,
                        help="Number of concept clusters")
    parser.add_argument("--amplification_factor", type=float, default=2.0,
                        help="Factor for concept amplification")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for processing")
    
    opt = parser.parse_args()
    
    # Initialize model
    try:
        model = ConceptStyleTransfer(
            vgg_path=opt.vgg_path,
            decoder_path=opt.decoder_path,
            matrix_path=opt.matrix_path
        ).cuda()
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure all required model files are present in the specified paths.")
        sys.exit(1)
    
    # Create output directories
    os.makedirs(opt.output_path, exist_ok=True)
    concept_path = os.path.join(opt.output_path, "concepts")
    transfer_path = os.path.join(opt.output_path, "transfers")
    os.makedirs(concept_path, exist_ok=True)
    os.makedirs(transfer_path, exist_ok=True)
    
    # Process dataset and extract concepts
    print("Extracting concepts from dataset...")
    features_pca, labels = process_dataset_concepts(
        opt.dataset_path,
        concept_path,
        model,
        opt.batch_size
    )
    
    # Save concept analysis results
    np.save(os.path.join(opt.output_path, "features_pca.npy"), features_pca)
    np.save(os.path.join(opt.output_path, "cluster_labels.npy"), labels)
    
    # Load and process content and style images
    print("Performing concept-aware style transfer...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    
    content_images = [f for f in os.listdir(opt.content_path) if f.endswith(('.jpg', '.png'))]
    style_images = [f for f in os.listdir(opt.style_path) if f.endswith(('.jpg', '.png'))]
    
    for content_name in tqdm(content_images, desc="Processing content images"):
        content_path = os.path.join(opt.content_path, content_name)
        content = transform(Image.open(content_path).convert('RGB')).unsqueeze(0).cuda()
        
        for style_name in tqdm(style_images, desc=f"Styles for {content_name}", leave=False):
            style_path = os.path.join(opt.style_path, style_name)
            style = transform(Image.open(style_path).convert('RGB')).unsqueeze(0).cuda()
            
            # Perform concept-aware style transfer
            with torch.no_grad():
                transfer, content_cluster, style_cluster = model(
                    content,
                    style,
                    opt.amplification_factor
                )
            
            # Save results
            output_name = f"{os.path.splitext(content_name)[0]}_x_{os.path.splitext(style_name)[0]}"
            output_name += f"_c{content_cluster}_s{style_cluster}.png"
            vutils.save_image(
                transfer,
                os.path.join(transfer_path, output_name),
                normalize=True
            )
    
    print("Processing complete! Results saved in:", opt.output_path)


if __name__ == "__main__":
    main()