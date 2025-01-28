import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict

class DINOConceptVectors:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load ViT model
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize hooks
        self.activation = {}
        self.hooks = []
        
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        
        # Add hooks to blocks
        for i, block in enumerate(self.model.blocks):
            self.hooks.append(block.register_forward_hook(get_activation(f'block_{i}')))
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # Define concept categories based on our analysis
        self.concept_categories = {
            'texture': [1, 2, 3],     # Early blocks for texture
            'composition': [8, 9, 10], # Later blocks for composition
            'brushwork': [4, 5, 6]    # Mid blocks for brushwork
        }

    def __del__(self):
        if hasattr(self, 'hooks'):
            for hook in self.hooks:
                hook.remove()

    @torch.no_grad()
    def extract_features(self, image_path):
        """Extract features for a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            img = self.transform(image).unsqueeze(0).to(self.device)
            
            # Forward pass
            _ = self.model(img)
            
            # Extract features for each concept category
            features = {}
            for category, blocks in self.concept_categories.items():
                category_features = []
                for block_idx in blocks:
                    block_output = self.activation[f'block_{block_idx}']
                    
                    if category == 'texture':
                        # Local pattern features
                        patch_features = block_output[:, 1:, :]  # Skip CLS token
                        mean_features = patch_features.mean(dim=1)
                        std_features = patch_features.std(dim=1)
                        features_combined = torch.cat([mean_features, std_features], dim=1)
                        category_features.append(features_combined)
                    
                    elif category == 'composition':
                        # Global composition features
                        cls_token = block_output[:, 0, :]
                        patch_tokens = block_output[:, 1:, :]
                        global_features = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
                        category_features.append(global_features)
                    
                    else:  # brushwork
                        # Mid-level features focusing on stroke patterns
                        patch_features = block_output[:, 1:, :]
                        spatial_features = patch_features.view(1, -1, 16, 16)  # Reshape to spatial grid
                        category_features.append(spatial_features.mean(dim=(2, 3)))
                
                features[category] = torch.cat(category_features, dim=1)
            
            return {k: v.cpu().numpy() for k, v in features.items()}
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def create_concept_dataset(self, positive_path, negative_path, category):
        """Create dataset for training CAVs"""
        positive_features = []
        negative_features = []
        
        # Process positive examples
        pos_path = Path(positive_path)
        for img_path in tqdm(list(pos_path.rglob('*.jpg'))[:500], desc="Processing positive examples"):
            features = self.extract_features(img_path)
            if features is not None:
                positive_features.append(features[category])
        
        # Process negative examples
        neg_path = Path(negative_path)
        for img_path in tqdm(list(neg_path.rglob('*.jpg'))[:500], desc="Processing negative examples"):
            features = self.extract_features(img_path)
            if features is not None:
                negative_features.append(features[category])
        
        if not positive_features or not negative_features:
            raise ValueError("No valid features extracted")
        
        # Create labels
        X = np.vstack(positive_features + negative_features)
        y = np.array([1] * len(positive_features) + [0] * len(negative_features))
        
        return X, y

    def train_cav(self, X, y, random_state=42):
        """Train a CAV using linear SVM"""
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=random_state
        )
        
        # Train linear SVM
        svm = LinearSVC(random_state=random_state)
        svm.fit(X_train, y_train)
        
        # Get accuracy
        train_acc = svm.score(X_train, y_train)
        test_acc = svm.score(X_test, y_test)
        
        print(f"Training accuracy: {train_acc:.3f}")
        print(f"Testing accuracy: {test_acc:.3f}")
        
        return svm, scaler

    def get_concept_sensitivity(self, image_path, cav_model, scaler, category):
        """Get concept sensitivity score for an image"""
        features = self.extract_features(image_path)
        if features is None:
            return None
            
        # Scale features
        features_scaled = scaler.transform(features[category].reshape(1, -1))
        
        # Get decision function value
        sensitivity = cav_model.decision_function(features_scaled)[0]
        
        return sensitivity

def create_style_cavs(base_path, style_concepts):
    """Create CAVs for different style concepts"""
    cav_creator = DINOConceptVectors()
    cavs = {}
    
    for concept, (positive_style, negative_style) in style_concepts.items():
        print(f"\nTraining CAV for concept: {concept}")
        for category in cav_creator.concept_categories:
            print(f"\nProcessing {category} features")
            
            # Create datasets
            positive_path = Path(base_path) / positive_style
            negative_path = Path(base_path) / negative_style
            
            try:
                X, y = cav_creator.create_concept_dataset(positive_path, negative_path, category)
                
                # Train CAV
                print(f"Training {concept} CAV for {category}")
                svm, scaler = cav_creator.train_cav(X, y)
                
                # Store CAV
                cavs[f"{concept}_{category}"] = {
                    'model': svm,
                    'scaler': scaler,
                    'category': category
                }
                
            except Exception as e:
                print(f"Error creating CAV for {concept} {category}: {str(e)}")
                continue
    
    return cavs, cav_creator

# Example usage
if __name__ == "__main__":
    # Define style concepts to create CAVs for
    style_concepts = {
        'impressionist_texture': ('Impressionism', 'Realism'),
        'abstract_composition': ('Abstract_Expressionism', 'Realism'),
        'modern_brushwork': ('Post_Impressionism', 'Baroque')
    }
    
    # Create CAVs
    base_path = "datasets/wikiArt/wikiart"
    cavs, cav_creator = create_style_cavs(base_path, style_concepts)
    
    # Test on a new image
    test_image = "path/to/test/image.jpg"
    for cav_name, cav_data in cavs.items():
        sensitivity = cav_creator.get_concept_sensitivity(
            test_image,
            cav_data['model'],
            cav_data['scaler'],
            cav_data['category']
        )
        print(f"{cav_name} sensitivity: {sensitivity:.3f}")