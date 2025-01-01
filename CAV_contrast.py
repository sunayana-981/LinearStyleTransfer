
import os
import torch
import argparse
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from torchvision import transforms
from PIL import Image
from sklearn.svm import LinearSVC
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from libs.Loader import Dataset
from libs.Matrix import MulLayer
from libs.utils import print_options
from libs.models import encoder3, encoder4, decoder3, decoder4
from tqdm import tqdm

class CAVVisualizer:
    def __init__(self, num_strengths):
        self.fig_size = (15, 8)
        self.num_strengths = num_strengths
    
    def create_comparison_plot(self, transfers, strengths, content_img, style_img, 
                             save_dir, content_name, style_num):
        fig = plt.figure(figsize=self.fig_size, constrained_layout=True)
        gs = GridSpec(2, self.num_strengths + 2, figure=fig)
        
        def tensor_to_image(tensor):
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            return tensor.cpu().permute(1, 2, 0).numpy()
        
        # Plot original images
        ax_content = fig.add_subplot(gs[0, 0])
        ax_content.imshow(tensor_to_image(content_img))
        ax_content.set_title('Content Image')
        ax_content.axis('off')
        
        ax_style = fig.add_subplot(gs[1, 0])
        ax_style.imshow(tensor_to_image(style_img))
        ax_style.set_title('Style Image')
        ax_style.axis('off')
        
        # Plot CAV variations
        for idx, (transfer, strength) in enumerate(zip(transfers, strengths)):
            ax = fig.add_subplot(gs[:, idx + 1])
            ax.imshow(tensor_to_image(transfer))
            ax.set_title(f'Contrast Strength: {strength:.1f}')
            ax.axis('off')
        
        # Add color bar
        norm = plt.Normalize(min(strengths), max(strengths))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=norm)
        cbar_ax = fig.add_subplot(gs[:, -1])
        plt.colorbar(sm, cax=cbar_ax, label='Contrast Strength')
        
        plot_filename = f'contrast_comparison_content{content_name}_style_{style_num:02d}.png'
        plot_path = os.path.join(save_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

class ContrastCAVController:
    def __init__(self, vgg, matrix, decoder, layer_name='r41', device='cuda'):
        self.vgg = vgg
        self.matrix = matrix
        self.decoder = decoder
        self.layer_name = layer_name
        self.device = device
        self.activations = []
        self._register_hooks()
    
    def _register_hooks(self):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.activations.append(output[0].detach())
            else:
                self.activations.append(output.detach())
        self.matrix.register_forward_hook(hook_fn)
    
    def get_style_features(self, image_path):
        self.activations = []
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor()
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.vgg(image_tensor)
            content_features = features[self.layer_name]
            style_features = features[self.layer_name]
            transformed_features, _ = self.matrix(content_features, style_features)
        
        if not self.activations:
            raise ValueError("No activations captured")
        return self.activations[-1]
    
    def collect_edges_images(self, style_dir):
        try:
            style_num = style_dir.split('style_')[-1]
            edges_base_dir = os.path.join('data', 'processed_styles', 'edges_styles')
            
            if not os.path.exists(edges_base_dir):
                raise ValueError(f"edges styles directory not found: {edges_base_dir}")
            
            # Get all edge intensity directories
            edge_dirs = []
            for item in os.listdir(edges_base_dir):
                if item.startswith('edges_') and os.path.isdir(os.path.join(edges_base_dir, item)):
                    try:
                        intensity = float(item.split('_')[1])
                        edge_dirs.append((intensity, item))
                    except ValueError:
                        continue
            
            if not edge_dirs:
                raise ValueError("No edges directories found")
            
            # Sort by intensity
            edge_dirs.sort(key=lambda x: x[0])
            
            # Collect image paths
            image_paths = []
            for _, dir_name in edge_dirs:
                full_dir = os.path.join(edges_base_dir, dir_name)
                image_name = f"{style_num}_edges_{dir_name.split('_')[1]}.png"
                image_path = os.path.join(full_dir, image_name)
                
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    print(f"Found edge image: {image_path}")  # Debug print
            
            if not image_paths:
                raise ValueError("No valid edges images found")
            
            print(f"Total edge images found: {len(image_paths)}")  # Debug print
            return str(image_paths[0]), [str(p) for p in image_paths[1:]]
            
        except Exception as e:
            print(f"Error in collect_edges_images: {str(e)}")
            raise
    
    def learn_contrast_cav(self, style_dir):
        try:
            # Get image paths
            original_path, contrast_paths = self.collect_contrast_images(style_dir)
            
            with tqdm(total=3, desc="Learning CAV", leave=False) as pbar:
                # Get features for original image
                original_features = self.get_style_features(original_path)
                original_flat = original_features.view(original_features.size(0), -1).cpu().numpy()
                pbar.update(1)
                
                # Get features for contrasted images
                contrast_features = []
                contrast_paths = [contrast_paths[0], contrast_paths[-1]]
                for path in contrast_paths:
                    features = self.get_style_features(path)
                    contrast_features.append(features.view(features.size(0), -1).cpu().numpy())
                pbar.update(1)
                
                # Prepare and train SVM
                contrast_flat = np.concatenate(contrast_features, axis=0)
                features = np.concatenate([original_flat, contrast_flat])
                labels = np.concatenate([
                    np.zeros(len(original_flat)),
                    np.ones(len(contrast_flat))
                ])
                
                svm = LinearSVC(
                    C=0.01,
                    max_iter=100,
                    tol=1e-2,
                    dual=False,
                    random_state=42
                )
                svm.fit(features, labels)
                pbar.update(1)
                
                cav = torch.tensor(svm.coef_[0]).reshape(original_features.shape[1:]).to(self.device)
                return cav
                
        except Exception as e:
            print(f"Error in CAV learning: {str(e)}")
            raise
    
    def apply_cav(self, content_image, style_image, cav, strength=1.0):
        with torch.no_grad():
            content_image = content_image.float()
            style_image = style_image.float()
            
            content_features = self.vgg(content_image)[self.layer_name]
            style_features = self.vgg(style_image)[self.layer_name]
            
            transformed_features, matrix = self.matrix(content_features, style_features)
            transformed_features = transformed_features.float()
            
            cav = cav.float()
            cav_norm = torch.norm(cav)
            feature_norm = torch.norm(transformed_features)
            
            scale_factor = feature_norm / cav_norm
            adaptive_strength = strength * scale_factor * 0.1
            
            modified_features = transformed_features + (adaptive_strength * cav.unsqueeze(0)).float()
            result = self.decoder(modified_features)
            
            return result.clamp(0, 1), matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contentPath", default="data/content/",
                      help='path to content images')
    parser.add_argument("--stylePath", default="data/style/",
                      help='path to style images')
    parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                      help='pre-trained encoder path')
    parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                      help='pre-trained decoder path')
    parser.add_argument("--matrixPath", default='models/r41.pth',
                      help='pre-trained matrix path')
    parser.add_argument("--outf", default="Artistic/contrast/",
                      help='path to output images')
    parser.add_argument("--layer", default="r41",
                      help='which features to transfer')
    parser.add_argument("--cav_strengths", type=float, nargs='+',
                      default=[-3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0],
                      help='strengths at which to apply CAV')
    
    opt = parser.parse_args()
    opt.cuda = torch.cuda.is_available()
    print_options(opt)
    
    # Create output directory
    comparison_dir = os.path.join(opt.outf, 'comparisons')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Initialize models
    print("Initializing models...")
    vgg = encoder4()
    dec = decoder4()
    matrix = MulLayer(opt.layer)
    
    # Load weights
    vgg.load_state_dict(torch.load(opt.vgg_dir, weights_only=True))
    dec.load_state_dict(torch.load(opt.decoder_dir, weights_only=True))
    matrix.load_state_dict(torch.load(opt.matrixPath, weights_only=True))
    
    if opt.cuda:
        vgg.cuda()
        dec.cuda()
        matrix.cuda()
    
    # Initialize controllers
    cav_controller = ContrastCAVController(vgg, matrix, dec, opt.layer)
    visualizer = CAVVisualizer(len(opt.cav_strengths))
    
    # Load datasets
    print("Loading datasets...")
    content_dataset = Dataset(opt.contentPath, 256, 256)
    style_dataset = Dataset(opt.stylePath, 256, 256)
    
    content_loader = torch.utils.data.DataLoader(content_dataset, batch_size=1, shuffle=False)
    style_loader = torch.utils.data.DataLoader(style_dataset, batch_size=1, shuffle=False)
    
    print("Starting processing...")
    # Process each pair with progress bars
    for content_idx, (content, contentName) in enumerate(tqdm(content_loader, desc="Processing content images")):
        contentV = content.cuda() if opt.cuda else content
        
        for style_idx, (style, styleName) in enumerate(tqdm(style_loader, desc=f"Processing styles for content {content_idx+1}", leave=False)):
            styleV = style.cuda() if opt.cuda else style
            style_dir = f'style_{style_idx+1:02d}'
            
            try:
                # Learn CAV
                contrast_cav = cav_controller.learn_contrast_cav(style_dir)
                
                # Generate variations with progress bar
                transfers = []
                for strength in tqdm(opt.cav_strengths, desc="Generating CAV variations", leave=False):
                    transfer, _ = cav_controller.apply_cav(contentV, styleV, contrast_cav, strength)
                    transfers.append(transfer)
                
                # Create comparison plot
                visualizer.create_comparison_plot(
                    transfers,
                    opt.cav_strengths,
                    contentV,
                    styleV,
                    comparison_dir,
                    content_idx + 1,
                    style_idx + 1
                )
                
            except Exception as e:
                print(f"Error processing {style_dir}: {str(e)}")
                continue
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
