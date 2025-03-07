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
    def __init__(self, grid_size):
        self.fig_size = (15, 12)
        self.grid_size = grid_size  # (rows, cols)
    
    def create_grid_plot(self, images, contrast_strengths, texture_strengths, 
                         content_img, style_img, save_dir, content_name, style_num):
        fig = plt.figure(figsize=self.fig_size, constrained_layout=True)
        rows, cols = self.grid_size
        
        # Create grid with space for content and style images and colorbars
        gs = GridSpec(rows + 1, cols + 2, figure=fig)
        
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
        
        # Plot CAV variations in grid
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < len(images):
                    ax = fig.add_subplot(gs[i, j+1])
                    ax.imshow(tensor_to_image(images[idx]))
                    ax.set_title(f'C:{contrast_strengths[i]:.1f}, T:{texture_strengths[j]:.1f}')
                    ax.axis('off')
        
        # Add Contrast colorbar (vertical)
        norm_contrast = plt.Normalize(min(contrast_strengths), max(contrast_strengths))
        sm_contrast = plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=norm_contrast)
        cbar_ax_contrast = fig.add_subplot(gs[:, -2])
        plt.colorbar(sm_contrast, cax=cbar_ax_contrast, label='Contrast Strength')
        
        # Add Texture colorbar (horizontal)
        norm_texture = plt.Normalize(min(texture_strengths), max(texture_strengths))
        sm_texture = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=norm_texture)
        cbar_ax_texture = fig.add_subplot(gs[-1, 1:-2])
        plt.colorbar(sm_texture, cax=cbar_ax_texture, orientation='horizontal', 
                     label='Texture Strength')
        
        plot_filename = f'combined_cav_content{content_name}_style_{style_num:02d}.png'
        plot_path = os.path.join(save_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

    def create_interpolation_plot(self, images, alphas, content_img1, style_img1, 
                                 content_img2, style_img2, save_dir, content_name1, 
                                 style_num1, content_name2, style_num2):
        fig = plt.figure(figsize=(15, 8), constrained_layout=True)
        num_images = len(images)
        gs = GridSpec(2, num_images + 2, figure=fig)
        
        def tensor_to_image(tensor):
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            return tensor.cpu().permute(1, 2, 0).numpy()
        
        # Plot source and target
        ax_source_content = fig.add_subplot(gs[0, 0])
        ax_source_content.imshow(tensor_to_image(content_img1))
        ax_source_content.set_title('Source Content')
        ax_source_content.axis('off')
        
        ax_source_style = fig.add_subplot(gs[1, 0])
        ax_source_style.imshow(tensor_to_image(style_img1))
        ax_source_style.set_title('Source Style')
        ax_source_style.axis('off')
        
        ax_target_content = fig.add_subplot(gs[0, -1])
        ax_target_content.imshow(tensor_to_image(content_img2))
        ax_target_content.set_title('Target Content')
        ax_target_content.axis('off')
        
        ax_target_style = fig.add_subplot(gs[1, -1])
        ax_target_style.imshow(tensor_to_image(style_img2))
        ax_target_style.set_title('Target Style')
        ax_target_style.axis('off')
        
        # Plot interpolations
        for idx, (image, alpha) in enumerate(zip(images, alphas)):
            ax = fig.add_subplot(gs[:, idx + 1])
            ax.imshow(tensor_to_image(image))
            ax.set_title(f'Alpha: {alpha:.2f}')
            ax.axis('off')
        
        plot_filename = f'interpolation_c{content_name1}_s{style_num1}_to_c{content_name2}_s{style_num2}.png'
        plot_path = os.path.join(save_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

class CAVController:
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
    
    def collect_contrast_images(self, style_dir):
        try:
            style_num = style_dir.split('style_')[-1]
            contrast_base_dir = os.path.join('data', 'processed_styles', 'contrast_styles')
            
            if not os.path.exists(contrast_base_dir):
                raise ValueError(f"Contrast styles directory not found: {contrast_base_dir}")
            
            # Get all contrast intensity directories
            contrast_dirs = []
            for item in os.listdir(contrast_base_dir):
                if item.startswith('contrast_') and os.path.isdir(os.path.join(contrast_base_dir, item)):
                    try:
                        intensity = float(item.split('_')[1])
                        contrast_dirs.append((intensity, item))
                    except ValueError:
                        continue
            
            if not contrast_dirs:
                raise ValueError("No contrast directories found")
            
            # Sort by intensity
            contrast_dirs.sort(key=lambda x: x[0])
            
            # Collect image paths
            image_paths = []
            for _, dir_name in contrast_dirs:
                full_dir = os.path.join(contrast_base_dir, dir_name)
                image_name = f"{style_num}_contrast_{dir_name.split('_')[1]}.png"
                image_path = os.path.join(full_dir, image_name)
                
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    # print(f"Found contrast image: {image_path}")  # Debug print
            
            if not image_paths:
                raise ValueError("No valid contrast images found")
            
            # print(f"Total contrast images found: {len(image_paths)}")  # Debug print
            return str(image_paths[0]), [str(p) for p in image_paths[1:]]
            
        except Exception as e:
            print(f"Error in collect_contrast_images: {str(e)}")
            raise
    
    def collect_texture_images(self, style_dir):
        style_num = style_dir.split('style_')[-1]
        texture_dir = os.path.join('data', 'effect_styled_outputs', 'texture')
        
        if not os.path.exists(texture_dir):
            raise ValueError(f"Texture styles directory not found: {texture_dir}")
        
        texture_list = []
        for item in os.listdir(texture_dir):
            if item.startswith('texture_') and os.path.isdir(os.path.join(texture_dir, item)):
                try:
                    intensity = float(item.split('_')[1])
                    texture_list.append((intensity, item))
                except ValueError:
                    print(f"Skipping invalid directory: {item}")
        
        if not texture_list:
            raise ValueError("No texture directories found")
        
        texture_list.sort(key=lambda x: x[0])
        
        image_paths = []
        for intensity, texture_dir_name in texture_list:
            full_dir = os.path.join(texture_dir, texture_dir_name, style_dir)
            formatted_intensity = f"{intensity:.1f}"
            image_name = f"02_style_{style_num}.png"
            image_path = os.path.join(full_dir, image_name)
            
            if os.path.exists(image_path):
                image_paths.append(image_path)
            else:
                print(f"File not found: {image_path}")
        
        if not image_paths:
            raise ValueError("No valid texture images found")
        
        return str(image_paths[0]), [str(p) for p in image_paths[1:]]
    
    def learn_cavs(self, style_dir):
        contrast_cav = None
        texture_cav = None
        
        try:
            # Learn contrast CAV
            with tqdm(total=2, desc="Learning CAVs", leave=False) as pbar:
                try:
                    original_path, contrast_paths = self.collect_contrast_images(style_dir)
                    
                    original_features = self.get_style_features(original_path)
                    original_flat = original_features.view(original_features.size(0), -1).cpu().numpy()
                    
                    contrast_features = []
                    contrast_paths = [contrast_paths[0], contrast_paths[-1]]
                    for path in contrast_paths:
                        features = self.get_style_features(path)
                        contrast_features.append(features.view(features.size(0), -1).cpu().numpy())
                    
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
                    
                    contrast_cav = torch.tensor(svm.coef_[0]).reshape(original_features.shape[1:]).to(self.device)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error learning contrast CAV: {str(e)}")
                    contrast_cav = None
                
                # Learn texture CAV
                try:
                    original_path, texture_paths = self.collect_texture_images(style_dir)
                    
                    original_features = self.get_style_features(original_path)
                    original_flat = original_features.view(original_features.size(0), -1).cpu().numpy()
                    
                    texture_features = []
                    texture_paths = [texture_paths[0], texture_paths[-1]]
                    for path in texture_paths:
                        features = self.get_style_features(path)
                        texture_features.append(features.view(features.size(0), -1).cpu().numpy())
                    
                    texture_flat = np.concatenate(texture_features, axis=0)
                    features = np.concatenate([original_flat, texture_flat])
                    labels = np.concatenate([
                        np.zeros(len(original_flat)),
                        np.ones(len(texture_flat))
                    ])
                    
                    svm = LinearSVC(
                        C=0.01,
                        max_iter=100,
                        tol=1e-2,
                        dual=False,
                        random_state=42
                    )
                    svm.fit(features, labels)
                    
                    texture_cav = torch.tensor(svm.coef_[0]).reshape(original_features.shape[1:]).to(self.device)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error learning texture CAV: {str(e)}")
                    texture_cav = None
        
        except Exception as e:
            print(f"Error in CAV learning: {str(e)}")
        
        return contrast_cav, texture_cav
    
    def apply_cavs(self, content_image, style_image, contrast_cav=None, texture_cav=None, 
                 contrast_strength=0.0, texture_strength=0.0, module='both'):
        with torch.no_grad():
            content_image = content_image.float()
            style_image = style_image.float()
            
            content_features = self.vgg(content_image)[self.layer_name]
            style_features = self.vgg(style_image)[self.layer_name]
            
            transformed_features, matrix = self.matrix(content_features, style_features)
            transformed_features = transformed_features.float()
            
            # Apply CAVs based on module selection
            if module in ['content', 'both'] and contrast_cav is not None:
                contrast_cav = contrast_cav.float()
                cav_norm = torch.norm(contrast_cav)
                feature_norm = torch.norm(content_features)
                
                scale_factor = feature_norm / cav_norm
                adaptive_strength = contrast_strength * scale_factor * 0.1
                
                # Apply to content
                content_features = content_features + (adaptive_strength * contrast_cav.unsqueeze(0)).float()
                # Recompute transformation
                transformed_features, matrix = self.matrix(content_features, style_features)
            
            if module in ['style', 'both'] and texture_cav is not None:
                texture_cav = texture_cav.float()
                cav_norm = torch.norm(texture_cav)
                feature_norm = torch.norm(style_features)
                
                scale_factor = feature_norm / cav_norm
                adaptive_strength = texture_strength * scale_factor * 0.1
                
                # Apply to style
                style_features = style_features + (adaptive_strength * texture_cav.unsqueeze(0)).float()
                # Recompute transformation
                transformed_features, matrix = self.matrix(content_features, style_features)
            
            # Apply directly to combined features for additional control
            if module == 'combined':
                if contrast_cav is not None:
                    contrast_cav = contrast_cav.float()
                    cav_norm = torch.norm(contrast_cav)
                    feature_norm = torch.norm(transformed_features)
                    
                    scale_factor = feature_norm / cav_norm
                    adaptive_strength = contrast_strength * scale_factor * 0.1
                    
                    transformed_features = transformed_features + (adaptive_strength * contrast_cav.unsqueeze(0)).float()
                
                if texture_cav is not None:
                    texture_cav = texture_cav.float()
                    cav_norm = torch.norm(texture_cav)
                    feature_norm = torch.norm(transformed_features)
                    
                    scale_factor = feature_norm / cav_norm
                    adaptive_strength = texture_strength * scale_factor * 0.1
                    
                    transformed_features = transformed_features + (adaptive_strength * texture_cav.unsqueeze(0)).float()
            
            result = self.decoder(transformed_features)
            
            return result.clamp(0, 1), matrix
    
    def interpolate_styles(self, content_image1, style_image1, content_image2, style_image2, 
                         contrast_cav1, texture_cav1, contrast_cav2, texture_cav2, 
                         alphas, contrast_strength=1.0, texture_strength=1.0):
        results = []
        
        for alpha in alphas:
            # Interpolate content and style images
            content_interp = content_image1 * (1 - alpha) + content_image2 * alpha
            style_interp = style_image1 * (1 - alpha) + style_image2 * alpha
            
            # Interpolate CAVs
            contrast_cav = None
            if contrast_cav1 is not None and contrast_cav2 is not None:
                contrast_cav = contrast_cav1 * (1 - alpha) + contrast_cav2 * alpha
            
            texture_cav = None
            if texture_cav1 is not None and texture_cav2 is not None:
                texture_cav = texture_cav1 * (1 - alpha) + texture_cav2 * alpha
            
            # Apply interpolated CAVs
            result, _ = self.apply_cavs(
                content_interp, style_interp,
                contrast_cav, texture_cav,
                contrast_strength, texture_strength,
                'both'
            )
            
            results.append(result)
        
        return results

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
    parser.add_argument("--outf", default="Artistic/combined/",
                      help='path to output images')
    parser.add_argument("--layer", default="r41",
                      help='which features to transfer')
    parser.add_argument("--contrast_strengths", type=float, nargs='+',
                      default=[-3.0,-2.0, -1.0, 0.0, 1.0, 2.0,3.0],
                      help='strengths at which to apply contrast CAV')
    parser.add_argument("--texture_strengths", type=float, nargs='+',
                      default=[-3.0,-2.0, -1.0, 0.0, 1.0, 2.0,3.0],
                      help='strengths at which to apply texture CAV')
    parser.add_argument("--mode", default="grid",
                      choices=["grid", "interpolate", "module_compare"],
                      help='visualization mode')
    parser.add_argument("--alphas", type=float, nargs='+',
                      default=[0.0, 0.25, 0.5, 0.75, 1.0],
                      help='alphas for interpolation mode')
    parser.add_argument("--modules", default="both",
                      choices=["content", "style", "both", "combined"],
                      help='which modules to apply CAVs to')
    
    opt = parser.parse_args()
    opt.cuda = torch.cuda.is_available()
    print_options(opt)
    
    # Create output directories
    grid_dir = os.path.join(opt.outf, 'grid')
    interp_dir = os.path.join(opt.outf, 'interpolation')
    module_dir = os.path.join(opt.outf, 'module_compare')
    
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(interp_dir, exist_ok=True)
    os.makedirs(module_dir, exist_ok=True)
    
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
    cav_controller = CAVController(vgg, matrix, dec, opt.layer)
    grid_size = (len(opt.contrast_strengths), len(opt.texture_strengths))
    visualizer = CAVVisualizer(grid_size)
    
    # Load datasets
    print("Loading datasets...")
    content_dataset = Dataset(opt.contentPath, 256, 256)
    style_dataset = Dataset(opt.stylePath, 256, 256)
    
    content_loader = torch.utils.data.DataLoader(content_dataset, batch_size=1, shuffle=False)
    style_loader = torch.utils.data.DataLoader(style_dataset, batch_size=1, shuffle=False)
    
    # Convert to list for multiple access
    content_list = [(content, name) for content, name in content_loader]
    style_list = [(style, name) for style, name in style_loader]
    
    # Process based on mode
    if opt.mode == "grid":
        print("Starting grid processing...")
        for content_idx, (content, contentName) in enumerate(tqdm(content_list, desc="Processing content images")):
            contentV = content.cuda() if opt.cuda else content
            
            for style_idx, (style, styleName) in enumerate(tqdm(style_list, desc=f"Processing styles for content {content_idx+1}", leave=False)):
                styleV = style.cuda() if opt.cuda else style
                style_dir = f'style_{style_idx+1:02d}'
                
                try:
                    # Learn CAVs
                    contrast_cav, texture_cav = cav_controller.learn_cavs(style_dir)
                    
                    if contrast_cav is None and texture_cav is None:
                        print(f"Skipping {style_dir} - no CAVs learned")
                        continue
                    
                    # Generate grid variations
                    grid_images = []
                    for c_strength in tqdm(opt.contrast_strengths, desc="Generating grid", leave=False):
                        for t_strength in opt.texture_strengths:
                            transfer, _ = cav_controller.apply_cavs(
                                contentV, styleV, 
                                contrast_cav, texture_cav,
                                c_strength, t_strength,
                                opt.modules
                            )
                            grid_images.append(transfer)
                    
                    # Create comparison plot
                    visualizer.create_grid_plot(
                        grid_images,
                        opt.contrast_strengths,
                        opt.texture_strengths,
                        contentV,
                        styleV,
                        grid_dir,
                        content_idx + 1,
                        style_idx + 1
                    )
                    
                except Exception as e:
                    print(f"Error processing {style_dir}: {str(e)}")
                    continue
    
    elif opt.mode == "interpolate":
        print("Starting interpolation processing...")
        # Use different pairs for interpolation
        if len(content_list) >= 2 and len(style_list) >= 2:
            # For simplicity, we'll use the first two content and style images
            content1, contentName1 = content_list[0]
            content2, contentName2 = content_list[1]
            style1, styleName1 = style_list[0]
            style2, styleName2 = style_list[1]
            
            contentV1 = content1.cuda() if opt.cuda else content1
            contentV2 = content2.cuda() if opt.cuda else content2
            styleV1 = style1.cuda() if opt.cuda else style1
            styleV2 = style2.cuda() if opt.cuda else style2
            
            style_dir1 = 'style_01'
            style_dir2 = 'style_02'
            
            try:
                # Learn CAVs for both styles
                contrast_cav1, texture_cav1 = cav_controller.learn_cavs(style_dir1)
                contrast_cav2, texture_cav2 = cav_controller.learn_cavs(style_dir2)
                
                # Generate interpolations
                interp_results = cav_controller.interpolate_styles(
                    contentV1, styleV1, contentV2, styleV2,
                    contrast_cav1, texture_cav1, contrast_cav2, texture_cav2,
                    opt.alphas, 1.0, 1.0
                )
                
                # Create interpolation plot
                visualizer.create_interpolation_plot(
                    interp_results,
                    opt.alphas,
                    contentV1,
                    styleV1,
                    contentV2,
                    styleV2,
                    interp_dir,
                    1, 1, 2, 2
                )
                
            except Exception as e:
                print(f"Error processing interpolation: {str(e)}")
        else:
            print("Not enough content or style images for interpolation")
    
    elif opt.mode == "module_compare":
        print("Starting module comparison processing...")
        for content_idx, (content, contentName) in enumerate(tqdm(content_list[:1], desc="Processing content image")):
            contentV = content.cuda() if opt.cuda else content
            
            for style_idx, (style, styleName) in enumerate(tqdm(style_list[:1], desc="Processing style", leave=False)):
                styleV = style.cuda() if opt.cuda else style
                style_dir = f'style_{style_idx+1:02d}'
                
                try:
                    # Learn CAVs
                    contrast_cav, texture_cav = cav_controller.learn_cavs(style_dir)
                    
                    if contrast_cav is None and texture_cav is None:
                        print(f"Skipping {style_dir} - no CAVs learned")
                        continue
                    
                    # Apply to different modules
                    modules = ['content', 'style', 'both', 'combined']
                    module_strengths = [0.0, 1.0, 2.0]  # Using fewer strengths for clarity
                    
                    for module in tqdm(modules, desc="Processing modules", leave=False):
                        module_images = []
                        
                        for strength in module_strengths:
                            # Apply same strength to both CAVs for this comparison
                            transfer, _ = cav_controller.apply_cavs(
                                contentV, styleV, 
                                contrast_cav, texture_cav,
                                strength, strength,
                                module
                            )
                            module_images.append(transfer)
                        
                        # Save individual images for module comparison
                        for s_idx, (img, strength) in enumerate(zip(module_images, module_strengths)):
                            img_np = img.squeeze(0).cpu().permute(1, 2, 0).numpy()
                            plt.figure(figsize=(5, 5))
                            plt.imshow(img_np)
                            plt.title(f'Module: {module}, Strength: {strength}')
                            plt.axis('off')
                            plt.savefig(
                                os.path.join(module_dir, f'module_{module}_strength_{strength}_c{content_idx+1}_s{style_idx+1}.png'),
                                bbox_inches='tight',
                                dpi=300
                            )
                            plt.close()
                    
                except Exception as e:
                    print(f"Error processing module comparison: {str(e)}")
                    continue
    
    print("Processing complete!")

if __name__ == "__main__":
    main()