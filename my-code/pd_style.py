import os
import torch
import argparse
import logging
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from libs.models import encoder4, decoder4

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('style_transfer_debug.log')
    ]
)

def load_image(image_path, size=256):
    """Load and preprocess image."""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    logging.info(f"Loaded image {image_path}, tensor shape: {tensor.shape}")
    return tensor

def save_image(tensor, path):
    """Save tensor as image."""
    image = tensor.squeeze(0).cpu().clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)
    logging.info(f"Saved image to {path}")

class FeatureSpacePDController:
    def __init__(self, vgg_encoder, decoder, kp=0.5, kd=0.1, target_threshold=0.01):
        self.vgg = vgg_encoder
        self.decoder = decoder
        self.kp = kp
        self.kd = kd
        self.target_threshold = target_threshold
        self.previous_error = None
        self.history = []
        logging.info(f"Initialized PD Controller with kp={kp}, kd={kd}")

    def get_features(self, image):
        """Extract features from image."""
        try:
            with torch.no_grad():
                # VGG encoder returns a dict with layer names as keys
                features_dict = self.vgg(image)
                # Extract r41 features (assuming this is the layer we want)
                features = features_dict['r41']
                logging.info(f"Extracted features shape: {features.shape}")
                return features
        except Exception as e:
            logging.error(f"Error extracting features: {str(e)}")
            raise

    def optimize_step(self, current_features, target_features):
        """Perform optimization step."""
        try:
            # Ensure tensors are on the same device
            current_features = current_features.float()
            target_features = target_features.float()
            
            # Normalize features
            current_norm = torch.norm(current_features.view(current_features.size(0), -1), dim=1, keepdim=True)
            target_norm = torch.norm(target_features.view(target_features.size(0), -1), dim=1, keepdim=True)
            
            current_normalized = current_features / (current_norm.view(-1, 1, 1, 1) + 1e-7)
            target_normalized = target_features / (target_norm.view(-1, 1, 1, 1) + 1e-7)
            
            # Compute error
            error = F.mse_loss(current_normalized, target_normalized)
            
            # Compute control signal
            p_term = self.kp * (target_features - current_features)
            
            if self.previous_error is not None:
                d_term = self.kd * (error - self.previous_error)
                control = p_term - d_term.view(-1, 1, 1, 1) * torch.ones_like(current_features)
            else:
                control = p_term
                
            updated_features = current_features + control
            
            # Store history
            self.history.append({
                'error': error.item(),
                'p_term_norm': torch.norm(p_term).item(),
                'control_norm': torch.norm(control).item()
            })
            
            self.previous_error = error
            
            logging.info(f"Step error: {error.item():.6f}")
            return updated_features, control, error
            
        except Exception as e:
            logging.error(f"Error in optimization step: {str(e)}")
            raise

    def is_converged(self):
        """Check if optimization has converged."""
        if len(self.history) < 2:
            return False
        return self.history[-1]['error'] < self.target_threshold

class StyleTransferOptimizer:
    def __init__(self, pd_controller, max_iterations=50):
        self.pd_controller = pd_controller
        self.max_iterations = max_iterations
        logging.info(f"Initialized Optimizer with max_iterations={max_iterations}")

    def optimize(self, content_image, style_image):
        """Run optimization process."""
        try:
            # Get target features
            logging.info("Extracting target features...")
            target_features = self.pd_controller.get_features(style_image)
            
            # Initialize with content features
            current_features = self.pd_controller.get_features(content_image)
            
            results = []
            for i in range(self.max_iterations):
                logging.info(f"Iteration {i+1}/{self.max_iterations}")
                
                # Optimization step
                updated_features, control, error = self.pd_controller.optimize_step(
                    current_features, target_features
                )
                
                # Decode to image
                with torch.no_grad():
                    current_image = self.pd_controller.decoder(updated_features)
                    current_image = current_image.clamp(0, 1)
                
                results.append({
                    'image': current_image.detach(),
                    'error': error.item()
                })
                
                current_features = updated_features
                
                if self.pd_controller.is_converged():
                    logging.info(f"Converged after {i+1} iterations")
                    break
                    
            return results
            
        except Exception as e:
            logging.error(f"Error in optimization process: {str(e)}")
            raise

    def visualize_results(self, results, save_path):
        """Visualize optimization results."""
        try:
            plt.figure(figsize=(15, 5))
            
            # Plot error curve
            plt.subplot(1, 2, 1)
            errors = [r['error'] for r in results]
            plt.plot(errors)
            plt.title('Optimization Error')
            plt.xlabel('Iteration')
            plt.ylabel('Error')
            
            # Plot final image
            plt.subplot(1, 2, 2)
            final_image = results[-1]['image'].cpu().squeeze(0).permute(1, 2, 0).numpy()
            plt.imshow(final_image)
            plt.title('Final Result')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating visualization: {str(e)}")
            raise

def run_experiment(opt):
    """Run the experiment with detailed logging."""
    try:
        logging.info("Initializing models...")
        device = torch.device('cuda' if opt.cuda else 'cpu')
        logging.info(f"Using device: {device}")

        # Initialize models
        vgg = encoder4().to(device)
        decoder = decoder4().to(device)
        
        logging.info("Loading model weights...")
        vgg.load_state_dict(torch.load(opt.vgg_path))
        decoder.load_state_dict(torch.load(opt.decoder_path))
        
        vgg.eval()
        decoder.eval()
        logging.info("Models initialized and set to eval mode")

        # Create PD controller and optimizer
        pd_controller = FeatureSpacePDController(
            vgg_encoder=vgg, 
            decoder=decoder, 
            kp=opt.kp, 
            kd=opt.kd
        )
        optimizer = StyleTransferOptimizer(pd_controller, opt.max_iterations)

        # Create output directories
        os.makedirs(opt.output_dir, exist_ok=True)
        results_dir = os.path.join(opt.output_dir, 'results')
        plots_dir = os.path.join(opt.output_dir, 'plots')
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        logging.info(f"Created output directories")

        # Process images
        content_files = sorted(Path(opt.content_dir).glob('*.jpg'))
        style_files = sorted(Path(opt.style_dir).glob('*.jpg'))
        
        logging.info(f"Found {len(content_files)} content images and {len(style_files)} style images")

        for content_path in content_files:
            content_name = content_path.stem
            logging.info(f"Processing content image: {content_name}")
            
            content_img = load_image(str(content_path)).to(device)
            
            for style_path in style_files:
                style_name = style_path.stem
                logging.info(f"Processing style image: {style_name}")
                
                try:
                    style_img = load_image(str(style_path)).to(device)
                    
                    # Run optimization
                    results = optimizer.optimize(content_img, style_img)
                    
                    # Save final result
                    output_path = os.path.join(
                        results_dir, 
                        f'content_{content_name}_style_{style_name}.png'
                    )
                    save_image(results[-1]['image'], output_path)
                    
                    # Save visualization
                    plot_path = os.path.join(
                        plots_dir,
                        f'plot_content_{content_name}_style_{style_name}.png'
                    )
                    optimizer.visualize_results(results, plot_path)
                    
                    logging.info(f"Completed processing {content_name}-{style_name}")
                    
                except Exception as e:
                    logging.error(f"Error processing {content_name}-{style_name}: {str(e)}")
                    continue

    except Exception as e:
        logging.error(f"Experiment failed: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', required=True)
    parser.add_argument('--style_dir', required=True)
    parser.add_argument('--output_dir', default='pd_output')
    parser.add_argument('--vgg_path', default='models/vgg_r41.pth')
    parser.add_argument('--decoder_path', default='models/dec_r41.pth')
    parser.add_argument('--kp', type=float, default=0.5)
    parser.add_argument('--kd', type=float, default=0.1)
    parser.add_argument('--max_iterations', type=int, default=50)
    parser.add_argument('--cuda', action='store_true')
    
    opt = parser.parse_args()
    run_experiment(opt)

if __name__ == "__main__":
    main()



    