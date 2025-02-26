import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

class FeatureSpacePDController:
    def __init__(self, vgg_encoder, decoder, layer='r41', kp=0.5, kd=0.1, target_threshold=0.01):
        self.vgg = vgg_encoder
        self.decoder = decoder
        self.layer = layer  # Specify which VGG layer to use
        self.kp = kp
        self.kd = kd
        self.target_threshold = target_threshold
        self.previous_error = None
        self.history = []
        logging.info(f"Initialized PD Controller with layer={layer}, kp={kp}, kd={kd}")

    def get_features(self, image):
        """Extract features from image."""
        try:
            with torch.no_grad():
                features_dict = self.vgg(image)
                features = features_dict[self.layer]  # Extract specific layer features
                logging.info(f"Extracted features shape: {features.shape}")
                return features
        except Exception as e:
            logging.error(f"Error extracting features: {str(e)}")
            raise

    def optimize_step(self, current_features, target_features):
        """Perform optimization step."""
        try:
            # Compute error using normalized features
            current_norm = torch.norm(current_features, dim=(2, 3), keepdim=True)
            target_norm = torch.norm(target_features, dim=(2, 3), keepdim=True)
            
            current_normalized = current_features / (current_norm + 1e-7)
            target_normalized = target_features / (target_norm + 1e-7)
            
            error = F.mse_loss(current_normalized, target_normalized)
            
            # Compute control signal
            p_term = self.kp * (target_features - current_features)
            
            if self.previous_error is not None:
                d_term = self.kd * (error - self.previous_error)
                control = p_term - d_term.view(-1, 1, 1, 1)
            else:
                control = p_term
                
            updated_features = current_features + control
            
            self.previous_error = error
            
            logging.debug(f"Step error: {error.item():.6f}")
            return updated_features, control, error
            
        except Exception as e:
            logging.error(f"Error in optimization step: {str(e)}")
            raise

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
                
                logging.info(f"Iteration {i+1} error: {error.item():.6f}")
                
                if error.item() < self.pd_controller.target_threshold:
                    logging.info("Converged!")
                    break
                    
            return results
            
        except Exception as e:
            logging.error(f"Error in optimization process: {str(e)}")
            raise

    def visualize_results(self, results, save_path):
        """
        Create visualization of the optimization process.
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Plot error trajectory
        plt.subplot(2, 2, 1)
        errors = [r['error'] for r in results]
        plt.plot(errors, 'b-', label='Feature Error')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title('Optimization Error')
        plt.legend()
        
        # Plot control signal magnitude
        plt.subplot(2, 2, 2)
        controls = [r['control_norm'] for r in results]
        plt.plot(controls, 'r-', label='Control Magnitude')
        plt.xlabel('Iteration')
        plt.ylabel('Control Signal Norm')
        plt.title('Control Signal History')
        plt.legend()
        
        # Show image progression
        plt.subplot(2, 1, 2)
        num_samples = min(5, len(results))
        sample_indices = np.linspace(0, len(results)-1, num_samples, dtype=int)
        
        images = [results[idx]['image'] for idx in sample_indices]
        image_grid = make_grid(torch.stack(images), nrow=num_samples)
        plt.imshow(image_grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.title('Image Progression')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()