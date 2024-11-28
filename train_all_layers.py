import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import traceback
import time
from datetime import datetime
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from libs.Matrix import MulLayer
from libs.Criterion import LossCriterion
from libs.models import encoder1, encoder2
from libs.models import decoder1, decoder2
from libs.models import encoder5 as loss_network
from multiprocessing import freeze_support
from PIL import Image
from torchvision import transforms
import logging
import sys
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, loadSize, fineSize, debug=False):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.debug = debug
        self.loadSize = loadSize
        self.fineSize = fineSize
        
        # Suppress PIL warnings about large images
        Image.MAX_IMAGE_PIXELS = None
        
        # Filter valid images during initialization
        self.image_list = []
        total_files = 0
        valid_files = 0
        
        for x in os.listdir(data_path):
            total_files += 1
            if self._is_image_file(x):
                full_path = os.path.join(data_path, x)
                if os.path.getsize(full_path) > 100:  # Skip obviously corrupted files
                    self.image_list.append(x)
                    valid_files += 1
        
        if debug:
            logging.info(f"Found {valid_files}/{total_files} valid images in {data_path}")
        
        # Define transforms with error handling
        self.transform = transforms.Compose([
            transforms.Resize(loadSize),
            transforms.RandomCrop(fineSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def _is_image_file(self, filename):
        """Check if a file is an allowed image type."""
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff']
        return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)
    
    def _load_image_safely(self, path):
        """Safely load and verify an image with multiple fallback attempts."""
        try:
            # Try to open with PIL's default behavior
            with Image.open(path) as img:
                # Verify image can be loaded completely
                img.verify()
            
            # Reopen image after verify
            with Image.open(path) as img:
                # Convert to RGB mode
                img = img.convert('RGB')
                
                # Basic size sanity check
                width, height = img.size
                if width < 32 or height < 32:
                    if self.debug:
                        logging.warning(f"Image too small: {path}")
                    return None
                    
                if width > 10000 or height > 10000:
                    if self.debug:
                        logging.warning(f"Image too large: {path}")
                    return None
                
                return img
                
        except (OSError, IOError) as e:
            if self.debug:
                logging.warning(f"Error loading image {path}: {str(e)}")
            return None
            
        except Exception as e:
            if self.debug:
                logging.warning(f"Unexpected error loading {path}: {str(e)}")
            return None

    def __getitem__(self, index):
        max_attempts = 5
        attempts = 0
        
        while attempts < max_attempts:
            try:
                # Get image path
                img_name = self.image_list[index]
                path = os.path.join(self.data_path, img_name)
                
                # Try to load image
                img = self._load_image_safely(path)
                
                if img is not None:
                    # Apply transforms
                    try:
                        img_tensor = self.transform(img)
                        
                        # Verify tensor is valid
                        if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
                            raise ValueError("Invalid tensor values detected")
                            
                        return img_tensor, path
                        
                    except Exception as e:
                        if self.debug:
                            logging.warning(f"Transform error for {path}: {str(e)}")
                
                # If we get here, try next image
                index = (index + 1) % len(self.image_list)
                attempts += 1
                
            except Exception as e:
                if self.debug:
                    logging.warning(f"Error processing index {index}: {str(e)}")
                index = (index + 1) % len(self.image_list)
                attempts += 1
        
        # If all attempts fail, return a valid blank tensor
        if self.debug:
            logging.warning(f"Failed to load any valid image after {max_attempts} attempts")
        return torch.zeros(3, self.fineSize, self.fineSize), "none"

    def __len__(self):
        return len(self.image_list)

def load_model_safely(model, path, device):
    """Safely load a model with error handling."""
    try:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        return True
    except Exception as e:
        logging.error(f"Error loading model from {path}: {str(e)}")
        return False

def save_checkpoint(state, filename):
    """Safely save a checkpoint with error handling."""
    try:
        torch.save(state, filename)
        logging.info(f"Successfully saved checkpoint: {filename}")
        return True
    except Exception as e:
        logging.error(f"Error saving checkpoint {filename}: {str(e)}")
        return False

def train():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--vgg1_dir", default='trained_models/encoder_r11.pth')
    parser.add_argument("--vgg2_dir", default='trained_models/encoder_r21.pth')
    parser.add_argument("--loss_network_dir", default='models/vgg_r51.pth')
    parser.add_argument("--decoder1_dir", default='trained_models/decoder_r11.pth')
    parser.add_argument("--decoder2_dir", default='trained_models/decoder_r21.pth')
    parser.add_argument("--stylePath", default="datasets/wikiArt/")
    parser.add_argument("--contentPath", default="datasets/coco2014/images/train2014/")
    parser.add_argument("--outf", default="trainingOutput/")
    
    # Training parameters
    parser.add_argument("--content_layers", default="r11,r21")
    parser.add_argument("--style_layers", default="r11,r21")
    parser.add_argument("--batchSize", type=int, default=8)
    parser.add_argument("--niter", type=int, default=100000)
    parser.add_argument('--loadSize', type=int, default=300)
    parser.add_argument('--fineSize', type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--content_weight", type=float, default=1.0)
    parser.add_argument("--style_weight", type=float, default=0.02)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--debug", action='store_true')
    
    opt = parser.parse_args()
    opt.content_layers = opt.content_layers.split(',')
    opt.style_layers = opt.style_layers.split(',')
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    opt.outf = os.path.join(opt.outf, timestamp)
    os.makedirs(opt.outf, exist_ok=True)
    
    # Set up device
    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Initialize datasets with error handling
    try:
        content_dataset = Dataset(opt.contentPath, opt.loadSize, opt.fineSize, debug=opt.debug)
        style_dataset = Dataset(opt.stylePath, opt.loadSize, opt.fineSize, debug=opt.debug)
        
        content_loader = torch.utils.data.DataLoader(
            dataset=content_dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )
        
        style_loader = torch.utils.data.DataLoader(
            dataset=style_dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )
    except Exception as e:
        logging.error(f"Error initializing datasets: {str(e)}")
        return

    # Initialize models
    vgg5 = loss_network().to(device)
    vgg1 = encoder1().to(device)
    vgg2 = encoder2().to(device)
    dec1 = decoder1().to(device)
    dec2 = decoder2().to(device)
    
    encoders = {'r11': vgg1, 'r21': vgg2}
    decoders = {'r11': dec1, 'r21': dec2}
    matrix = {'r11': MulLayer('r11').to(device), 'r21': MulLayer('r21').to(device)}
    
    # Load pre-trained models
    model_paths = {
        'vgg1': opt.vgg1_dir,
        'vgg2': opt.vgg2_dir,
        'dec1': opt.decoder1_dir,
        'dec2': opt.decoder2_dir,
        'vgg5': opt.loss_network_dir
    }
    
    for name, path in model_paths.items():
        if not os.path.exists(path):
            logging.error(f"Model file not found: {path}")
            return
    
    models = {'vgg1': vgg1, 'vgg2': vgg2, 'dec1': dec1, 'dec2': dec2, 'vgg5': vgg5}
    for name, model in models.items():
        if not load_model_safely(model, model_paths[name], device):
            return

    # Initialize criterion and optimizers
    criterion = LossCriterion(
        style_layers=opt.style_layers,
        content_layers=opt.content_layers,
        style_weight=opt.style_weight,
        content_weight=opt.content_weight
    ).to(device)
    
    optimizers = {
        layer: optim.Adam(matrix[layer].parameters(), lr=opt.lr)
        for layer in matrix.keys()
    }

    # Training loop
    logging.info("Starting training...")
    start_time = time.time()
    best_loss = float('inf')
    running_loss = {'total': 0, 'style': 0, 'content': 0}
    
    try:
        for epoch in range(1, opt.niter + 1):
            content_iter = iter(content_loader)
            style_iter = iter(style_loader)
            
            pbar = tqdm(range(min(len(content_loader), len(style_loader))),
                       desc=f"Epoch {epoch}")
            
            for i in pbar:
                # Zero gradients
                for optimizer in optimizers.values():
                    optimizer.zero_grad()

                try:
                    content, _ = next(content_iter)
                    style, _ = next(style_iter)
                except StopIteration:
                    break

                # Move to device
                content = content.to(device)
                style = style.to(device)

                # Process each layer
                total_loss = 0
                style_loss = 0
                content_loss = 0

                for layer in matrix:
                    encoder = encoders[layer]
                    decoder = decoders[layer]

                    # Forward pass
                    with torch.no_grad():
                        sF = encoder(style)
                        cF = encoder(content)

                    # Style transfer
                    feature, _ = matrix[layer](cF, sF)
                    transfer = decoder(feature)

                    # Compute losses
                    with torch.no_grad():
                        sF_loss = vgg5(style)
                        cF_loss = vgg5(content)
                    tF = vgg5(transfer)

                    loss, s_loss, c_loss = criterion(tF, sF_loss, cF_loss)
                    
                    # Backward pass
                    loss.backward()
                    optimizers[layer].step()

                    # Update losses
                    total_loss += loss.item()
                    style_loss += s_loss.item()
                    content_loss += c_loss.item()

                # Update running losses
                running_loss['total'] += total_loss
                running_loss['style'] += style_loss
                running_loss['content'] += content_loss

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{total_loss:.4f}",
                    'style': f"{style_loss:.4f}",
                    'content': f"{content_loss:.4f}"
                })

                # Log and save
                if i % opt.log_interval == 0:
                    avg_loss = {k: v / opt.log_interval for k, v in running_loss.items()}
                    
                    logging.info(
                        f"Epoch: [{epoch}][{i}/{len(content_loader)}]\n"
                        f"Average Loss: {avg_loss['total']:.4f} "
                        f"(Style: {avg_loss['style']:.4f}, "
                        f"Content: {avg_loss['content']:.4f})"
                    )

                    # Reset running losses
                    running_loss = {'total': 0, 'style': 0, 'content': 0}

                    # Save best model
                    if avg_loss['total'] < best_loss:
                        best_loss = avg_loss['total']
                        for layer in matrix:
                            save_checkpoint(
                                matrix[layer].state_dict(),
                                f'{opt.outf}/best_{layer}.pth'
                            )

 # Save checkpoints and samples
                if i % opt.save_interval == 0:
                    # Save sample images
                    with torch.no_grad():
                        transfer = transfer.clamp(0, 1)
                        concat = torch.cat([content, style, transfer], dim=0)
                        vutils.save_image(
                            concat,
                            f'{opt.outf}/sample_epoch{epoch}_iter{i}.png',
                            normalize=True,
                            nrow=opt.batchSize
                        )

                    # Save checkpoints
                    for layer in matrix:
                        save_checkpoint({
                            'epoch': epoch,
                            'iteration': i,
                            'state_dict': matrix[layer].state_dict(),
                            'optimizer': optimizers[layer].state_dict(),
                            'loss': total_loss,
                        }, f'{opt.outf}/checkpoint_{layer}_epoch{epoch}_iter{i}.pth')

    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user. Saving current state...")
        for layer in matrix:
            save_checkpoint({
                'epoch': epoch,
                'iteration': i,
                'state_dict': matrix[layer].state_dict(),
                'optimizer': optimizers[layer].state_dict(),
                'loss': total_loss,
            }, f'{opt.outf}/interrupted_{layer}.pth')
        logging.info("State saved successfully.")

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        traceback.print_exc()

if __name__ == '__main__':
    try:
        freeze_support()
        train()
    except Exception as e:
        logging.error(f"Error during script execution: {str(e)}")
        traceback.print_exc()