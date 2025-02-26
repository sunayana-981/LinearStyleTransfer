import os
from pathlib import Path
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the CLIPGuidedWarmthController from previous code
from clip_warmth import CleanWarmthController

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = output_dir / f"style_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def list_image_files(directory):
    """List all supported image files in directory"""
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    directory = Path(directory)
    
    image_files = []
    for ext in supported_extensions:
        image_files.extend(directory.glob(f'*{ext}'))
        image_files.extend(directory.glob(f'*{ext.upper()}'))
    
    # Log found files
    logging.info(f"Found files in {directory}:")
    for file in image_files:
        logging.info(f"  {file.name}")
        
    return image_files

def process_single_style(style_path, output_dir, controller):
    """Process a single style image and save variations"""
    try:
        # Create output subdirectory based on style name
        style_name = style_path.stem
        style_output_dir = output_dir / style_name
        style_output_dir.mkdir(exist_ok=True)
        
        logging.info(f"Processing {style_path}")
        
        # Load and process image
        image = Image.open(style_path).convert('RGB')
        logging.info(f"Loaded image size: {image.size}")
        
        # Generate variations
        variations, scores = controller.create_warmth_variations(image)
        
        # Save variations and original
        results = []
        image.save(style_output_dir / f"{style_name}_original.png")
        
        for i, (variation, score) in enumerate(zip(variations, scores)):
            variation_path = style_output_dir / f"{style_name}_warm_v{i}_score{score:.2f}.png"
            variation.save(variation_path)
            results.append({
                'variation': i,
                'score': float(score),
                'path': str(variation_path)
            })
        
        # Save metadata
        metadata = {
            'original_path': str(style_path),
            'variations': results,
            'average_score': sum(scores) / len(scores)
        }
        
        with open(style_output_dir / f"{style_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return True, style_name, metadata
        
    except Exception as e:
        logging.error(f"Error processing {style_path}: {str(e)}", exc_info=True)
        return False, style_path.stem, str(e)

def process_style_directory(style_dir, output_dir, num_variations=5, batch_size=1):
    """
    Process all style images in a directory using CLIP-guided warmth controller
    
    Args:
        style_dir (str or Path): Directory containing style images
        output_dir (str or Path): Directory to save processed images
        num_variations (int): Number of variations to generate per style
        batch_size (int): Number of images to process in parallel
    """
    style_dir = Path(style_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting style processing for directory: {style_dir}")
    
    # List all image files
    style_images = list_image_files(style_dir)
    logger.info(f"Found {len(style_images)} style images to process")
    
    if len(style_images) == 0:
        logger.error(f"No image files found in {style_dir}. Please check the directory path and file extensions.")
        return
    
    # Initialize controller
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    controller = CleanWarmthController(device=device)
    
    # Process images with progress bar
    success_count = 0
    failure_count = 0
    results = []
    
    # Process images sequentially if batch_size is 1
    if batch_size == 1:
        for style_path in tqdm(style_images, desc="Processing styles"):
            success, style_name, result = process_single_style(style_path, output_dir, controller)
            if success:
                success_count += 1
                results.append(result)
                logger.info(f"Successfully processed {style_name}")
            else:
                failure_count += 1
                logger.error(f"Failed to process {style_name}: {result}")
    else:
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_style = {
                executor.submit(process_single_style, style_path, output_dir, controller): style_path
                for style_path in style_images
            }
            
            for future in tqdm(as_completed(future_to_style), total=len(style_images), 
                             desc="Processing styles"):
                style_path = future_to_style[future]
                try:
                    success, style_name, result = future.result()
                    if success:
                        success_count += 1
                        results.append(result)
                        logger.info(f"Successfully processed {style_name}")
                    else:
                        failure_count += 1
                        logger.error(f"Failed to process {style_name}: {result}")
                except Exception as e:
                    failure_count += 1
                    logger.error(f"Error processing {style_path.name}: {str(e)}")
    
    # Save overall results
    summary = {
        'total_processed': len(style_images),
        'successful': success_count,
        'failed': failure_count,
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    with open(output_dir / 'processing_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Processing complete. Successful: {success_count}, Failed: {failure_count}")
    return summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process style images with CLIP-guided warmth')
    parser.add_argument('--style_dir', type=str, required=True, help='Directory containing style images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed images')
    parser.add_argument('--num_variations', type=int, default=5, help='Number of variations per style')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images to process in parallel')
    
    args = parser.parse_args()
    
    summary = process_style_directory(
        args.style_dir,
        args.output_dir,
        num_variations=args.num_variations,
        batch_size=args.batch_size
    )