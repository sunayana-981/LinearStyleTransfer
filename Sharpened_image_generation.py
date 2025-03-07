import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def apply_enhanced_sharpening(image, intensity):
    """
    Apply enhanced sharpening using custom kernels with stronger effects.
    
    Args:
        image: Input image
        intensity: Controls the strength of sharpening (1.0 to 5.0)
    Returns:
        Sharpened image
    """
    # Convert to float32 for processing
    img_float = image.astype(np.float32)
    
    # Define a more aggressive sharpening kernel
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]]) * intensity
    
    # Apply the sharpening kernel
    sharpened = cv2.filter2D(img_float, -1, kernel)
    
    # Add additional edge enhancement
    edges = cv2.Laplacian(img_float, cv2.CV_32F)
    sharpened += edges * (intensity * 0.2)
    
    # Enhance local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for i in range(3):  # Process each color channel
        sharpened[:,:,i] = clahe.apply(np.uint8(np.clip(sharpened[:,:,i], 0, 255)))
    
    # Final normalization and conversion
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def generate_sharpening_dataset(style_dir, output_dir, 
                              intensity_levels=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]):
    """
    Generate a dataset of sharpened images with different intensity levels.
    
    Args:
        style_dir: Directory containing input style images
        output_dir: Directory to save sharpened images
        intensity_levels: List of intensity values for sharpening effect
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Generating enhanced sharpened dataset...")

    # Process each intensity level
    for intensity in tqdm(intensity_levels, desc="Processing intensity levels"):
        # Create directory for this intensity
        sharp_dir = os.path.join(output_dir, f"sharp_{intensity:.1f}")
        os.makedirs(sharp_dir, exist_ok=True)

        # Process each style image
        for style_file in os.listdir(style_dir):
            if style_file.endswith((".jpg", ".png")):
                # Read image
                style_path = os.path.join(style_dir, style_file)
                image = cv2.imread(style_path)
                
                if image is not None:
                    # Apply enhanced sharpening
                    sharpened = apply_enhanced_sharpening(image, intensity)

                    # Generate output filename
                    base_name = os.path.splitext(style_file)[0]
                    output_name = f"{base_name}_sharp_{intensity:.1f}.png"
                    output_path = os.path.join(sharp_dir, output_name)

                    # Save sharpened image
                    cv2.imwrite(output_path, sharpened)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stylePath", default="data/style/",
                        help="path to style images")
    parser.add_argument("--outputPath", default="data/sharpened_styles/",
                        help="path to save sharpened images")
    parser.add_argument("--intensity_levels", type=float, nargs='+',
                        default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                        help="intensity levels for sharpening")

    opt = parser.parse_args()

    # Print configuration
    print("Generating enhanced sharpened dataset with:")
    print(f"Input directory: {opt.stylePath}")
    print(f"Output directory: {opt.outputPath}")
    print(f"Intensity levels: {opt.intensity_levels}")

    # Generate sharpened dataset
    generate_sharpening_dataset(opt.stylePath, opt.outputPath, opt.intensity_levels)
    
    print("Dataset generation complete!")