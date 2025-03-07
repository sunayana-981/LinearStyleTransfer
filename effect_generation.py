
import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

class ImageEffectGenerator:
    @staticmethod
    def apply_contrast(image, intensity):
        """Apply contrast adjustment"""
        alpha = 1.0 + (intensity - 1.0) * 0.5
        return np.clip(alpha * image, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_saturation(image, intensity):
        """Modify color saturation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * intensity, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def apply_texture(image, intensity):
        """Enhance texture details"""
        blur = cv2.GaussianBlur(image, (0, 0), 3)
        detail = image.astype(np.float32) - blur.astype(np.float32)
        enhanced = image.astype(np.float32) + detail * intensity
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_edges(image, intensity):
        """Enhance edges"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = cv2.cvtColor(edges.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        enhanced = cv2.addWeighted(image, 1, edges, intensity * 0.5, 0)
        return np.clip(enhanced, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_posterization(image, intensity):
        """
        Apply posterization effect with fixed integer types.
        Higher intensity = fewer levels = more extreme posterization
        """
        max_levels = 32
        min_levels = 2
        num_levels = int(max_levels - ((intensity - 1.0) * (max_levels - min_levels) / 4.0))
        num_levels = max(min_levels, min(max_levels, num_levels))
        
        # Create lookup table for posterization
        indices = np.arange(0, 256)
        divider = np.linspace(0, 255, num_levels + 1)[1]
        quantiz = np.linspace(0, 255, num_levels).astype(np.int32)
        indices = quantiz[np.minimum(np.array(indices/divider, dtype=np.int32), num_levels-1)]
        
        # Apply to each channel
        posterized = image.copy()
        for i in range(3):
            posterized[:,:,i] = indices[image[:,:,i]]
        
        return posterized

def process_all_effects(style_dir, base_output_dir, intensity_levels):
    """
    Process all images with all effects and intensity levels.
    """
    effects = {
        'contrast': ImageEffectGenerator.apply_contrast,
        'saturation': ImageEffectGenerator.apply_saturation,
        'texture': ImageEffectGenerator.apply_texture,
        'edges': ImageEffectGenerator.apply_edges,
        'poster': ImageEffectGenerator.apply_posterization
    }

    os.makedirs(base_output_dir, exist_ok=True)
    style_files = [f for f in os.listdir(style_dir) if f.endswith(('.jpg', '.png'))]
    
    for effect_name, effect_func in effects.items():
        print(f"\nProcessing {effect_name} effect...")
        effect_dir = os.path.join(base_output_dir, f"{effect_name}_styles")
        
        for intensity in tqdm(intensity_levels, desc=f"Processing {effect_name} levels"):
            intensity_dir = os.path.join(effect_dir, f"{effect_name}_{intensity:.1f}")
            os.makedirs(intensity_dir, exist_ok=True)
            
            for style_file in style_files:
                try:
                    image_path = os.path.join(style_dir, style_file)
                    image = cv2.imread(image_path)
                    
                    if image is not None:
                        processed = effect_func(image, intensity)
                        base_name = os.path.splitext(style_file)[0]
                        output_name = f"{base_name}_{effect_name}_{intensity:.1f}.png"
                        output_path = os.path.join(intensity_dir, output_name)
                        cv2.imwrite(output_path, processed)
                except Exception as e:
                    print(f"Error processing {style_file} with {effect_name}: {str(e)}")
                    continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stylePath", default="data/style/",
                        help="path to style images")
    parser.add_argument("--outputPath", default="data/processed_styles/",
                        help="path to save processed images")
    parser.add_argument("--intensity_levels", type=float, nargs='+',
                        default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                        help="intensity levels for effects")

    opt = parser.parse_args()

    print("Starting image processing with:")
    print(f"Input directory: {opt.stylePath}")
    print(f"Output directory: {opt.outputPath}")
    print(f"Intensity levels: {opt.intensity_levels}")

    process_all_effects(opt.stylePath, opt.outputPath, opt.intensity_levels)
    print("\nAll processing complete!")

if __name__ == "__main__":
    main()