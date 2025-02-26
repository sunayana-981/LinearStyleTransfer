import os
import cv2
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import segmentation, color, graph, morphology

class StyleTransferMaskGenerator:
    def __init__(self, input_dir, output_dir, mode="high_quality"):
        """
        Initialize Style Transfer Mask Generator.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save generated masks
            mode: Quality mode ('high_quality', 'simple', or 'sketch')
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.mode = mode
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Color palette - bright, contrasting colors for style transfer
        self.colors = [
            (0, 0, 0),        # Background (black)
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green  
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
            (0, 128, 255),    # Light Blue
            (255, 255, 255),  # White
        ]
    
    def create_high_quality_mask(self, image):
        """
        Create a high-quality mask using multiple segmentation techniques.
        This uses felzenszwalb + region adjacency graph for cleaner boundaries.
        """
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # Convert to LAB color space for better segmentation
        image_lab = color.rgb2lab(image)
        
        # Step 1: Initial segmentation with felzenszwalb
        segments = segmentation.felzenszwalb(
            image, scale=100, sigma=0.5, min_size=50)
            
        # Step 2: Clean small regions using Region Adjacency Graph
        g = graph.rag_mean_color(image, segments)
        segments = graph.cut_threshold(segments, g, 30)
        
        # Step 3: Further merge small regions
        merged_segments = self.merge_small_regions(segments, min_size_percent=0.02)
        
        # Optional: Apply morphological operations to clean boundaries
        for region_id in np.unique(merged_segments):
            region_mask = (merged_segments == region_id).astype(np.uint8)
            # Close small holes
            kernel = np.ones((3, 3), np.uint8)
            region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel)
            merged_segments[region_mask > 0] = region_id
        
        # Create colored mask
        colored_mask = self.create_colored_mask(merged_segments)
        
        return colored_mask
        
    def create_simple_mask(self, image):
        """Create a simpler mask with fewer regions."""
        # Reduce noise with bilateral filter
        image_filtered = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Convert to RGB if needed
        if len(image_filtered.shape) == 2:
            image_filtered = cv2.cvtColor(image_filtered, cv2.COLOR_GRAY2RGB)
        elif image_filtered.shape[2] == 4:
            image_filtered = cv2.cvtColor(image_filtered, cv2.COLOR_RGBA2RGB)
        
        # Use mean shift segmentation
        shifted = cv2.pyrMeanShiftFiltering(image_filtered, 21, 51)
        
        # Convert to grayscale and threshold to get fewer regions
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(thresh)
        
        # Merge small regions
        merged_labels = self.merge_small_regions(labels, min_size_percent=0.05)
        
        # Create colored mask
        colored_mask = self.create_colored_mask(merged_labels)
        
        return colored_mask
    
    def create_sketch_mask(self, image):
        """Create a mask that looks like a sketch or drawing."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Dilate edges to make them more prominent
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Create cartoon-like effect
        bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        cartoon = cv2.bitwise_and(bilateral, bilateral, mask=255-dilated_edges)
        
        # Convert to grayscale and threshold
        cartoon_gray = cv2.cvtColor(cartoon, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(cartoon_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(thresh)
        
        # Create colored mask
        colored_mask = self.create_colored_mask(labels)
        
        return colored_mask
    
    def merge_small_regions(self, segments, min_size_percent=0.02):
        """Merge small regions into neighboring larger regions."""
        # Calculate sizes
        unique_segments, counts = np.unique(segments, return_counts=True)
        total_pixels = segments.size
        min_size = total_pixels * min_size_percent
        
        # Find small regions
        small_segments = unique_segments[counts < min_size]
        
        if len(small_segments) == 0:
            return segments
            
        # Create a new segmentation map
        merged = segments.copy()
        
        # Process each small region
        for small_segment in small_segments:
            # Skip if already merged
            if np.sum(merged == small_segment) == 0:
                continue
                
            # Get the small region mask
            small_mask = (merged == small_segment)
            
            # Dilate to find neighbors
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(small_mask.astype(np.uint8), kernel) > 0
            
            # Find neighboring segments
            neighbor_mask = dilated & ~small_mask
            neighbors = np.unique(merged[neighbor_mask])
            
            # Remove background or other small segments from neighbors
            neighbors = [n for n in neighbors if n != small_segment and n not in small_segments]
            
            if len(neighbors) > 0:
                # Choose the largest neighbor
                largest_neighbor = max(neighbors, key=lambda n: np.sum(merged == n))
                # Merge with the largest neighbor
                merged[small_mask] = largest_neighbor
            
        return merged
    
    def create_colored_mask(self, segments):
        """Convert segmentation to a colored mask."""
        # Get unique segments
        unique_segments = np.unique(segments)
        
        # Create colored mask
        h, w = segments.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i, segment in enumerate(unique_segments):
            # Skip background (usually 0)
            if segment == 0 and len(unique_segments) > 1:
                colored_mask[segments == segment] = (0, 0, 0)  # Black for background
            else:
                # Assign a color from our palette
                color_idx = (i % (len(self.colors) - 1)) + 1  # Skip black (index 0) for non-background
                colored_mask[segments == segment] = self.colors[color_idx]
        
        return colored_mask
    
    def post_process_mask(self, mask):
        """Apply post-processing to improve mask quality."""
        # Convert to grayscale
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Threshold to binary
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw filled contours with different colors
        result = np.zeros_like(mask)
        for i, contour in enumerate(contours):
            color_idx = (i % (len(self.colors) - 1)) + 1  # Skip black
            cv2.drawContours(result, [contour], -1, self.colors[color_idx], -1)
        
        return result
    
    def process_image(self, image_path):
        """Process a single image to generate a mask."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                return None
                
            # Generate mask based on selected mode
            if self.mode == "high_quality":
                mask = self.create_high_quality_mask(image)
            elif self.mode == "simple":
                mask = self.create_simple_mask(image)
            elif self.mode == "sketch":
                mask = self.create_sketch_mask(image)
            else:
                mask = self.create_high_quality_mask(image)
                
            # Post-process to improve quality if needed
            # mask = self.post_process_mask(mask)
                
            return mask
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def process_directory(self):
        """Process all images in the input directory."""
        if not os.path.exists(self.input_dir):
            print(f"Input directory not found: {self.input_dir}")
            return
            
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(
                [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) 
                 if f.lower().endswith(ext)]
            )
            
        if not image_files:
            print(f"No image files found in {self.input_dir}")
            return
            
        print(f"Found {len(image_files)} images. Generating masks...")
        
        for image_path in tqdm(image_files):
            # Get filename without extension
            filename = os.path.basename(image_path)
            name, _ = os.path.splitext(filename)
            
            # Process image
            mask = self.process_image(image_path)
            if mask is None:
                continue
                
            # Save mask
            mask_path = os.path.join(self.output_dir, f"{name}_mask.png")
            cv2.imwrite(mask_path, mask)
            
            # Also save corresponding binary masks
            self.save_binary_masks(mask, name)
                
        print(f"Mask generation complete. Masks saved to {self.output_dir}")
    
    def save_binary_masks(self, colored_mask, base_name):
        """Save individual binary masks for each color segment."""
        # Create a subdirectory for binary masks
        binary_dir = os.path.join(self.output_dir, f"{base_name}_segments")
        os.makedirs(binary_dir, exist_ok=True)
        
        # Extract unique colors from the mask
        unique_colors = np.unique(colored_mask.reshape(-1, 3), axis=0)
        
        # Save a binary mask for each color
        for i, color in enumerate(unique_colors):
            # Create binary mask
            binary = np.all(colored_mask == color.reshape(1, 1, 3), axis=2).astype(np.uint8) * 255
            
            # Save mask
            binary_path = os.path.join(binary_dir, f"segment_{i}.png")
            cv2.imwrite(binary_path, binary)

def main():
    parser = argparse.ArgumentParser(description="Generate high-quality masks for style transfer")
    parser.add_argument("--input_dir", type=str, default="concept_sets/sunny_vs_cloudy/positive", help="Directory containing input images")
    parser.add_argument("--output_dir", type=str,default="concept_sets/sunny_vs_cloudy/positive_masks_unet", help="Directory to save generated masks")
    parser.add_argument("--mode", type=str, default="high_quality", 
                       choices=["high_quality", "simple", "sketch"],
                       help="Mask generation mode")
    
    args = parser.parse_args()
    
    generator = StyleTransferMaskGenerator(
        args.input_dir,
        args.output_dir,
        mode=args.mode
    )
    
    generator.process_directory()

if __name__ == "__main__":
    main()


