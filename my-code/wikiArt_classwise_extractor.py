import os
import requests
import json
import time
from PIL import Image
from io import BytesIO
import zipfile
from datetime import datetime
from tqdm import tqdm
import random

class ArtDatasetCollector:
    def __init__(self, base_path: str = "art_dataset"):
        self.base_path = base_path
        self.image_path = os.path.join(base_path, "images")
        self.metadata_path = os.path.join(base_path, "metadata")
        self.processed_path = os.path.join(base_path, "processed")
        
        # Create necessary directories
        for path in [self.image_path, self.metadata_path, self.processed_path]:
            os.makedirs(path, exist_ok=True)
        
        self.collected_metadata = []
        
        # Sample artwork URLs by style (pre-collected from Google Arts & Culture)
        self.style_urls = {
            "Impressionism": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Claude_Monet%2C_Impression%2C_soleil_levant.jpg/1280px-Claude_Monet%2C_Impression%2C_soleil_levant.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Pierre-Auguste_Renoir_-_Luncheon_of_the_Boating_Party_-_Google_Art_Project.jpg/1280px-Pierre-Auguste_Renoir_-_Luncheon_of_the_Boating_Party_-_Google_Art_Project.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Claude_Monet_-_Water_Lilies_-_Google_Art_Project.jpg/1280px-Claude_Monet_-_Water_Lilies_-_Google_Art_Project.jpg",
                # Add more URLs here
            ],
            "Post-Impressionism": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Vincent_van_Gogh_-_Sunflowers_-_Google_Art_Project.jpg/1280px-Vincent_van_Gogh_-_Sunflowers_-_Google_Art_Project.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/The_Starry_Night.jpg/1280px-The_Starry_Night.jpg",
                # Add more URLs here
            ],
            "Renaissance": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Mona_Lisa.jpg/800px-Mona_Lisa.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Michelangelo_-_Creation_of_Adam.jpg/1280px-Michelangelo_-_Creation_of_Adam.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/The_Birth_of_Venus_by_Sandro_Botticelli.jpg/1280px-The_Birth_of_Venus_by_Sandro_Botticelli.jpg",
                # Add more URLs here
            ]
            # Add more styles here
        }

    def download_image(self, url: str, save_path: str) -> bool:
        """Download and save an image from a URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')  # Convert to RGB to ensure consistency
            
            # Resize if image is too large
            if max(img.size) > 1024:
                ratio = 1024 / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            img.save(save_path, 'JPEG', quality=85)
            print(f"Successfully downloaded: {save_path}")
            return True
            
        except Exception as e:
            print(f"Error downloading image {url}: {e}")
            return False

    def collect_style_images(self, style: str, urls: list, num_images: int = 10) -> list:
        """Collect images for a specific style"""
        style_path = os.path.join(self.image_path, style.lower().replace(" ", "_"))
        os.makedirs(style_path, exist_ok=True)
        
        print(f"\nCollecting images for style: {style}")
        collected_images = []
        
        # Use available URLs
        for idx, url in enumerate(urls[:num_images]):
            image_name = f"{style.lower()}_{idx:03d}.jpg"
            save_path = os.path.join(style_path, image_name)
            
            if self.download_image(url, save_path):
                metadata = {
                    "style": style,
                    "filename": image_name,
                    "original_url": url,
                    "download_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                collected_images.append(metadata)
                self.collected_metadata.append(metadata)
                
                # Save intermediate metadata
                self._save_intermediate_metadata()
            
            time.sleep(random.uniform(1, 2))  # Random delay between downloads
        
        return collected_images

    def _save_intermediate_metadata(self):
        """Save current metadata to prevent data loss"""
        if self.collected_metadata:
            metadata_file = os.path.join(
                self.metadata_path,
                f"metadata_{time.strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(metadata_file, 'w') as f:
                json.dump(self.collected_metadata, f, indent=2)

    def create_dataset(self, num_images_per_style: int = 10):
        """Create the complete dataset"""
        self.collected_metadata = []
        
        print(f"Collecting {num_images_per_style} images for each of {len(self.style_urls)} styles...")
        
        for style, urls in tqdm(self.style_urls.items()):
            style_metadata = self.collect_style_images(style, urls, num_images_per_style)
            print(f"Collected {len(style_metadata)} images for {style}")
        
        if not self.collected_metadata:
            print("Warning: No images were collected!")
            return {}
        
        # Create labels dictionary
        labels = {
            meta['filename']: list(self.style_urls.keys()).index(meta['style'])
            for meta in self.collected_metadata
        }
        
        # Save final labels
        labels_file = os.path.join(self.metadata_path, "labels.json")
        with open(labels_file, 'w') as f:
            json.dump(labels, f, indent=2)
        
        print(f"\nDataset collection complete. Total images: {len(self.collected_metadata)}")
        return labels

    def save_dataset(self, version: str = None):
        """Save the collected dataset"""
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        save_dir = os.path.join(self.processed_path, f"dataset_v{version}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save dataset info
        dataset_info = {
            'version': version,
            'creation_date': datetime.now().isoformat(),
            'num_styles': len(self.style_urls),
            'styles': list(self.style_urls.keys()),
            'total_images': len(self.collected_metadata)
        }
        
        with open(os.path.join(save_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Create archives
        print("Creating image archive...")
        with zipfile.ZipFile(os.path.join(save_dir, 'images.zip'), 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(self.image_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.image_path)
                    zipf.write(file_path, arcname)
        
        print("Creating metadata archive...")
        with zipfile.ZipFile(os.path.join(save_dir, 'metadata.zip'), 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(self.metadata_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.metadata_path)
                    zipf.write(file_path, arcname)
        
        return save_dir

def main():
    try:
        collector = ArtDatasetCollector(base_path="art_dataset")
        
        # Test download with first URL
        test_style = next(iter(collector.style_urls))
        test_url = collector.style_urls[test_style][0]
        test_path = os.path.join(collector.image_path, "test.jpg")
        
        print("Testing image download...")
        if not collector.download_image(test_url, test_path):
            raise Exception("Unable to download test image")
        os.remove(test_path)  # Clean up test image
        
        # Create dataset
        labels = collector.create_dataset(num_images_per_style=10)
        
        if labels:
            save_dir = collector.save_dataset(version="1.0.0")
            print(f"Dataset saved to: {save_dir}")
            
            # Print summary
            print("\nDataset Summary:")
            print(f"Total styles: {len(collector.style_urls)}")
            print(f"Total images: {len(labels)}")
            print(f"Images per style: {len(labels) / len(collector.style_urls):.1f}")
        else:
            print("Dataset creation failed - no images were collected")
        
    except Exception as e:
        print(f"Error during dataset collection: {e}")
        raise

if __name__ == "__main__":
    main()