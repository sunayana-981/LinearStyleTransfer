import torch
from pathlib import Path
import logging
from typing import Optional, Union
import json
from tqdm import tqdm
import pandas as pd

from style_analysis import StyleAnalysisConfig, StyleAnalyzer

class WikiArtConfig(StyleAnalysisConfig):
    """Configuration specifically tailored for WikiArt dataset analysis."""
    
    def __init__(self):
        super().__init__()
        # Update style categories to match WikiArt's organization
        self.style_categories = {
            'art_movement': [
                "impressionist painting",
                "post-impressionist artwork",
                "expressionist painting",
                "art nouveau style",
                "baroque painting",
                "renaissance artwork",
                "surrealist painting",
                "cubist artwork",
                "abstract artwork",
                "pop art style"
            ],
            'technique': [
                "artwork with oil paint",
                "artwork with watercolor",
                "artwork with pencil drawing",
                "artwork with pastel colors",
                "artwork with thick brushstrokes",
                "artwork with fine details"
            ],
            'composition': [
                "artwork with landscape composition",
                "artwork with portrait composition",
                "artwork with still life arrangement",
                "artwork with abstract composition",
                "artwork with geometric composition",
                "artwork with figurative elements"
            ],
            'style_elements': [
                "artwork with bold colors",
                "artwork with muted palette",
                "artwork with dramatic lighting",
                "artwork with subtle shading",
                "artwork with visible brushwork",
                "artwork with smooth surface"
            ]
        }
        
        # WikiArt-specific settings
        self.min_images_per_category = 50  # Minimum images needed for analysis
        self.max_images_per_category = 500  # Cap to manage memory and time
        self.category_type = 'style'  # Can be 'style', 'artist', or 'genre'

class WikiArtAnalyzer:
    """Handles WikiArt dataset organization and analysis."""
    
    def __init__(self, 
                 data_dir: Union[str, Path],
                 output_dir: Union[str, Path],
                 config: Optional[WikiArtConfig] = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.config = config or WikiArtConfig()
        
        # Set up logging
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize style analyzer
        self.analyzer = StyleAnalyzer(self.config)
    
    def prepare_dataset(self):
        """
        Organize WikiArt dataset for analysis.
        Returns DataFrame with image paths and categories.
        """
        image_data = []
        
        # Walk through dataset structure
        for style_dir in tqdm(self.data_dir.glob("*"),
                            desc="Scanning dataset"):
            if not style_dir.is_dir():
                continue
            
            style_name = style_dir.name
            image_paths = list(style_dir.rglob("*.[jJ][pP][gG]"))
            
            # Filter if too few or too many images
            if len(image_paths) < self.config.min_images_per_category:
                self.logger.warning(
                    f"Skipping {style_name}: insufficient images "
                    f"({len(image_paths)} < {self.config.min_images_per_category})"
                )
                continue
            
            # Randomly sample if too many images
            if len(image_paths) > self.config.max_images_per_category:
                image_paths = np.random.choice(
                    image_paths,
                    self.config.max_images_per_category,
                    replace=False
                ).tolist()
            
            # Add to dataset
            for path in image_paths:
                image_data.append({
                    'path': str(path),
                    'style': style_name,
                    'artist': path.parent.name if self.config.category_type == 'artist' else None,
                    'genre': path.parent.parent.name if self.config.category_type == 'genre' else None
                })
        
        df = pd.DataFrame(image_data)
        self.logger.info(f"Prepared dataset with {len(df)} images from {df['style'].nunique()} styles")
        
        # Save dataset information
        df.to_csv(self.output_dir / 'dataset_info.csv', index=False)
        return df
    
    def run_analysis(self):
        """Run complete analysis pipeline on WikiArt dataset."""
        # Prepare dataset
        df = self.prepare_dataset()
        
        # Run style analysis
        embeddings_2d, labels, statistics = self.analyzer.analyze_dataset(df['path'].tolist())
        
        # Create additional WikiArt-specific visualizations
        self._create_style_period_analysis(df, embeddings_2d, labels)
        self._create_artist_influence_analysis(df, statistics)
        
        # Save results
        self.analyzer.visualize_results(
            embeddings_2d,
            labels,
            statistics,
            self.output_dir
        )
        
        # Save additional metadata
        metadata = {
            'dataset_stats': {
                'total_images': len(df),
                'total_styles': df['style'].nunique(),
                'total_artists': df['artist'].nunique() if 'artist' in df else None,
                'time_periods': self._get_time_periods(df)
            },
            'analysis_config': self.config.__dict__
        }
        
        with open(self.output_dir / 'analysis_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return embeddings_2d, labels, statistics
    
    def _create_style_period_analysis(self, df, embeddings_2d, labels):
        """Create visualizations showing evolution of styles over time periods."""
        # Implementation depends on available metadata
        pass
    
    def _create_artist_influence_analysis(self, df, statistics):
        """Analyze and visualize artist influences and style relationships."""
        # Implementation depends on available metadata
        pass
    
    def _get_time_periods(self, df):
        """Extract time period information if available in dataset."""
        # Implementation depends on available metadata
        return None

def main():
    """Main execution function with error handling."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run WikiArt style analysis')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to WikiArt dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Path to save analysis results')
    parser.add_argument('--category_type', type=str, default='style',
                      choices=['style', 'artist', 'genre'],
                      help='Primary categorization type for analysis')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for image processing')
    parser.add_argument('--max_images', type=int, default=500,
                      help='Maximum images per category')
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = WikiArtConfig()
        config.batch_size = args.batch_size
        config.max_images_per_category = args.max_images
        config.category_type = args.category_type
        
        # Initialize and run analyzer
        analyzer = WikiArtAnalyzer(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config=config
        )
        
        # Run analysis
        embeddings, labels, statistics = analyzer.run_analysis()
        
        print(f"Analysis complete. Results saved to {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise

import sys
import traceback

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)