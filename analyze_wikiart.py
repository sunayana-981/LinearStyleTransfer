import argparse
from pathlib import Path
import logging
from style_analysis import StyleUnderstandingAnalyzer
import torch
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikiArtRunner:
    """Handles the analysis of WikiArt dataset using StyleUnderstandingAnalyzer."""
    
    def __init__(self, data_dir, output_dir, max_images_per_style=100):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_images_per_style = max_images_per_style
        
        # Initialize the analyzer
        self.analyzer = StyleUnderstandingAnalyzer()
        
    def collect_dataset_info(self):
        """Analyzes the structure of the WikiArt dataset."""
        logger.info("Analyzing dataset structure...")
        
        dataset_info = {
            'styles': {},
            'total_images': 0
        }
        
        # Collect information about each style directory
        for style_dir in self.data_dir.iterdir():
            if not style_dir.is_dir():
                continue
                
            style_name = style_dir.name
            image_files = list(style_dir.glob("**/*.jpg")) + list(style_dir.glob("**/*.png"))
            
            if len(image_files) > 0:
                dataset_info['styles'][style_name] = {
                    'image_count': len(image_files),
                    'sample_paths': [str(p) for p in image_files[:self.max_images_per_style]]
                }
                dataset_info['total_images'] += len(image_files)
        
        logger.info(f"Found {len(dataset_info['styles'])} styles with "
                   f"{dataset_info['total_images']} total images")
        
        return dataset_info
    
    def analyze_dataset(self):
        """
        Runs the complete analysis pipeline on WikiArt dataset.
        
        Returns:
            tuple: (embeddings_2d, labels, style_consistency)
                - embeddings_2d: 2D array of style embeddings after dimensionality reduction
                - labels: List of style labels corresponding to each embedding
                - style_consistency: Dictionary containing consistency metrics for each style
        """
        # First, collect dataset information
        dataset_info = self.collect_dataset_info()
        
        # Save dataset information to a JSON file for later reference
        with open(self.output_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Begin the style embedding analysis
        logger.info("Analyzing style embeddings...")
        all_image_paths = []
        all_style_labels = []
        
        # Collect paths and labels for each style
        for style_name, style_data in dataset_info['styles'].items():
            paths = [Path(p) for p in style_data['sample_paths']]
            all_image_paths.extend(paths)
            all_style_labels.extend([style_name] * len(paths))
        
        logger.info(f"Processing {len(all_image_paths)} images...")
        
        # Get embeddings and create visualization
        try:
            embeddings_2d, labels = self.analyzer.analyze_style_dataset(all_image_paths)
            
            # Create and save the visualization
            fig = self.analyzer.visualize_style_space(embeddings_2d, labels)
            fig.savefig(str(self.output_dir / "style_space.png"))
            
            # Analyze consistency for each style
            logger.info("Analyzing style consistency...")
            style_consistency = {}
            
            # Process each style's consistency metrics
            for style_name, style_data in tqdm(dataset_info['styles'].items(),
                                            desc="Analyzing styles"):
                consistency = self.analyzer.evaluate_style_consistency(
                    style_data['sample_paths']
                )
                style_consistency[style_name] = consistency
            
            # Save consistency results
            with open(self.output_dir / 'style_consistency.json', 'w') as f:
                json.dump(style_consistency, f, indent=2)
            
            # Create the analysis report
            self.create_analysis_report(dataset_info, style_consistency)
            
            # Return all required values
            return embeddings_2d, labels, style_consistency
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise
    
    def create_analysis_report(self, dataset_info, style_consistency):
        """Creates a human-readable analysis report."""
        report = ["# WikiArt Style Analysis Report\n"]
        
        # Dataset overview
        report.append("## Dataset Overview")
        report.append(f"- Total number of styles: {len(dataset_info['styles'])}")
        report.append(f"- Total number of images: {dataset_info['total_images']}")
        report.append("\n### Style Distribution")
        for style, data in dataset_info['styles'].items():
            report.append(f"- {style}: {data['image_count']} images")
        
        # Style consistency analysis
        report.append("\n## Style Consistency Analysis")
        for style, consistency in style_consistency.items():
            report.append(f"\n### {style}")
            for category, metrics in consistency.items():
                report.append(f"\n#### {category}")
                report.append(f"- Mean variance: {metrics['mean_variance']:.3f}")
        
        # Save report
        with open(self.output_dir / 'analysis_report.md', 'w') as f:
            f.write('\n'.join(report))

def main():
    parser = argparse.ArgumentParser(description='Analyze WikiArt dataset styles')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to WikiArt dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Path to save analysis results')
    parser.add_argument('--max_images', type=int, default=100,
                      help='Maximum images per style to analyze')
    
    args = parser.parse_args()
    
    try:
        runner = WikiArtRunner(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_images_per_style=args.max_images
        )
        
        embeddings, labels, consistency = runner.analyze_dataset()
        logger.info(f"Analysis complete. Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()