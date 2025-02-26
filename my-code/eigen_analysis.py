import torch
import os
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

class MatrixAnalyzer:
    def __init__(self, matrix_path: str):
        self.matrix_path = matrix_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def compute_matrix_properties(self, matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute eigenvalues, eigenvectors, and determinant of a matrix."""
        # Ensure matrix is on CPU for numerical stability
        matrix = matrix.cpu()
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eig(matrix)
        
        # Compute determinant
        determinant = torch.linalg.det(matrix)
        
        # Compute condition number
        condition_number = torch.linalg.cond(matrix)
        
        # Compute rank
        rank = torch.linalg.matrix_rank(matrix)
        
        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'determinant': determinant,
            'condition_number': condition_number,
            'rank': rank
        }
    
    def analyze_style_directory(self, style_dir: str, layer: str) -> Dict[str, List]:
        """Analyze all matrices for a specific style and layer."""
        style_path = os.path.join(self.matrix_path, style_dir)
        matrix_files = [f for f in os.listdir(style_path) if f.endswith('.pth') and layer in f]
        
        results = defaultdict(list)
        
        for matrix_file in tqdm(matrix_files, desc=f"Processing {style_dir} - {layer}"):
            matrix_path = os.path.join(style_path, matrix_file)
            matrix = torch.load(matrix_path, map_location=self.device)
            
            properties = self.compute_matrix_properties(matrix)
            
            for key, value in properties.items():
                results[key].append(value)
        
        return results
    
    def plot_eigenvalue_distribution(self, eigenvalues: List[torch.Tensor], style_dir: str, layer: str):
        """Plot the distribution of eigenvalues."""
        # Convert complex eigenvalues to magnitudes
        magnitudes = [torch.abs(ev) for ev in eigenvalues]
        all_magnitudes = torch.cat(magnitudes).numpy()
        
        plt.figure(figsize=(10, 6))
        plt.hist(all_magnitudes, bins=50, alpha=0.7)
        plt.xlabel('Eigenvalue Magnitude')
        plt.ylabel('Frequency')
        plt.title(f'Eigenvalue Distribution for {style_dir} - {layer}')
        plt.grid(True)
        plt.savefig(f'eigenvalue_dist_{style_dir}_{layer}.png')
        plt.close()
        
    def plot_determinant_distribution(self, determinants: List[torch.Tensor], style_dir: str, layer: str):
        """Plot the distribution of determinants."""
        det_values = [d.item() for d in determinants]
        
        plt.figure(figsize=(10, 6))
        plt.hist(det_values, bins=50, alpha=0.7)
        plt.xlabel('Determinant Value')
        plt.ylabel('Frequency')
        plt.title(f'Determinant Distribution for {style_dir} - {layer}')
        plt.grid(True)
        plt.savefig(f'determinant_dist_{style_dir}_{layer}.png')
        plt.close()
    
    def analyze_eigenvector_alignment(self, eigenvectors: List[torch.Tensor]) -> np.ndarray:
        """Analyze the alignment between eigenvectors of different matrices."""
        n_matrices = len(eigenvectors)
        alignment_matrix = np.zeros((n_matrices, n_matrices))
        
        for i in range(n_matrices):
            for j in range(i+1, n_matrices):
                # Compute alignment as the average absolute dot product
                alignment = torch.abs(torch.mm(
                    eigenvectors[i].T,
                    eigenvectors[j]
                )).mean().item()
                alignment_matrix[i, j] = alignment
                alignment_matrix[j, i] = alignment
        
        return alignment_matrix
    
    def plot_eigenvector_alignment(self, alignment_matrix: np.ndarray, style_dir: str, layer: str):
        """Plot the eigenvector alignment matrix as a heatmap."""
        plt.figure(figsize=(10, 8))
        plt.imshow(alignment_matrix, cmap='viridis')
        plt.colorbar(label='Average Alignment')
        plt.title(f'Eigenvector Alignment for {style_dir} - {layer}')
        plt.savefig(f'eigenvector_alignment_{style_dir}_{layer}.png')
        plt.close()

def main():
    matrix_path = "models/"
    layers = ['r11', 'r21', 'r31', 'r41']  # Add or modify layers as needed
    
    analyzer = MatrixAnalyzer(matrix_path)
    style_dirs = [d for d in os.listdir(matrix_path) if os.path.isdir(os.path.join(matrix_path, d))]
    
    # Create results directory
    os.makedirs('analysis_results', exist_ok=True)
    
    # Store summary statistics
    summary_stats = defaultdict(lambda: defaultdict(dict))
    
    for style_dir in style_dirs:
        print(f"\nAnalyzing style: {style_dir}")
        
        for layer in layers:
            try:
                results = analyzer.analyze_style_directory(style_dir, layer)
                
                # Plot distributions
                analyzer.plot_eigenvalue_distribution(results['eigenvalues'], style_dir, layer)
                analyzer.plot_determinant_distribution(results['determinants'], style_dir, layer)
                
                # Analyze eigenvector alignment
                alignment_matrix = analyzer.analyze_eigenvector_alignment(results['eigenvectors'])
                analyzer.plot_eigenvector_alignment(alignment_matrix, style_dir, layer)
                
                # Compute summary statistics
                summary_stats[style_dir][layer] = {
                    'mean_eigenvalue_magnitude': torch.mean(torch.abs(torch.cat(results['eigenvalues']))).item(),
                    'std_eigenvalue_magnitude': torch.std(torch.abs(torch.cat(results['eigenvalues']))).item(),
                    'mean_determinant': torch.mean(torch.stack(results['determinants'])).item(),
                    'mean_condition_number': torch.mean(torch.stack(results['condition_number'])).item(),
                    'mean_rank': torch.mean(torch.stack(results['rank'])).item()
                }
                
            except Exception as e:
                print(f"Error processing {style_dir} - {layer}: {str(e)}")
    
    # Save summary statistics
    import json
    with open('analysis_results/summary_stats.json', 'w') as f:
        json.dump(summary_stats, f, indent=4)
    
    print("\nAnalysis complete! Results have been saved to the analysis_results directory.")

if __name__ == "__main__":
    main()