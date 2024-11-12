import torch
import torch.nn as nn
from libs.models import encoder3,encoder4
from libs.models import decoder3,decoder4
import numpy as np
from libs.Matrix import MulLayer
from libs.Criterion import LossCriterion
from sklearn.decomposition import PCA


class LossCriterion(nn.Module):
    def __init__(self, style_layers, content_layers, style_weight, content_weight):
        super(LossCriterion, self).__init__()
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.styleLosses = [styleLoss()] * len(style_layers)
        self.contentLosses = [nn.MSELoss()] * len(content_layers)

    def forward(self, tF, sF, cF):
        # Content loss
        totalContentLoss = 0
        for i, layer in enumerate(self.content_layers):
            cf_i = cF[layer].detach()
            tf_i = tF[layer]
            loss_i = self.contentLosses[i]
            totalContentLoss += loss_i(tf_i, cf_i)
        totalContentLoss = totalContentLoss * self.content_weight

        # Style loss
        
        totalStyleLoss = 0
        for i, layer in enumerate(self.style_layers):
            sf_i = sF[layer].detach()
            tf_i = tF[layer]
            loss_i = self.styleLosses[i]
            totalStyleLoss += loss_i(tf_i, sf_i)
        totalStyleLoss = totalStyleLoss * self.style_weight

        loss = totalStyleLoss + totalContentLoss
        return loss, totalStyleLoss, totalContentLoss


class styleLoss(nn.Module):
    def forward(self,input,target):
        ib,ic,ih,iw = input.size()
        iF = input.view(ib,ic,-1)
        iMean = torch.mean(iF,dim=2)
        iCov = GramMatrix()(input)

        tb,tc,th,tw = target.size()
        tF = target.view(tb,tc,-1)
        tMean = torch.mean(tF,dim=2)
        tCov = GramMatrix()(target)

        loss = nn.MSELoss(size_average=False)(iMean,tMean) + nn.MSELoss(size_average=False)(iCov,tCov)
        return loss/tb

class GramMatrix(nn.Module):
    def forward(self,input):
        b, c, h, w = input.size()
        f = input.view(b,c,h*w) # bxcx(hxw)
        # torch.bmm(batch1, batch2, out=None)   #
        # batch1: bxmxp, batch2: bxpxn -> bxmxn #
        G = torch.bmm(f,f.transpose(1,2)) # f: bxcx(hxw), f.transpose: bx(hxw)xc -> bxcxc
        return G.div_(c*h*w)
    
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from libs.Loader import Dataset
from libs.models import encoder4, decoder4
from libs.Criterion import LossCriterion
from libs.Matrix import MulLayer
import os
from typing import List, Tuple
from tqdm import tqdm
from sklearn.cluster import KMeans
 
class LossSensitivity:
    def __init__(self, vgg: nn.Module, dec: nn.Module, matrix: MulLayer,
                 style_layers: List[str], content_layers: List[str],
                 style_weight: float, content_weight: float, device: torch.device):
        self.vgg = vgg.to(device)
        self.dec = dec.to(device)
        self.matrix = matrix.to(device)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.criterion = LossCriterion(style_layers, content_layers, style_weight, content_weight)
        self.device = device
 
    def add_noise(self, matrix: torch.Tensor, sigma: float) -> torch.Tensor:
        return matrix + torch.randn_like(matrix) * sigma
 
    @torch.no_grad()
    def forward(self, contentV: torch.Tensor, styleV: torch.Tensor) -> Tuple[dict, dict]:
        return self.vgg(styleV), self.vgg(contentV)
 
    def compute_loss(self, contentV: torch.Tensor, styleV: torch.Tensor, noisy_matrix: torch.Tensor) -> float:
        sF, cF = self.forward(contentV, styleV)
        
        transformed_features, _ = self.matrix(cF[self.style_layers[0]], sF[self.style_layers[0]])
        b, c, h, w = transformed_features.size()
        compressed_features = self.matrix.compress(transformed_features)
        
        if noisy_matrix.size(1) != compressed_features.view(b, self.matrix.matrixSize, -1).size(1):
            print(f"Dimension mismatch: {noisy_matrix.size()} vs {compressed_features.size()}")
            return float('inf')
        
        noisy_transfeature = torch.bmm(noisy_matrix, compressed_features.view(b, self.matrix.matrixSize, -1))
        noisy_transfeature = noisy_transfeature.view(b, self.matrix.matrixSize, h, w)
        noisy_transfeature = self.matrix.unzip(noisy_transfeature)
        
        noisy_transfer = self.dec(noisy_transfeature).clamp(0, 1)
        tF = self.vgg(noisy_transfer)
        
        total_loss, _, _ = self.criterion(tF, sF, cF)
        return total_loss.item()
 
    def compute_matrix_metrics(self, original_matrix: torch.Tensor, noisy_matrix: torch.Tensor) -> Tuple[float, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        # Frobenius Norm Difference
        frobenius_diff = torch.norm(original_matrix - noisy_matrix, p='fro').item()
        
        # Eigenvalue/Eigenvector Changes
        orig_eigenvalues, orig_eigenvectors = torch.linalg.eig(original_matrix)
        noisy_eigenvalues, noisy_eigenvectors = torch.linalg.eig(noisy_matrix)
        
        return frobenius_diff, (orig_eigenvalues, noisy_eigenvalues), (orig_eigenvectors, noisy_eigenvectors)
 
    def run_experiment(self, contentV: torch.Tensor, styleV: torch.Tensor,
                       sigmas: np.ndarray, matrix: torch.Tensor) -> Tuple[List[float], List[float], List[float], List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        sigma_values = []
        loss_values = []
        frobenius_diffs = []
        eigenvalue_changes = []
        eigenvector_changes = []
 
        for sigma in sigmas:
            noisy_matrix = self.add_noise(matrix, sigma)
            loss = self.compute_loss(contentV, styleV, noisy_matrix)
            if loss == float('inf'):
                print(f"Skipping sigma {sigma} due to dimension mismatch.")
                continue
            
            frobenius_diff, eigenvalues, eigenvectors = self.compute_matrix_metrics(matrix, noisy_matrix)
            
            sigma_values.append(sigma)
            loss_values.append(loss)
            frobenius_diffs.append(frobenius_diff)
            eigenvalue_changes.append(eigenvalues)
            eigenvector_changes.append(eigenvectors)
 
        return sigma_values, loss_values, frobenius_diffs, eigenvalue_changes, eigenvector_changes
 

def process_style_dir(style_dir: str, opt, loss_sensitivity: LossSensitivity,
                      sigmas: np.ndarray, device: torch.device) -> Tuple[List[List[float]], List[List[float]], List[List[Tuple[torch.Tensor, torch.Tensor]]], List[List[Tuple[torch.Tensor, torch.Tensor]]]]:
    style_path = os.path.join(opt.matrixPath, style_dir)
    matrix_files = [f for f in os.listdir(style_path) if f.endswith('.pth')]
    all_loss_values = []
    all_frobenius_diffs = []
    all_eigenvalue_changes = []
    all_eigenvector_changes = []

    content_dataset = Dataset(opt.contentPath, opt.loadSize, opt.fineSize)
    style_dataset = Dataset(opt.stylePath, opt.loadSize, opt.fineSize)
    contentV, _ = content_dataset[0]
    styleV, _ = style_dataset[0]
    contentV = contentV.unsqueeze(0).to(device)
    styleV = styleV.unsqueeze(0).to(device)

    for matrix_file in tqdm(matrix_files, desc=f"Processing {style_dir}"):
        matrix_path = os.path.join(style_path, matrix_file)
        saved_matrix = torch.load(matrix_path, map_location=device)
        _, loss_values, frobenius_diffs, eigenvalue_changes, eigenvector_changes = loss_sensitivity.run_experiment(contentV, styleV, sigmas, saved_matrix)
        all_loss_values.append(loss_values)
        all_frobenius_diffs.append(frobenius_diffs)
        all_eigenvalue_changes.append(eigenvalue_changes)
        all_eigenvector_changes.append(eigenvector_changes)

    return all_loss_values, all_frobenius_diffs, all_eigenvalue_changes, all_eigenvector_changes

def plot_style_results(style_dir: str, sigmas: np.ndarray, avg_loss_values: np.ndarray, avg_frobenius_diffs: np.ndarray):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    ax1.plot(sigmas[:len(avg_loss_values)], avg_loss_values, '-o')
    ax1.set_xlabel('Sigma (Noise Level)')
    ax1.set_ylabel('Average Total Loss')
    ax1.set_title(f'Average Noise Sensitivity for Style: {style_dir}')
    ax1.grid(True)
    
    ax2.plot(sigmas[:len(avg_frobenius_diffs)], avg_frobenius_diffs, '-o')
    ax2.set_xlabel('Sigma (Noise Level)')
    ax2.set_ylabel('Average Frobenius Norm Difference')
    ax2.set_title(f'Average Matrix Change for Style: {style_dir}')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'noise_sensitivity_{style_dir}.png')
    plt.close()

def analyze_eigenvalue_changes(style_dir: str, sigmas: np.ndarray, eigenvalue_changes: List[Tuple[torch.Tensor, torch.Tensor]]):
    avg_eigenvalue_diffs = []
    for orig, noisy in eigenvalue_changes:
        avg_diff = torch.mean(torch.abs(orig - noisy)).item()
        avg_eigenvalue_diffs.append(avg_diff)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sigmas, avg_eigenvalue_diffs, '-o')
    plt.xlabel('Sigma (Noise Level)')
    plt.ylabel('Average Eigenvalue Difference')
    plt.title(f'Eigenvalue Sensitivity for Style: {style_dir}')
    plt.grid(True)
    plt.savefig(f'eigenvalue_sensitivity_{style_dir}.png')
    plt.close()

def plot_average_frobenius_norm_trend(sigmas: np.ndarray, all_styles_frobenius_diffs: List[np.ndarray]):
    avg_frobenius_diffs = np.mean(all_styles_frobenius_diffs, axis=0)
    std_frobenius_diffs = np.std(all_styles_frobenius_diffs, axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(sigmas, avg_frobenius_diffs, '-', color='blue', label='Average')
    plt.fill_between(sigmas,
                     avg_frobenius_diffs - std_frobenius_diffs,
                     avg_frobenius_diffs + std_frobenius_diffs,
                     alpha=0.3, color='lightblue', label='±1 Standard Deviation')

    plt.plot(sigmas, avg_frobenius_diffs - std_frobenius_diffs, '--', color='red', alpha=0.5, label='-1 Std Dev')
    plt.plot(sigmas, avg_frobenius_diffs + std_frobenius_diffs, '--', color='red', alpha=0.5, label='+1 Std Dev')
    
    plt.xlabel('Sigma (Noise Level)')
    plt.ylabel('Average Frobenius Norm Difference')
    plt.title('Average Frobenius Norm Trend Across All Styles')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('average_frobenius_norm_trend.png', dpi=300)
    plt.close()

    print(f"Maximum average Frobenius norm difference: {np.max(avg_frobenius_diffs):.4f}")
    print(f"Maximum standard deviation: {np.max(std_frobenius_diffs):.4f}")
    print(f"Sigma at maximum average difference: {sigmas[np.argmax(avg_frobenius_diffs)]:.4f}")

def identify_sensitive_components(style_dir: str, sigmas: np.ndarray, eigenvalue_changes: List[Tuple[torch.Tensor, torch.Tensor]]):
    num_eigenvalues = eigenvalue_changes[0][0].shape[0]
    eigenvalue_sensitivities = [[] for _ in range(num_eigenvalues)]
    
    for orig, noisy in eigenvalue_changes:
        diffs = torch.abs(orig - noisy)
        for i in range(num_eigenvalues):
            eigenvalue_sensitivities[i].append(diffs[i].item())
    
    plt.figure(figsize=(12, 8))
    for i in range(num_eigenvalues):
        plt.plot(sigmas, eigenvalue_sensitivities[i], label=f'Eigenvalue {i+1}')
    plt.xlabel('Sigma (Noise Level)')
    plt.ylabel('Eigenvalue Difference')
    plt.title(f'Individual Eigenvalue Sensitivities for Style: {style_dir}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'individual_eigenvalue_sensitivities_{style_dir}.png')
    plt.close()

class Options:
    def __init__(self):
        self.contentPath = "data/content/"
        self.stylePath = "data/style/"
        self.loadSize = 256
        self.fineSize = 256
        self.matrixPath = "Matrices/"

def load_models(device: torch.device) -> Tuple[nn.Module, nn.Module, MulLayer]:
    vgg = encoder4()
    dec = decoder4()
    matrix = MulLayer('r41')
    vgg.load_state_dict(torch.load('models/vgg_r41.pth', map_location=device))
    dec.load_state_dict(torch.load('models/dec_r41.pth', map_location=device))
    return vgg, dec, matrix

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg, dec, matrix = load_models(device)
    
    style_layers = ['r41']
    content_layers = ['r41']
    style_weight = 1e5
    content_weight = 1.0

    opt = Options()
    loss_sensitivity = LossSensitivity(vgg, dec, matrix, style_layers, content_layers,
                                       style_weight, content_weight, device)

    sigmas = np.linspace(0, 0.2, 100)
    style_dirs = [d for d in os.listdir(opt.matrixPath) if os.path.isdir(os.path.join(opt.matrixPath, d))]
    
    all_styles_frobenius_diffs = []

    for style_dir in style_dirs:
        try:
            all_loss_values, all_frobenius_diffs, all_eigenvalue_changes, all_eigenvector_changes = process_style_dir(style_dir, opt, loss_sensitivity, sigmas, device)
            
            if all_loss_values:
                avg_loss_values = np.mean(all_loss_values, axis=0)
                avg_frobenius_diffs = np.mean(all_frobenius_diffs, axis=0)
                all_styles_frobenius_diffs.append(avg_frobenius_diffs)
                plot_style_results(style_dir, sigmas, avg_loss_values, avg_frobenius_diffs)
                analyze_eigenvalue_changes(style_dir, sigmas, all_eigenvalue_changes[0])  # Analyze first set of eigenvalue changes
                identify_sensitive_components(style_dir, sigmas, all_eigenvalue_changes[0])
            else:
                print(f"Warning: No data available for style directory {style_dir}")
        except Exception as e:
            print(f"Error processing style directory {style_dir}: {str(e)}")

    if all_styles_frobenius_diffs:
        plot_average_frobenius_norm_trend(sigmas, all_styles_frobenius_diffs)
    else:
        print("Warning: No Frobenius norm data available for analysis across styles.")

if __name__ == "__main__":
    main()