import torch
import torch.nn as nn
from libs.models import encoder4, decoder4
import numpy as np
from libs.Matrix import MulLayer
from libs.Loader import Dataset
import os
from typing import List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Set PYTORCH_CUDA_ALLOC_CONF to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class StyleTransferApplication:
    def __init__(self, vgg: nn.Module, dec: nn.Module, matrix: MulLayer, style_layers: List[str], device: torch.device):
        self.device = torch.device('cpu')  # Force CPU to avoid CUDA OOM issues

        # Move all models to CPU and set to evaluation mode
        self.vgg = vgg.to(self.device).eval()
        self.dec = dec.to(self.device).eval()
        self.matrix = matrix.to(self.device)

        self.style_layers = style_layers

    @torch.no_grad()
    def generate_stylized_image(self, contentV: torch.Tensor, styleV: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Generates the stylized image using the given transformation matrix on CPU."""
        # Force all operations to CPU
        contentV, styleV, matrix = contentV.to('cpu'), styleV.to('cpu'), matrix.to('cpu')

        sF, cF = self.vgg(styleV), self.vgg(contentV)

        transformed_features, _ = self.matrix(cF[self.style_layers[0]], sF[self.style_layers[0]])
        b, c, h, w = transformed_features.size()
        compressed_features = self.matrix.compress(transformed_features)

        transformed_feature = torch.bmm(matrix, compressed_features.view(b, self.matrix.matrixSize, -1))
        transformed_feature = transformed_feature.view(b, self.matrix.matrixSize, h, w)
        transformed_feature = self.matrix.unzip(transformed_feature)

        stylized_image = self.dec(transformed_feature).clamp(0, 1)
        return stylized_image


def process_and_save_stylized_images(style_dir: str, opt, style_transfer: StyleTransferApplication):
    style_path = os.path.join(opt.matrixPath, style_dir)
    matrix_files = [f for f in os.listdir(style_path) if f.endswith('.pth')]

    content_dataset = Dataset(opt.contentPath, opt.loadSize, opt.fineSize)
    style_dataset = Dataset(opt.stylePath, opt.loadSize, opt.fineSize)

    # Use only the first content and style image for this analysis
    contentV, _ = content_dataset[0]
    styleV, _ = style_dataset[0]
    contentV = contentV.unsqueeze(0).to('cpu')
    styleV = styleV.unsqueeze(0).to('cpu')

    for matrix_file in tqdm(matrix_files, desc=f"Processing {style_dir}"):
        matrix_path = os.path.join(style_path, matrix_file)
        saved_matrix = torch.load(matrix_path, map_location='cpu', weights_only=True)  # Load matrix on CPU

        # Generate stylized image
        stylized_image = style_transfer.generate_stylized_image(contentV, styleV, saved_matrix)

        # Plot and save stylized image
        plt.figure(figsize=(5, 5))
        image = stylized_image.squeeze().permute(1, 2, 0).numpy()
        plt.imshow(image)
        plt.title(f'Stylized Image - Style: {style_dir} - Matrix: {matrix_file}')
        plt.axis('off')
        output_file = f'stylized_image_{style_dir}_{matrix_file}.png'.replace('.pth', '')
        plt.savefig(output_file)
        plt.close()


def main():
    # Use CPU for all computations to avoid CUDA memory issues
    device = torch.device("cpu")

    # Load models
    vgg, dec, matrix = load_models(device)

    # Define options and parameters
    opt = Options()

    # Initialize the style transfer object
    style_transfer = StyleTransferApplication(vgg, dec, matrix, style_layers=['r41'], device=device)

    # Loop over style directories
    style_dirs = os.listdir(opt.matrixPath)

    for style_dir in style_dirs:
        # Process each style directory and save stylized images
        process_and_save_stylized_images(style_dir, opt, style_transfer)


class Options:
    def __init__(self):
        self.contentPath = "data/content/"
        self.stylePath = "data/style/"
        self.loadSize = 128  # Reduced load size to minimize memory usage
        self.fineSize = 128  # Reduced fine size to minimize memory usage
        self.matrixPath = "Matrices/"


def load_models(device: torch.device) -> Tuple[nn.Module, nn.Module, MulLayer]:
    vgg = encoder4()
    dec = decoder4()
    matrix = MulLayer('r41')
    # Load models on CPU
    vgg.load_state_dict(torch.load('models/vgg_r41.pth', map_location='cpu'))
    dec.load_state_dict(torch.load('models/dec_r41.pth', map_location='cpu'))
    return vgg, dec, matrix


if __name__ == "__main__":
    main()
