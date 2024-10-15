import os
import torch
import argparse
from libs.Loader import Dataset
from libs.Matrix import MulLayer
from libs.models import encoder4
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vgg_dir", default='models/vgg_r41.pth', help='pre-trained encoder path')
    parser.add_argument("--matrixPath", default='models/r41.pth', help='pre-trained model path')
    parser.add_argument("--stylePath", default="data/style/", help='path to style images')
    parser.add_argument("--contentPath", default="data/content/", help='path to content images or single image')
    parser.add_argument("--matrixOutf", default="Matrices/", help='path to save transformation matrices')
    parser.add_argument("--loadSize", type=int, default=256, help='scale image size')
    parser.add_argument("--layer", default="r41", help='which features to transfer')
    return parser.parse_args()

def setup_model(opt):
    vgg = encoder4()
    matrix = MulLayer(opt.layer)
    vgg.load_state_dict(torch.load(opt.vgg_dir, map_location='cpu'))
    matrix.load_state_dict(torch.load(opt.matrixPath, map_location='cpu'))
    
    if torch.cuda.is_available():
        vgg.cuda()
        matrix.cuda()
    
    return vgg, matrix

def load_image(path, size):
    if os.path.isfile(path):
        try:
            # If path is a file, load it directly
            img = Image.open(path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return transform(img).unsqueeze(0)
        except UnidentifiedImageError:
            print(f"Skipping file {path}: not a valid image.")
            return None
    else:
        # If path is a directory, use the Dataset class
        dataset = Dataset(path, size, size)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        return next(iter(loader))[0]

def main():
    opt = parse_args()
    os.makedirs(opt.matrixOutf, exist_ok=True)

    vgg, matrix = setup_model(opt)

    if os.path.isfile(opt.contentPath):
        content_files = [os.path.basename(opt.contentPath)]
    else:
        content_files = [f for f in os.listdir(opt.contentPath) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    style_files = [f for f in os.listdir(opt.stylePath) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    for style_file in style_files:
        style_path = os.path.join(opt.stylePath, style_file)
        style = load_image(style_path, opt.loadSize)

        if style is None:
            continue

        if torch.cuda.is_available():
            style = style.cuda()

        for content_file in content_files:
            content_path = opt.contentPath if os.path.isfile(opt.contentPath) else os.path.join(opt.contentPath, content_file)
            content = load_image(content_path, opt.loadSize)

            if content is None:
                continue

            if torch.cuda.is_available():
                content = content.cuda()

            with torch.no_grad():
                cF = vgg(content)   
                sF = vgg(style)
                
                _, transmatrix = matrix(cF[opt.layer], sF[opt.layer])

            matrix_filename = f'matrix_{os.path.splitext(style_file)[0]}_{os.path.splitext(content_file)[0]}.pth'
            style_out_dir = os.path.join(opt.matrixOutf, os.path.splitext(style_file)[0])
            os.makedirs(style_out_dir, exist_ok=True)
            torch.save(transmatrix, os.path.join(style_out_dir, matrix_filename))
            print(f'Saved matrix for style {style_file} and content {content_file}: {matrix_filename}')

if __name__ == "__main__":
    main()