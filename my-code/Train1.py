import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from libs.Loader import Dataset
from libs.Matrix import MulLayer
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from libs.utils import print_options
from libs.Criterion import LossCriterion
from libs.models import encoder3, encoder4, decoder3, decoder4
from libs.models import encoder5 as loss_network
from transformers import ViTModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', type=str, required=True, help='content image directory')
    parser.add_argument('--style_dir', type=str, required=True, help='style image directory')
    parser.add_argument('--vgg_dir', type=str, required=True, help='directory to pretrained VGG model')
    parser.add_argument('--decoder_dir', type=str, required=True, help='directory to pretrained decoder model')
    parser.add_argument('--loss_network_dir', type=str, required=True, help='directory to pretrained loss network')
    parser.add_argument('--cuda', type=int, required=True, help='set it to 1 for running on GPU, 0 for CPU')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--max_iter', type=int, default=160000, help='total train iteration')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--log_dir', type=str, default='logs', help='directory to save logs')
    parser.add_argument('--model_dir', type=str, default='models', help='directory to save models')
    parser.add_argument('--matrixPath', type=str, default=None, help='path to pretrained linear transformation matrices')
    opt = parser.parse_args()
    
    print_options(opt)

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)
    if not os.path.exists(opt.model_dir):
        os.makedirs(opt.model_dir)

    cudnn.benchmark = True
    device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")

    # Define the transform for DINO input
    dino_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Define transform for VGG input
    vgg_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust size as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load models
    vgg = encoder3()
    dec = decoder3()
    vgg5 = loss_network()

    vgg.load_state_dict(torch.load(opt.vgg_dir, map_location=device, weights_only=True))
    dec.load_state_dict(torch.load(opt.decoder_dir, map_location=device, weights_only=True))
    vgg5.load_state_dict(torch.load(opt.loss_network_dir, map_location=device, weights_only=True))

    vgg.to(device).eval()
    dec.to(device)
    vgg5.to(device).eval()

    # Load DINO model
    dino_model = ViTModel.from_pretrained("facebook/dino-vits16").to(device)
    dino_model.eval()

    # Define the linear transformation layer
    matrix = MulLayer(512)  # Adjust size if needed
    if opt.matrixPath:
        matrix.load_state_dict(torch.load(opt.matrixPath))
    matrix.to(device)

    # Define optimizer
    optimizer = optim.Adam(matrix.parameters(), lr=opt.lr)

    # Define loss criterion
    criterion = LossCriterion(opt.batch_size, device)

    # Load datasets
    content_dataset = Dataset(opt.content_dir, opt.batch_size, vgg_transform)
    style_dataset = Dataset(opt.style_dir, opt.batch_size, dino_transform)
    content_loader = iter(content_dataset)
    style_loader = iter(style_dataset)

    for iteration in range(opt.max_iter):
        optimizer.zero_grad()

        try:
            content_images = next(content_loader)
        except StopIteration:
            content_loader = iter(content_dataset)
            content_images = next(content_loader)

        try:
            style_images = next(style_loader)
        except StopIteration:
            style_loader = iter(style_dataset)
            style_images = next(style_loader)

        content_images = content_images.to(device)
        style_images = style_images.to(device)

        # Extract features
        with torch.no_grad():
            content_features = vgg(content_images)
            style_features = dino_model(style_images).last_hidden_state

        # You may need to reshape or adapt style_features to match content_features
        style_features = style_features.view(style_features.size(0), -1, content_features.size(2), content_features.size(3))

        # Apply linear transformation
        transformed_features = matrix(content_features, style_features)

        # Generate output image
        output_images = dec(transformed_features)

        # Compute loss
        loss_c, loss_s, loss_identity = criterion(vgg, vgg5, content_images, style_images, output_images)
        
        # You might want to add an additional loss term using DINO features for style
        with torch.no_grad():
            dino_output = dino_model(output_images).last_hidden_state
        
        # Define and compute an additional style loss using DINO features
        dino_style_loss = nn.MSELoss()(dino_output, style_features)
        
        # Combine losses
        loss = loss_c + loss_s + loss_identity + dino_style_loss

        loss.backward()
        optimizer.step()

        if (iteration + 1) % 100 == 0:
            print(f"Iteration: {iteration+1}, Loss: {loss.item()}")

        if (iteration + 1) % 5000 == 0:
            # Save model
            torch.save(matrix.state_dict(), os.path.join(opt.model_dir, f'matrix_iter_{iteration+1}.pth'))
            # Save sample images
            vutils.save_image(output_images, 
                              os.path.join(opt.log_dir, f'output_iter_{iteration+1}.jpg'),
                              normalize=True)

if __name__ == '__main__':
    main()