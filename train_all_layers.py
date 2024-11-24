import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import traceback
import time
from datetime import datetime
from libs.Loader import Dataset
from libs.Matrix import MulLayer
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from libs.utils import print_options
from libs.Criterion import LossCriterion
from libs.models import encoder1, encoder2
from libs.models import decoder1, decoder2
from libs.models import encoder5 as loss_network
from multiprocessing import freeze_support

def train():
    ################# ARGUMENT PARSING #################
    parser = argparse.ArgumentParser()
    # Paths for pre-trained encoders and datasets
    parser.add_argument("--vgg1_dir", default='trained_models/encoder_r11.pth', help='pre-trained encoder1 path')
    parser.add_argument("--vgg2_dir", default='trained_models/encoder_r21.pth', help='pre-trained encoder2 path')
    parser.add_argument("--loss_network_dir", default='models/vgg_r51.pth', help='used for loss network')
    parser.add_argument("--decoder1_dir", default='trained_models/decoder_r11.pth', help='pre-trained decoder1 path')
    parser.add_argument("--decoder2_dir", default='trained_models/decoder_r21.pth', help='pre-trained decoder2 path')
    parser.add_argument("--stylePath", default="datasets/wikiArt/", help='path to wikiArt dataset')
    parser.add_argument("--contentPath", default="datasets/coco2014/images/train2014/", help='path to MSCOCO dataset')
    parser.add_argument("--outf", default="trainingOutput1/", help='folder to output images and model checkpoints')
    parser.add_argument("--content_layers", default="r11,r21", help='layers for content')
    parser.add_argument("--style_layers", default="r11,r21", help='layers for style')
    parser.add_argument("--batchSize", type=int, default=8, help='batch size')
    parser.add_argument("--niter", type=int, default=100000, help='iterations to train the model')
    parser.add_argument('--loadSize', type=int, default=300, help='scale image size')
    parser.add_argument('--fineSize', type=int, default=256, help='crop image size')
    parser.add_argument("--lr", type=float, default=1e-4, help='learning rate')
    parser.add_argument("--content_weight", type=float, default=1.0, help='content loss weight')
    parser.add_argument("--style_weight", type=float, default=0.02, help='style loss weight')
    parser.add_argument("--log_interval", type=int, default=100, help='log interval')
    parser.add_argument("--gpu_id", type=int, default=0, help='which gpu to use')
    parser.add_argument("--save_interval", type=int, default=5000, help='checkpoint save interval')

    ################# PREPARATIONS #################
    opt = parser.parse_args()
    opt.content_layers = opt.content_layers.split(',')
    opt.style_layers = opt.style_layers.split(',')
    
    # GPU Setup
    opt.cuda = torch.cuda.is_available()
    if opt.cuda:
        print(f"Using GPU device {opt.gpu_id}")
        torch.cuda.set_device(opt.gpu_id)
    else:
        print("CUDA is not available. Using CPU...")

    # Check directories
    for path in [opt.contentPath, opt.stylePath]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset path not found: {path}")

    # Create output directory
    os.makedirs(opt.outf, exist_ok=True)
    cudnn.benchmark = True
    print_options(opt)

    ################# DATA #################
    content_dataset = Dataset(opt.contentPath, opt.loadSize, opt.fineSize)
    content_loader_ = torch.utils.data.DataLoader(dataset=content_dataset,
                                                batch_size=opt.batchSize,
                                                shuffle=True,
                                                num_workers=1,
                                                drop_last=True)
    content_loader = iter(content_loader_)
    
    style_dataset = Dataset(opt.stylePath, opt.loadSize, opt.fineSize)
    style_loader_ = torch.utils.data.DataLoader(dataset=style_dataset,
                                              batch_size=opt.batchSize,
                                              shuffle=True,
                                              num_workers=1,
                                              drop_last=True)
    style_loader = iter(style_loader_)

    ################# MODEL #################
    # Initialize models
    vgg5 = loss_network()
    vgg1 = encoder1()
    vgg2 = encoder2()
    dec1 = decoder1()
    dec2 = decoder2()

    encoders = {'r11': vgg1, 'r21': vgg2}
    decoders = {'r11': dec1, 'r21': dec2}
    matrix = {'r11': MulLayer('r11'), 'r21': MulLayer('r21')}

    # Load pretrained models with safety check
    try:
        vgg1.load_state_dict(torch.load(opt.vgg1_dir, weights_only=True))
        vgg2.load_state_dict(torch.load(opt.vgg2_dir, weights_only=True))
        dec1.load_state_dict(torch.load(opt.decoder1_dir, weights_only=True))
        dec2.load_state_dict(torch.load(opt.decoder2_dir, weights_only=True))
        vgg5.load_state_dict(torch.load(opt.loss_network_dir, weights_only=True))
    except Exception as e:
        print(f"Error loading pretrained models: {str(e)}")
        raise

    # Move models to GPU if available
    if opt.cuda:
        vgg5.cuda()
        for model_dict in [encoders, decoders, matrix]:
            for model in model_dict.values():
                model.cuda()

    # Initialize criterion and optimizers
    criterion = LossCriterion(
        style_layers=opt.style_layers,
        content_layers=opt.content_layers,
        style_weight=opt.style_weight,
        content_weight=opt.content_weight
    )
    
    if opt.cuda:
        criterion.cuda()

    optimizers = {
        layer: optim.Adam(matrix[layer].parameters(), lr=opt.lr)
        for layer in matrix.keys()
    }

    # Initialize tensors
    contentV = torch.FloatTensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
    styleV = torch.FloatTensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)

    if opt.cuda:
        contentV = contentV.cuda()
        styleV = styleV.cuda()

    ################# TRAINING LOOP #################
    print("Starting training...")
    running_loss = 0.0
    running_style_loss = 0.0
    running_content_loss = 0.0
    best_loss = float('inf')
    start_time = time.time()

    try:
        for iteration in range(1, opt.niter + 1):
            # Zero gradients
            for optimizer in optimizers.values():
                optimizer.zero_grad()

            # Load images
            try:
                content, _ = next(content_loader)
            except StopIteration:
                content_loader = iter(content_loader_)
                content, _ = next(content_loader)

            try:
                style, _ = next(style_loader)
            except StopIteration:
                style_loader = iter(style_loader_)
                style, _ = next(style_loader)

            # Transfer to GPU if available
            if opt.cuda:
                content = content.cuda()
                style = style.cuda()

            contentV.resize_(content.size()).copy_(content)
            styleV.resize_(style.size()).copy_(style)

            # Process each layer
            total_loss = 0
            total_style_loss = 0
            total_content_loss = 0

            for layer in matrix:
                # Get models for current layer
                encoder = encoders[layer]
                decoder = decoders[layer]

                # Extract features
                sF = {layer: encoder(styleV)}
                cF = {layer: encoder(contentV)}

                # Transform and decode
                feature, transmatrix = matrix[layer](cF[layer], sF[layer])
                transfer = decoder(feature)

                # Compute losses
                sF_loss = vgg5(styleV)
                cF_loss = vgg5(contentV)
                tF = vgg5(transfer)
                
                loss, style_loss, content_loss = criterion(tF, sF_loss, cF_loss)

                # Backward and optimize
                loss.backward()
                optimizers[layer].step()

                # Accumulate losses
                total_loss += loss.item()
                total_style_loss += style_loss.item()
                total_content_loss += content_loss.item()

            # Update running losses
            running_loss += total_loss
            running_style_loss += total_style_loss
            running_content_loss += total_content_loss

            # Log progress
            if iteration % opt.log_interval == 0:
                elapsed_time = time.time() - start_time
                avg_loss = running_loss / opt.log_interval
                avg_style_loss = running_style_loss / opt.log_interval
                avg_content_loss = running_content_loss / opt.log_interval
                
                print(f'\nIteration: [{iteration}/{opt.niter}] '
                      f'Time: {elapsed_time:.2f}s\n'
                      f'Total Loss: {total_loss:.4f} '
                      f'(Style: {total_style_loss:.4f}, '
                      f'Content: {total_content_loss:.4f})\n'
                      f'Average Loss: {avg_loss:.4f} '
                      f'(Style: {avg_style_loss:.4f}, '
                      f'Content: {avg_content_loss:.4f})')

                # Reset running losses
                running_loss = 0.0
                running_style_loss = 0.0
                running_content_loss = 0.0

                # Save best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    for layer in matrix:
                        torch.save(matrix[layer].state_dict(),
                                 f'{opt.outf}/best_{layer}.pth')

            # Save checkpoints
            if iteration % opt.save_interval == 0:
                # Save sample images
                transfer = transfer.clamp(0, 1)
                concat = torch.cat([contentV, styleV, transfer], dim=0)
                vutils.save_image(concat,
                                f'{opt.outf}/sample_{iteration}.png',
                                normalize=True,
                                scale_each=True,
                                nrow=opt.batchSize)

                # Save model checkpoints
                for layer in matrix:
                    torch.save({
                        'iteration': iteration,
                        'state_dict': matrix[layer].state_dict(),
                        'optimizer': optimizers[layer].state_dict(),
                        'loss': avg_loss,
                    }, f'{opt.outf}/checkpoint_{layer}_{iteration}.pth')

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")
        for layer in matrix:
            torch.save({
                'iteration': iteration,
                'state_dict': matrix[layer].state_dict(),
                'optimizer': optimizers[layer].state_dict(),
                'loss': total_loss,
            }, f'{opt.outf}/interrupted_{layer}.pth')
        print("State saved successfully.")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        traceback.print_exc()

if __name__ == '__main__':
    try:
        freeze_support()
        train()
    except Exception as e:
        print(f"Error during training: {str(e)}")
        traceback.print_exc()