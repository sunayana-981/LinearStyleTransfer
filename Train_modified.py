import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from libs.Loader import Dataset
from libs.Matrix1 import MulLayer
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from libs.utils import print_options
from libs.Criterion import LossCriterion
from libs.models import encoder3, encoder4
from libs.models import decoder3, decoder4
from libs.models import encoder5 as loss_network
from libs.models import decoder5 as decoder5
from multiprocessing import freeze_support

def adjust_learning_rate(optimizer, iteration):
    """Adjusts learning rate based on iteration"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] / (1 + iteration * 1e-5)

def main():
    freeze_support()

    ################# CONSTANTS #################
    # Define layers based on the MulLayer configuration
    LAYER_CONFIG = {
        'r11': 64,
        'r21': 128
    }
    WORKING_LAYERS = list(LAYER_CONFIG.keys())  # Only r11 and r21

    ################# ARGUMENTS #################
    parser = argparse.ArgumentParser()
    parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                        help='pre-trained encoder path')
    parser.add_argument("--loss_network_dir", default='models/vgg_r51.pth',
                        help='used for loss network')
    parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                        help='pre-trained decoder path')
    parser.add_argument("--stylePath", default="./datasets/wikiArt/",
                        help='path to wikiArt dataset')
    parser.add_argument("--contentPath", default="./datasets/coco2014/images/train2014",
                        help='path to MSCOCO dataset')
    parser.add_argument("--outf", default="trainingOutput/",
                        help='folder to output images and model checkpoints')
    parser.add_argument("--content_layers", default="r21",
                        help='layers for content')
    parser.add_argument("--style_layers", default="r11,r21",
                        help='layers for style')
    parser.add_argument("--batchSize", type=int, default=8,
                        help='batch size')
    parser.add_argument("--niter", type=int, default=100000,
                        help='iterations to train the model')
    parser.add_argument('--loadSize', type=int, default=300,
                        help='scale image size')
    parser.add_argument('--fineSize', type=int, default=256,
                        help='crop image size')
    parser.add_argument("--lr", type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument("--content_weight", type=float, default=1.0,
                        help='content loss weight')
    parser.add_argument("--style_weight", type=float, default=0.02,
                        help='style loss weight')
    parser.add_argument("--log_interval", type=int, default=500,
                        help='log interval')
    parser.add_argument("--gpu_id", type=int, default=0,
                        help='which gpu to use')
    parser.add_argument("--save_interval", type=int, default=5000,
                        help='checkpoint save interval')

    ################# PREPARATIONS #################
    opt = parser.parse_args()
    opt.content_layers = opt.content_layers.split(',')
    opt.style_layers = opt.style_layers.split(',')
    opt.cuda = torch.cuda.is_available()

    if opt.cuda:
        torch.cuda.set_device(opt.gpu_id)
        print(f"Using GPU: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available, using CPU.")

    os.makedirs(opt.outf, exist_ok=True)
    cudnn.benchmark = True
    print_options(opt)

    ################# DATA #################
    try:
        content_dataset = Dataset(opt.contentPath, opt.loadSize, opt.fineSize)
        content_loader_ = torch.utils.data.DataLoader(dataset=content_dataset,
                                                    batch_size=opt.batchSize,
                                                    shuffle=True,
                                                    num_workers=1,
                                                    drop_last=True)
        content_loader = iter(content_loader_)
        print("Content dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading content dataset: {e}")
        return

    try:
        style_dataset = Dataset(opt.stylePath, opt.loadSize, opt.fineSize)
        style_loader_ = torch.utils.data.DataLoader(dataset=style_dataset,
                                                  batch_size=opt.batchSize,
                                                  shuffle=True,
                                                  num_workers=1,
                                                  drop_last=True)
        style_loader = iter(style_loader_)
        print("Style dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading style dataset: {e}")
        return

    ################# MODEL #################
    try:
        print("Initializing models...")
        vgg5 = loss_network()
        vgg = encoder4()
        dec = decoder4()
        
        # Create matrix layers only for r11 and r21
        print(f"Creating matrix layers for: {WORKING_LAYERS}")
        matrix = {}
        for layer in WORKING_LAYERS:
            try:
                matrix[layer] = MulLayer(layer, matrixSize=32)
                print(f"Created matrix layer for {layer} with {LAYER_CONFIG[layer]} channels")
            except Exception as e:
                print(f"Error creating matrix layer {layer}: {e}")

        print("Loading model weights...")
        vgg.load_state_dict(torch.load(opt.vgg_dir, weights_only=True))
        dec.load_state_dict(torch.load(opt.decoder_dir, weights_only=True))
        vgg5.load_state_dict(torch.load(opt.loss_network_dir, weights_only=True))
        print("Model weights loaded successfully.")

        # Move models to GPU if available
        if opt.cuda:
            vgg.cuda()
            dec.cuda()
            vgg5.cuda()
            for layer in WORKING_LAYERS:
                print(f"Moving matrix layer {layer} to GPU")
                matrix[layer].cuda()
            print("All models and tensors moved to GPU.")

        # Freeze encoder and decoder parameters
        for param in vgg.parameters():
            param.requires_grad = False
        for param in vgg5.parameters():
            param.requires_grad = False
        for param in dec.parameters():
            param.requires_grad = False
        print("Freezing parameters for vgg, vgg5, and decoder...")

    except Exception as e:
        print(f"Error in model initialization: {e}")
        return

    ################# LOSS & OPTIMIZER #################
    criterion = LossCriterion(opt.style_layers,
                            opt.content_layers,
                            opt.style_weight,
                            opt.content_weight)
    
    # Create separate optimizers for each layer
    optimizers = {
        layer: optim.Adam(matrix[layer].parameters(), opt.lr) 
        for layer in WORKING_LAYERS
    }

    ################# TRAINING #################
    print("Starting training...")
    contentV = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
    styleV = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)

    if opt.cuda:
        contentV = contentV.cuda()
        styleV = styleV.cuda()

    for iteration in range(1, opt.niter + 1):
        print(f"\nIteration {iteration}/{opt.niter}")
        
        # Zero all optimizers
        for opt_layer in optimizers.values():
            opt_layer.zero_grad()

        try:
            content, _ = next(content_loader)
            style, _ = next(style_loader)
        except StopIteration:
            content_loader = iter(content_loader_)
            style_loader = iter(style_loader_)
            content, _ = next(content_loader)
            style, _ = next(style_loader)
        print("Batches fetched successfully.")

        contentV.resize_(content.size()).copy_(content)
        styleV.resize_(style.size()).copy_(style)

        # Forward pass
        print("Computing features...")
        sF = vgg(styleV)
        cF = vgg(contentV)

        # Process each layer individually
        print("Processing features through matrix layers...")
        features = {}
        transformations = {}
        losses = {}

        for layer in WORKING_LAYERS:
            try:
                # Get transformed features and transformation matrix
                features[layer], transformations[layer] = matrix[layer](cF[layer], sF[layer])
                print(f"Processed layer {layer} successfully")
                
                # Process each layer's output individually
                transfer_layer = dec(features[layer])
                
                # Compute loss for each layer
                sF_loss = vgg5(styleV)
                cF_loss = vgg5(contentV)
                tF = vgg5(transfer_layer)
                loss_layer, styleLoss_layer, contentLoss_layer = criterion(tF, sF_loss, cF_loss)
                losses[layer] = {
                    'total': loss_layer,
                    'style': styleLoss_layer,
                    'content': contentLoss_layer
                }
                
                # Backward pass for each layer individually
                loss_layer.backward()
                optimizers[layer].step()
                
                print(f"Layer {layer} - Loss: {loss_layer.item():.4f}, "
                      f"Style: {styleLoss_layer.item():.4f}, "
                      f"Content: {contentLoss_layer.item():.4f}")

                # Save transformed images for each layer
                if iteration % opt.log_interval == 0:
                    transfer_layer = transfer_layer.clamp(0, 1)
                    concat = torch.cat((content, style, transfer_layer.cpu()), dim=0)
                    vutils.save_image(concat, f'{opt.outf}/{layer}_{iteration}.png',
                                    normalize=True, scale_each=True, nrow=opt.batchSize)

            except Exception as e:
                print(f"Error processing layer {layer}: {e}")
                continue

        # Adjust learning rates
        for layer in WORKING_LAYERS:
            adjust_learning_rate(optimizers[layer], iteration)

        # Save transformation matrices and model states
        if iteration > 0 and iteration % opt.save_interval == 0:
            for layer in WORKING_LAYERS:
                # Save model state
                torch.save(matrix[layer].state_dict(), 
                         f'{opt.outf}/matrix_{layer}_{iteration}.pth')
                
                # Save transformation matrix
                if layer in transformations:
                    torch.save(transformations[layer], 
                             f'{opt.outf}/transformation_{layer}_{iteration}.pth')

if __name__ == '__main__':
    main()