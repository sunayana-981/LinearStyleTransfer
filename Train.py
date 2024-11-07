import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from libs.Loader import Dataset
from libs.Matrix import MulLayer
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from libs.utils import print_options
from libs.Criterion import LossCriterion
from libs.models import encoder3, encoder4
from libs.models import decoder3, decoder4
from libs.models import encoder5 as loss_network

################# ARGUMENT PARSING #################
parser = argparse.ArgumentParser()
# Paths for pre-trained encoders and datasets
parser.add_argument("--vgg_dir", default='models/vgg_r41.pth', help='pre-trained encoder path')
parser.add_argument("--loss_network_dir", default='models/vgg_r51.pth', help='used for loss network')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth', help='pre-trained decoder path')
parser.add_argument("--stylePath", default="/datasets/wikiArt/train/images/", help='path to wikiArt dataset')
parser.add_argument("--contentPath", default="datasets/MSCOCO/train2014/images/", help='path to MSCOCO dataset')
# Output folder for images and model checkpoints
parser.add_argument("--outf", default="trainingOutput/", help='folder to output images and model checkpoints')
# Model hyperparameters and training settings
parser.add_argument("--content_layers", default="r41", help='layers for content')
parser.add_argument("--style_layers", default="r11,r21,r31,r41", help='layers for style')
parser.add_argument("--batchSize", type=int, default=8, help='batch size')
parser.add_argument("--niter", type=int, default=100000, help='iterations to train the model')
parser.add_argument('--loadSize', type=int, default=300, help='scale image size')
parser.add_argument('--fineSize', type=int, default=256, help='crop image size')
parser.add_argument("--lr", type=float, default=1e-4, help='learning rate')
parser.add_argument("--content_weight", type=float, default=1.0, help='content loss weight')
parser.add_argument("--style_weight", type=float, default=0.02, help='style loss weight')
parser.add_argument("--log_interval", type=int, default=500, help='log interval')
parser.add_argument("--gpu_id", type=int, default=0, help='which gpu to use')
parser.add_argument("--save_interval", type=int, default=5000, help='checkpoint save interval')

################# PREPARATIONS #################
opt = parser.parse_args()
opt.content_layers = opt.content_layers.split(',')
opt.style_layers = opt.style_layers.split(',')
opt.cuda = torch.cuda.is_available()
if opt.cuda:
    torch.cuda.set_device(opt.gpu_id)

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
# Load the encoder and the loss network
vgg5 = loss_network()
vgg = encoder4()  # default encoder for r41
dec = decoder4()  # default decoder for r41

# Create matrix layers based on style layers
matrix = {}
if 'r11' in opt.style_layers:
    matrix['r11'] = MulLayer('r11')
if 'r21' in opt.style_layers:
    matrix['r21'] = MulLayer('r21')
if 'r31' in opt.style_layers:
    matrix['r31'] = MulLayer('r31')
if 'r41' in opt.style_layers:
    matrix['r41'] = MulLayer('r41')

# Load the pretrained models
vgg.load_state_dict(torch.load(opt.vgg_dir))
dec.load_state_dict(torch.load(opt.decoder_dir))
vgg5.load_state_dict(torch.load(opt.loss_network_dir))

# Freeze the weights for the pre-trained networks
for param in vgg.parameters():
    param.requires_grad = False
for param in vgg5.parameters():
    param.requires_grad = False
for param in dec.parameters():
    param.requires_grad = False

################# LOSS & OPTIMIZER #################
# Define the loss function and optimizers for each matrix layer
criterion = LossCriterion(opt.style_layers, opt.content_layers, opt.style_weight, opt.content_weight)
optimizers = {layer: optim.Adam(matrix[layer].parameters(), opt.lr) for layer in matrix.keys()}

################# GLOBAL VARIABLES #################
contentV = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
styleV = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)

################# GPU SETUP #################
if opt.cuda:
    vgg.cuda()
    dec.cuda()
    vgg5.cuda()
    for layer in matrix:
        matrix[layer].cuda()
    contentV = contentV.cuda()
    styleV = styleV.cuda()

################# TRAINING LOOP #################
def adjust_learning_rate(optimizer, iteration):
    """Adjusts the learning rate based on iteration count"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr / (1 + iteration * 1e-5)

for iteration in range(1, opt.niter + 1):
    # Zero the gradients for each optimizer
    for optimizer in optimizers.values():
        optimizer.zero_grad()

    # Load the content and style images
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

    contentV.resize_(content.size()).copy_(content)
    styleV.resize_(style.size()).copy_(style)

    # Forward pass: Extract features from VGG encoder
    sF = vgg(styleV)
    cF = vgg(contentV)

    total_loss = 0
    for layer in matrix:
        # Forward through the transformation matrix and decoder
        feature, transmatrix = matrix[layer](cF[layer], sF[layer])
        transfer = dec(feature)

        # Compute losses
        sF_loss = vgg5(styleV)
        cF_loss = vgg5(contentV)
        tF = vgg5(transfer)
        loss, styleLoss, contentLoss = criterion(tF, sF_loss, cF_loss)

        # Backward pass and optimization
        loss.backward()
        optimizers[layer].step()

        total_loss += loss.item()

    # Logging information
    print(f'Iteration: [{iteration}/{opt.niter}] Total Loss: {total_loss:.4f}')

    # Adjust learning rate
    adjust_learning_rate(optimizer, iteration)

    # Save images every log interval
    if iteration % opt.log_interval == 0:
        transfer = transfer.clamp(0, 1)
        concat = torch.cat((content, style, transfer.cpu()), dim=0)
        vutils.save_image(concat, f'{opt.outf}/{iteration}.png', normalize=True, scale_each=True, nrow=opt.batchSize)

    # Save matrix weights at specified save intervals
    if iteration % opt.save_interval == 0:
        for layer in matrix:
            torch.save(matrix[layer].state_dict(), f'{opt.outf}/{layer}_{iteration}.pth')

