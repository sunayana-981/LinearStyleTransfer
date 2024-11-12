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
from transformers import ViTModel

################# ARGUMENT PARSING #################
parser = argparse.ArgumentParser()
# Paths for pre-trained encoders and datasets
parser.add_argument("--vgg_dir", default='models/vgg_r41.pth', help='pre-trained encoder path')
parser.add_argument("--loss_network_dir", default='models/vgg_r51.pth', help='used for loss network')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth', help='pre-trained decoder path')
parser.add_argument("--stylePath", default="datasets/wikiArt/", help='path to wikiArt dataset')
parser.add_argument("--contentPath", default="datasets/coco2014/images/train2014/", help='path to MSCOCO dataset')
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
parser.add_argument("--use_dino", type=bool, default=True,
                    help='whether to use additional DINO loss')
parser.add_argument("--dino_weight", type=float, default=0.1,
                    help='weight for DINO style loss')

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
# Original VGG networks
vgg5 = loss_network()
if(opt.layer == 'r31'):
    matrix = MulLayer('r31')
    vgg = encoder3()
    dec = decoder3()
elif(opt.layer == 'r41'):
    matrix = MulLayer('r41')
    vgg = encoder4()
    dec = decoder4()

# Load pretrained weights
vgg.load_state_dict(torch.load(opt.vgg_dir))
dec.load_state_dict(torch.load(opt.decoder_dir))
vgg5.load_state_dict(torch.load(opt.loss_network_dir))

# Initialize DINO model for additional style guidance
dino = ViTModel.from_pretrained("facebook/dino-vits16")
for param in dino.parameters():
    param.requires_grad = False

# Freeze original networks
for param in vgg.parameters():
    param.requires_grad = False
for param in vgg5.parameters():
    param.requires_grad = False
for param in dec.parameters():
    param.requires_grad = False

################# LOSS & OPTIMIZER #################
criterion = LossCriterion(opt.style_layers,
                         opt.content_layers,
                         opt.style_weight,
                         opt.content_weight)
optimizer = optim.Adam(matrix.parameters(), opt.lr)

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    vgg5.cuda()
    matrix.cuda()
    dino.cuda()
    contentV = contentV.cuda()
    styleV = styleV.cuda()

################# TRAINING #################
for iteration in range(1, opt.niter+1):
    optimizer.zero_grad()
    
    # Load images (keeping original loading code)
    try:
        content, _ = content_loader.next()
    except StopIteration:
        content_loader = iter(content_loader_)
        content, _ = content_loader.next()
    
    try:
        style, _ = style_loader.next()
    except StopIteration:
        style_loader = iter(style_loader_)
        style, _ = style_loader.next()

    contentV.resize_(content.size()).copy_(content)
    styleV.resize_(style.size()).copy_(style)

    # Original VGG forward pass
    sF = vgg(styleV)
    cF = vgg(contentV)

    # Matrix transformation (original method)
    if(opt.layer == 'r41'):
        feature, transmatrix = matrix(cF[opt.layer], sF[opt.layer])
    else:
        feature, transmatrix = matrix(cF, sF)
    transfer = dec(feature)

    # Original losses
    sF_loss = vgg5(styleV)
    cF_loss = vgg5(contentV)
    tF = vgg5(transfer)
    vgg_loss, styleLoss, contentLoss = criterion(tF, sF_loss, cF_loss)

    # Additional DINO-based style loss
    if opt.use_dino:
        with torch.no_grad():
            dino_style = dino(styleV).last_hidden_state
            dino_transfer = dino(transfer).last_hidden_state
            
            # DINO style loss (using patch features, excluding CLS token)
            dino_style_loss = F.mse_loss(
                dino_transfer[:, 1:], 
                dino_style[:, 1:]
            )
            
            # Total loss combining VGG and DINO
            loss = vgg_loss + opt.dino_weight * dino_style_loss
    else:
        loss = vgg_loss

    # backward & optimization
    loss.backward()
    optimizer.step()
    
    # Adjust learning rate
    adjust_learning_rate(optimizer, iteration)

    # Logging and saving (keeping original code)
    if iteration % opt.log_interval == 0:
        print('Iteration: [%d/%d] Loss: %.4f ContentLoss: %.4f StyleLoss: %.4f DINOLoss: %.4f'%
              (iteration, opt.niter, loss.item(), contentLoss.item(), 
               styleLoss.item(), dino_style_loss.item() if opt.use_dino else 0))
        
        transfer = transfer.clamp(0, 1)
        concat = torch.cat((content, style, transfer.cpu()), dim=0)
        vutils.save_image(concat, '%s/%d.png'%(opt.outf, iteration),
                         normalize=True, scale_each=True, nrow=opt.batchSize)

    if iteration % opt.save_interval == 0:
        torch.save(matrix.state_dict(), '%s/%s.pth' % (opt.outf, opt.layer))

# After parsing arguments
print("\nParsing style layers:")
print(f"Raw style_layers argument: {opt.style_layers}")
opt.style_layers = opt.style_layers.split(',')
print(f"Processed style layers: {opt.style_layers}")

# After loading weights, add this check
print("\nVerifying decoder weights:")
for layer, decoder in decoders.items():
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"- {layer} decoder: {total_params} parameters")

# After initializing decoders
print("\nVerifying decoder initialization:")
for layer in decoders:
    print(f"- {layer} decoder:")
    print(f"  Input channels: {decoders[layer].reflecPad7.padding}")  # First layer
    if hasattr(decoders[layer], 'conv11'):
        print(f"  Output channels: {decoders[layer].conv11.out_channels}")  # Last layer

# Before training loop
print("\nAvailable layers:")
print(f"Style layers: {opt.style_layers}")
print(f"Initialized decoders: {list(decoders.keys())}")
print(f"Initialized matrices: {list(matrix.keys())}")