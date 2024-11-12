import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from libs.Loader import Dataset
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from libs.utils import print_options
from libs.models import decoder1, decoder2

parser = argparse.ArgumentParser()
parser.add_argument("--contentPath", default="datasets/coco2014/images/train2014/",
                    help='path to training images')
parser.add_argument("--outf", default="trained_models/",
                    help='folder to output model checkpoints')
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
parser.add_argument("--layer", default="r11",
                    help='which features to train for: r11 or r21')
parser.add_argument("--log_interval", type=int, default=500,
                    help='log interval')
parser.add_argument("--gpu_id", type=int, default=0,
                    help='which gpu to use')
parser.add_argument("--save_interval", type=int, default=5000,
                    help='checkpoint save interval')

################# PREPARATIONS #################
opt = parser.parse_args()
opt.cuda = torch.cuda.is_available()
if(opt.cuda):
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

class encoder1(nn.Module):
    def __init__(self):
        super(encoder1,self).__init__()
        # encoder for r11 (64 channels)
        self.conv1 = nn.Conv2d(3,3,1,1,0)
        self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
        self.conv2 = nn.Conv2d(3,64,3,1,0)
        self.relu2 = nn.ReLU(inplace=True)
        self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv3 = nn.Conv2d(64,64,3,1,0)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self,x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        out = self.relu3(out)
        return out

class encoder2(nn.Module):
    def __init__(self):
        super(encoder2,self).__init__()
        # encoder for r21 (128 channels)
        # First part same as encoder1
        self.conv1 = nn.Conv2d(3,3,1,1,0)
        self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
        self.conv2 = nn.Conv2d(3,64,3,1,0)
        self.relu2 = nn.ReLU(inplace=True)
        self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv3 = nn.Conv2d(64,64,3,1,0)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Additional layers for r21
        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv4 = nn.Conv2d(64,128,3,1,0)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self,x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.maxPool(out)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        return out

################# MODEL #################
if opt.layer == 'r11':
    enc = encoder1()
    dec = decoder1()
elif opt.layer == 'r21':
    enc = encoder2()
    dec = decoder2()
else:
    raise ValueError("Layer must be either r11 or r21")

# Initialize weights for both encoder and decoder
for m in [enc.modules(), dec.modules()]:
    for layer in m:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

################# LOSS & OPTIMIZER #################
criterion = nn.MSELoss()
# Optimize both encoder and decoder
optimizer = optim.Adam([
    {'params': enc.parameters()},
    {'params': dec.parameters()}
], opt.lr)

################# GPU  #################
if(opt.cuda):
    enc.cuda()
    dec.cuda()
    criterion.cuda()

def adjust_learning_rate(optimizer, iteration):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr / (1+iteration*1e-5)

################# TRAINING #################
print(f"\nTraining autoencoder for {opt.layer} features...")
for iteration in range(1, opt.niter+1):
    optimizer.zero_grad()
    
    # Get content images
    try:
        content, _ = next(content_loader)
    except StopIteration:
        content_loader = iter(content_loader_)
        content, _ = next(content_loader)

    if opt.cuda:
        content = content.cuda()

    # Forward pass through encoder and decoder
    features = enc(content)
    reconstructed = dec(features)
    
    # Loss computation
    loss = criterion(reconstructed, content)
    
    # Backward & optimization
    loss.backward()
    optimizer.step()
    
    # Adjust learning rate
    adjust_learning_rate(optimizer, iteration)
    
    # Logging
    if iteration % opt.log_interval == 0:
        print(f'Iteration: [{iteration}/{opt.niter}] Loss: {loss.item():.4f}')
        
        with torch.no_grad():
            # Move both tensors to CPU before concatenating
            content_vis = content.detach().cpu()
            reconstructed_vis = reconstructed.detach().cpu().clamp(0, 1)
            
            # Concatenate images (both now on CPU)
            concat = torch.cat((content.cpu(), reconstructed.cpu()), dim=0)

            
            # Save images
            vutils.save_image(
                concat,
                f'{opt.outf}/reconstruction_{iteration}.png',
                normalize=True,
                scale_each=True,
                nrow=opt.batchSize
            )
    
    # Save models
    if iteration % opt.save_interval == 0:
        # Save encoder
        torch.save(enc.state_dict(), f'{opt.outf}/encoder_{opt.layer}.pth')
        # Save decoder
        torch.save(dec.state_dict(), f'{opt.outf}/decoder_{opt.layer}.pth')

print(f"Training completed. Models saved in {opt.outf}")