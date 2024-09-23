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
from libs.models import encoder3, encoder4, decoder3, decoder4
from libs.models import encoder5 as loss_network
from transformers import DINOModel

# Custom Style Transformation Module
class StyleTransformationModule(nn.Module):
    def __init__(self):
        super(StyleTransformationModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.covariance_layer = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 16 * 16, 128)  # Assuming 16x16 feature size after conv layers
        self.fc2 = nn.Linear(128, 64)  # Reduce to match content matrix size

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.covariance_layer(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

# Argument parsing and configuration
parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_r41.pth')
parser.add_argument("--loss_network_dir", default='models/vgg_r51.pth')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth')
parser.add_argument("--stylePath", default="datasets/wikiArt/train/images/")
parser.add_argument("--contentPath", default="datasets/MSCOCO/train2014/images/")
parser.add_argument("--outf", default="trainingOutput/")
parser.add_argument("--content_layers", default="r41")
parser.add_argument("--style_layers", default="r11,r21,r31,r41")
parser.add_argument("--batchSize", type=int, default=8)
parser.add_argument("--niter", type=int, default=100000)
parser.add_argument('--loadSize', type=int, default=300)
parser.add_argument('--fineSize', type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--content_weight", type=float, default=1.0)
parser.add_argument("--style_weight", type=float, default=0.02)
parser.add_argument("--log_interval", type=int, default=500)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--save_interval", type=int, default=5000)
parser.add_argument("--layer", default="r41")

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
content_loader_ = torch.utils.data.DataLoader(dataset=content_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=1, drop_last=True)
content_loader = iter(content_loader_)
style_dataset = Dataset(opt.stylePath, opt.loadSize, opt.fineSize)
style_loader_ = torch.utils.data.DataLoader(dataset=style_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=1, drop_last=True)
style_loader = iter(style_loader_)

################# MODEL #################
vgg5 = loss_network()
if opt.layer == 'r31':
    matrix = MulLayer('r31')
    vgg = encoder3()
    dec = decoder3()
elif opt.layer == 'r41':
    matrix = MulLayer('r41')
    vgg = encoder4()
    dec = decoder4()
    
# vgg.load_state_dict(torch.load(opt.vgg_dir))
# dec.load_state_dict(torch.load(opt.decoder_dir))
# vgg5.load_state_dict(torch.load(opt.loss_network_dir))

#use the new version of torch.load to load the model
vgg.load_state_dict(torch.load(opt.vgg_dir, map_location=torch.device(opt.cuda)))
dec.load_state_dict(torch.load(opt.decoder_dir, map_location=torch.device(opt.cuda)))
vgg5.load_state_dict(torch.load(opt.loss_network_dir, map_location=torch.device(opt.cuda)))


for param in vgg.parameters():
    param.requires_grad = False
for param in vgg5.parameters():
    param.requires_grad = False
for param in dec.parameters():
    param.requires_grad = False

################# INTEGRATE DINO AND TRANSFORMATION MODULE #################
dino_model = DINOModel.from_pretrained('facebook/dino-vitb8')
transformation_module = StyleTransformationModule()

criterion = LossCriterion(opt.style_layers, opt.content_layers, opt.style_weight, opt.content_weight)
optimizer = optim.Adam(matrix.parameters(), opt.lr)

contentV = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
styleV = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)

if opt.cuda:
    vgg.cuda()
    dec.cuda()
    vgg5.cuda()
    matrix.cuda()
    dino_model.cuda()
    transformation_module.cuda()
    contentV = contentV.cuda()
    styleV = styleV.cuda()

################# TRAINING #################
def adjust_learning_rate(optimizer, iteration):
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr / (1 + iteration * 1e-5)

for iteration in range(1, opt.niter + 1):
    optimizer.zero_grad()
    try:
        content, _ = next(content_loader)
    except (IOError, StopIteration):
        content_loader = iter(content_loader_)
        content, _ = next(content_loader)

    try:
        style, _ = next(style_loader)
    except (IOError, StopIteration):
        style_loader = iter(style_loader_)
        style, _ = next(style_loader)

    contentV.resize_(content.size()).copy_(content)
    styleV.resize_(style.size()).copy_(style)

    # Forward pass for style: DINO + Transformation
    style_features = dino_model(styleV).last_hidden_state
    transformed_style = transformation_module(style_features)

    # Forward pass for content
    cF = vgg(contentV)
    content_matrix = matrix(cF[opt.layer])

    # Multiply transformed style features with content matrix
    feature = torch.matmul(content_matrix, transformed_style)
    transfer = dec(feature)

    # Loss computation
    sF_loss = vgg5(styleV)
    cF_loss = vgg5(contentV)
    tF = vgg5(transfer)
    loss, styleLoss, contentLoss = criterion(tF, sF_loss, cF_loss)

    # Backward & optimization
    loss.backward()
    optimizer.step()

    print(f'Iteration: [{iteration}/{opt.niter}] Loss: {loss.item()} contentLoss: {contentLoss.item()} styleLoss: {styleLoss.item()} Learning Rate: {optimizer.param_groups[0]["lr"]}')

    adjust_learning_rate(optimizer, iteration)

    if iteration % opt.log_interval == 0:
        transfer = transfer.clamp(0, 1)
        concat = torch.cat((content, style, transfer.cpu()), dim=0)
        vutils.save_image(concat, f'{opt.outf}/{iteration}.png', normalize=True, scale_each=True, nrow=opt.batchSize)

    if iteration > 0 and iteration % opt.save_interval == 0:
        torch.save(matrix.state_dict(), f'{opt.outf}/{opt.layer}_{iteration}.pth')

