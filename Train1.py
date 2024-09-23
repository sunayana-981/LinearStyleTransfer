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
from torchvision import transforms

def main():
    # Argument parsing and configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--vgg_dir", default='models/vgg_r41.pth')
    parser.add_argument("--loss_network_dir", default='models/vgg_r51.pth')
    parser.add_argument("--decoder_dir", default='models/dec_r41.pth')
    parser.add_argument("--stylePath", default="datasets/wikiArt/train/")
    parser.add_argument("--contentPath", default="datasets/coco2014/images/train2014/")
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
    device = torch.device("cuda" if opt.cuda else "cpu")
    if opt.cuda:
        torch.cuda.set_device(opt.gpu_id)

    os.makedirs(opt.outf, exist_ok=True)
    cudnn.benchmark = True
    print_options(opt)

    # Define the VGG transformation
    vgg_transform = transforms.Compose([
        transforms.Resize(opt.fineSize),
        transforms.CenterCrop(opt.fineSize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225])   # ImageNet stds
    ])

    ################# DATA #################
    content_dataset = Dataset(opt.contentPath, opt.loadSize, opt.fineSize, transform=vgg_transform)
    content_loader_ = torch.utils.data.DataLoader(
        dataset=content_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=1, drop_last=True)
    content_loader = iter(content_loader_)

    style_dataset = Dataset(opt.stylePath, opt.loadSize, opt.fineSize, transform=vgg_transform)
    style_loader_ = torch.utils.data.DataLoader(
        dataset=style_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=1, drop_last=True)
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

    # Load pre-trained models
    vgg.load_state_dict(torch.load(opt.vgg_dir, map_location=device))
    dec.load_state_dict(torch.load(opt.decoder_dir, map_location=device))
    vgg5.load_state_dict(torch.load(opt.loss_network_dir, map_location=device))

    for param in vgg.parameters():
        param.requires_grad = False
    for param in vgg5.parameters():
        param.requires_grad = False
    for param in dec.parameters():
        param.requires_grad = False

    criterion = LossCriterion(opt.style_layers, opt.content_layers, opt.style_weight, opt.content_weight)
    optimizer = optim.Adam(matrix.parameters(), opt.lr)

    if opt.cuda:
        vgg.to(device)
        dec.to(device)
        vgg5.to(device)
        matrix.to(device)

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

        content = content.to(device)
        style = style.to(device)

        # Get content and style features from VGG
        cF = vgg(content)
        sF = vgg(style)

        # Combine content and style features using the matrix module
        feature, transmatrix = matrix(cF[opt.layer], sF[opt.layer])

        # Alternatively, if you don't need transmatrix:
        # feature = matrix(cF[opt.layer], sF[opt.layer])[0]

        # Decode the combined features to get the stylized image
        transfer = dec(feature)

        # Compute losses
        sF_loss = vgg5(style)
        cF_loss = vgg5(content)
        tF = vgg5(transfer)
        loss, styleLoss, contentLoss = criterion(tF, sF_loss, cF_loss)

        # Backward & optimization
        loss.backward()
        optimizer.step()

        print(f'Iteration: [{iteration}/{opt.niter}] Loss: {loss.item():.4f} '
            f'contentLoss: {contentLoss.item():.4f} styleLoss: {styleLoss.item():.4f} '
            f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        adjust_learning_rate(optimizer, iteration)

        if iteration % opt.log_interval == 0:
            transfer = transfer.clamp(0, 1)
            concat = torch.cat((content.cpu(), style.cpu(), transfer.cpu()), dim=0)
            vutils.save_image(concat, f'{opt.outf}/{iteration}.png', normalize=True, scale_each=True, nrow=opt.batchSize)

        if iteration > 0 and iteration % opt.save_interval == 0:
            torch.save(matrix.state_dict(), f'{opt.outf}/{opt.layer}_{iteration}.pth')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
