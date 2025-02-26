class Options:
    
    def __init__(self):
        self.contentPath = "data/content/"
        self.stylePath = "data/style/"
        self.loadSize = 256
        self.fineSize = 256
        self.matrixPath = "Matrices/"
        self.vgg_dir = 'models/vgg_r41.pth'
        self.decoder_dir = 'models/dec_r41.pth'
        self.layer = 'r41'
        self.outf = "Artistic/"
        self.cuda = torch.cuda.is_available()
        self.batchSize = 1
        self.matrixPath = 'models/r41.pth'


if __name__ == "__main__":

    import os
    import torch
    import argparse
    from libs.Loader import Dataset
    from libs.Matrix import MulLayer
    import torchvision.utils as vutils
    import torch.backends.cudnn as cudnn
    from libs.utils import print_options
    from libs.models import encoder3, encoder4, encoder5
    from libs.models import decoder3, decoder4, decoder5
    from PIL import Image
    import torchvision.transforms as transforms
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
    #                     help='pre-trained encoder path')
    # parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
    #                     help='pre-trained decoder path')
    # parser.add_argument("--matrixPath", default='models/r41.pth',
    #                     help='pre-trained model path')
    # parser.add_argument("--stylePath", default="data/style",
    #                     help='path to a single style image')
    # parser.add_argument("--contentPath", default="data/content",
    #                     help='path to a single content image')
    # parser.add_argument("--outf", default="Artistic/",
    #                     help='path to save the transferred image')
    # parser.add_argument("--batchSize", type=int, default=1,
    #                     help='batch size (set to 1 for single image)')
    # parser.add_argument('--loadSize', type=int, default=256,
    #                     help='scale image size')
    # parser.add_argument('--fineSize', type=int, default=256,
    #                     help='crop image size')
    # parser.add_argument("--layer", default="r41",
    #                     help='which features to transfer, either r31 or r41')


    opt=Options()
    opt.cuda = torch.cuda.is_available()
    print_options(opt)

    os.makedirs(opt.outf, exist_ok=True)
    cudnn.benchmark = True

    ################# MODEL #################
    if opt.layer == 'r31':
        vgg = encoder3()
        dec = decoder3()
    elif opt.layer == 'r41':
        vgg = encoder4()
        dec = decoder4()
    matrix = MulLayer(opt.layer)
    vgg.load_state_dict(torch.load(opt.vgg_dir))
    dec.load_state_dict(torch.load(opt.decoder_dir))
    matrix.load_state_dict(torch.load(opt.matrixPath))


    ################# GPU #################
    if opt.cuda:
        vgg.cuda()
        dec.cuda()
        matrix.cuda()


    content_files = [f for f in os.listdir(opt.contentPath) if f.endswith(('.jpg', '.jpeg', '.png'))]
    style_files = [f for f in os.listdir(opt.stylePath) if f.endswith(('.jpg', '.jpeg', '.png'))]

    transform = transforms.Compose([
                transforms.ToTensor(),  # Convert the image to a tensor
                transforms.Resize((opt.fineSize, opt.fineSize))  # Resize the image
            ])

    # Add progress bar for the style loop
    for style in tqdm(style_files, desc="Processing Styles"):
        style_image = Image.open(opt.stylePath + style).convert('RGB')
        style_tensor = transform(style_image).unsqueeze(0)

        # Add progress bar for the content loop
        for content in tqdm(content_files, desc="Processing Contents", leave=False):
            content_image = Image.open(opt.contentPath + content).convert('RGB')
            content_tensor = transform(content_image).unsqueeze(0)

            contentV = torch.Tensor(1, 3, opt.fineSize, opt.fineSize).copy_(content_tensor)
            styleV = torch.Tensor(1, 3, opt.fineSize, opt.fineSize).copy_(style_tensor)
            contentV = contentV.cuda()
            styleV = styleV.cuda()

            ################# FORWARD PASS WITH NOISE #################
            images = []  # List to store the images with noise
            noise_levels = list(range(0, 101, 10))  # Noise levels from 0 to 100

            with torch.no_grad():
                sF = vgg(styleV)
                cF = vgg(contentV)

                for sigma in noise_levels:  # Vary noise levels from 0 to 100
                    # Compute the transformation matrix as usual
                    if opt.layer == 'r41':
                        feature, transmatrix = matrix(cF[opt.layer], sF[opt.layer], trans=True)
                    else:
                        feature, transmatrix = matrix(cF, sF, trans=True)
                    
                    # Apply the noise to the transformation matrix
                    noise = torch.randn_like(transmatrix) * (sigma / 100.0)
                    noisy_matrix = transmatrix + noise

                    # Apply the noisy matrix to the compressed content
                    compress_content = matrix.compress(cF[opt.layer] if opt.layer == 'r41' else cF)
                    b, c, h, w = compress_content.size()
                    compress_content = compress_content.view(b, c, -1)

                    # Perform batch matrix multiplication and reshape to the correct dimensions
                    transfeature = torch.bmm(noisy_matrix, compress_content).view(b, matrix.matrixSize, h, w)

                    # Decompress and add the mean back
                    out = matrix.unzip(transfeature)
                    out = out + torch.mean(cF[opt.layer if opt.layer == 'r41' else cF], dim=(2, 3), keepdim=True)

                    transfer_noisy = dec(out)
                    transfer_noisy = transfer_noisy.clamp(0, 1)

                    # Convert the tensor to a numpy array for visualization
                    img_numpy = transfer_noisy.squeeze().cpu().numpy().transpose(1, 2, 0)
                    images.append(img_numpy)  # Store the image in the list

                    torch.cuda.empty_cache()

            # Plot all the noisy images in a comparative plot
            fig, axes = plt.subplots(2, 6, figsize=(20, 7))
            for idx, ax in enumerate(axes.flatten()):
                if idx < len(images):
                    ax.imshow(images[idx])
                    ax.axis('off')
                    ax.set_title(f'Sigma={noise_levels[idx]}')
            plt.tight_layout()
            plt.savefig(f'{opt.outf}comparative_plot_{style}_{content}.png')  # Save file with unique name
