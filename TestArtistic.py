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

    parser = argparse.ArgumentParser()
    parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                        help='pre-trained encoder path')
    parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                        help='pre-trained decoder path')
    parser.add_argument("--matrixPath", default='models/r41.pth',
                        help='pre-trained model path')
    parser.add_argument("--stylePath", default="data/content2",
                        help='path to style image')
    parser.add_argument("--contentPath", default="data/content2",
                        help='path to frames')
    parser.add_argument("--outf", default="Artistic_temp/",
                        help='path to transferred images')
    parser.add_argument("--matrixOutf", default="Matrices_temp/",
                        help='path to save transformation matrices')
    parser.add_argument("--batchSize", type=int, default=1,
                        help='batch size')
    parser.add_argument('--loadSize', type=int, default=256,
                        help='scale image size')
    parser.add_argument('--fineSize', type=int, default=256,
                        help='crop image size')
    parser.add_argument("--layer", default="r41",
                        help='which features to transfer, either r31 or r41')

    ################# PREPARATIONS #################
    opt = parser.parse_args()
    opt.cuda = torch.cuda.is_available()
    print_options(opt)

    os.makedirs(opt.outf, exist_ok=True)
    os.makedirs(opt.matrixOutf, exist_ok=True)
    cudnn.benchmark = True

    ################# DATA #################
    content_dataset = Dataset(opt.contentPath, opt.loadSize, opt.fineSize)
    content_loader = torch.utils.data.DataLoader(dataset=content_dataset,
                                                 batch_size=opt.batchSize,
                                                 shuffle=False,
                                                 num_workers=0)

    style_dataset = Dataset(opt.stylePath, opt.loadSize, opt.fineSize)
    style_loader = torch.utils.data.DataLoader(dataset=style_dataset,
                                               batch_size=opt.batchSize,
                                               shuffle=False,
                                               num_workers=0)

    ################# MODEL #################
    if(opt.layer == 'r31'):
        vgg = encoder3()
        dec = decoder3()
    elif(opt.layer == 'r41'):
        vgg = encoder4()
        dec = decoder4()
    matrix = MulLayer(opt.layer)
    vgg.load_state_dict(torch.load(opt.vgg_dir))
    dec.load_state_dict(torch.load(opt.decoder_dir))
    matrix.load_state_dict(torch.load(opt.matrixPath))

    ################# GLOBAL VARIABLE #################
    contentV = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
    styleV = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)

    ################# GPU #################
    if(opt.cuda):
        vgg.cuda()
        dec.cuda()
        matrix.cuda()
        contentV = contentV.cuda()
        styleV = styleV.cuda()

    # Add a counter variable outside the loop
    image_counter = 0

    for ci, (content, contentName) in enumerate(content_loader):
        contentName = contentName[0]
        contentV.resize_(content.size()).copy_(content)

        for sj, (style, styleName) in enumerate(style_loader):
            styleName = styleName[0]
            styleV.resize_(style.size()).copy_(style)

            # forward
            with torch.no_grad():
                sF = vgg(styleV)
                cF = vgg(contentV)

                if(opt.layer == 'r41'):
                    feature, transmatrix = matrix(cF[opt.layer], sF[opt.layer])
                else:
                    feature, transmatrix = matrix(cF, sF)
                transfer = dec(feature)

            transfer = transfer.clamp(0, 1)

            # Increment the counter for each image
            image_counter += 1

            # Save image with a unique name using the counter
            vutils.save_image(transfer, f'{opt.outf}/{contentName}_{styleName}_{image_counter}.png',
                              normalize=True, scale_each=True, nrow=opt.batchSize)

            # Save transformation matrix with a unique name using the counter
            torch.save(transmatrix, f'{opt.matrixOutf}/{contentName}_{styleName}_{image_counter}_matrix.pth')

            print(f'Transferred image saved at {opt.outf}{contentName}_{styleName}_{image_counter}.png')
            print(f'Transformation matrix saved at {opt.matrixOutf}{contentName}_{styleName}_{image_counter}_matrix.pth')