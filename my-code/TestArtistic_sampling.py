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
    parser.add_argument("--matrixPath", default='trainingOutput/r41.pth',
                        help='pre-trained model path')
    parser.add_argument("--stylePath", default="data/style/",
                        help='path to style image')
    parser.add_argument("--contentPath", default="data/content/",
                        help='path to frames')
    parser.add_argument("--outf", default="outputs1/",
                        help='path to transferred images')
    parser.add_argument("--matrixOutf", default="Matrices/",
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

    # Get a list of all content and style files, ignoring non-image files
    content_files = sorted([f for f in os.listdir(opt.contentPath) if f.endswith(('.png', '.jpg', '.jpeg'))])
    style_files = sorted([f for f in os.listdir(opt.stylePath) if f.endswith(('.png', '.jpg', '.jpeg'))])

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

    # Iterate over styles
    for styleIdx, styleName in enumerate(style_files):
        style_image_path = os.path.join(opt.stylePath, styleName)
        style_image = style_dataset[styleIdx][0].unsqueeze(0)  # Load style image
        styleV.copy_(style_image)

        # Create a directory for each style image
        style_base = os.path.splitext(styleName)[0]
        style_output_dir = os.path.join(opt.outf, style_base)
        style_matrix_dir = os.path.join(opt.matrixOutf, style_base)

        # Ensure the directories exist for each style
        os.makedirs(style_output_dir, exist_ok=True)
        os.makedirs(style_matrix_dir, exist_ok=True)

        # Reset the counter for each style
        image_counter = 0

        for contentIdx, contentName in enumerate(content_files):
            content_image_path = os.path.join(opt.contentPath, contentName)
            content_image = content_dataset[contentIdx][0].unsqueeze(0)  # Load content image
            contentV.copy_(content_image)

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

            # Create base names based on content and style image names (excluding file extensions)
            content_base = os.path.splitext(contentName)[0]

            # Save paths (using style directories and content + style names in the filename)
            output_img_path = f"{style_output_dir}/{content_base}_{style_base}_{image_counter}.png"
            matrix_save_path = f"{style_matrix_dir}/{content_base}_{style_base}_{image_counter}_matrix.pth"

            # Save the image and matrix
            vutils.save_image(transfer, output_img_path, normalize=True, scale_each=True, nrow=opt.batchSize)
            torch.save(transmatrix, matrix_save_path)

            # Output success message
            print(f'Transferred image saved at {output_img_path}')
            print(f'Transformation matrix saved at {matrix_save_path}')

            # Debugging message to confirm path structure
            print(f"Style directory: {style_output_dir}, Content: {content_base}, Counter: {image_counter}")


