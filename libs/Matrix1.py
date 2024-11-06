import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, layer, matrixSize=32):
        super(CNN, self).__init__()
        # Dynamically initialize convolution layers based on the given layer
        layer_config = {
            'r11': (64, [64, 32]),
            'r21': (128, [128, 64]),
            'r31': (256, [256, 128, 64]),
            'r41': (512, [512, 256, 128]),
            'r51': (512, [512, 256, 128])
        }
        input_channels, channels = layer_config.get(layer, (256, [128, 64]))

        layers = []
        for out_channels in channels:
            layers.append(nn.Conv2d(input_channels, out_channels, 3, 1, 1))
            layers.append(nn.ReLU(inplace=True))
            input_channels = out_channels
        layers.append(nn.Conv2d(input_channels, matrixSize, 3, 1, 1))
        
        self.convs = nn.Sequential(*layers)
        self.fc = nn.Linear(matrixSize * matrixSize, matrixSize * matrixSize)

    def forward(self, x):
        out = self.convs(x)
        b, c, h, w = out.size()
        out = out.view(b, c, -1)
        out = torch.bmm(out, out.transpose(1, 2)).div(h * w)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class MulLayer(nn.Module):
    def __init__(self, layer, matrixSize=32):
        super(MulLayer, self).__init__()
        self.snet = CNN(layer, matrixSize)
        self.cnet = CNN(layer, matrixSize)
        self.matrixSize = matrixSize

        # Compression and decompression layers to adjust feature map channels
        layer_config = {
            'r11': 64,
            'r21': 128,
            'r31': 256,
            'r41': 512,
            'r51': 512
        }
        in_channels = layer_config.get(layer, 256)
        self.compress = nn.Conv2d(in_channels, matrixSize, 1, 1, 0)
        self.unzip = nn.Conv2d(matrixSize, in_channels, 1, 1, 0)
        self.transmatrix = None

    def forward(self, cF, sF, trans=True):
        # Backup of original content feature map
        cFBK = cF.clone()

        # Center the content feature map
        cb, cc, ch, cw = cF.size()
        cFF = cF.view(cb, cc, -1)
        cMean = torch.mean(cFF, dim=2, keepdim=True).unsqueeze(3).expand_as(cF)
        cF = cF - cMean

        # Center the style feature map
        sb, sc, sh, sw = sF.size()
        sFF = sF.view(sb, sc, -1)
        sMean = torch.mean(sFF, dim=2, keepdim=True).unsqueeze(3)
        sMeanC = sMean.expand_as(cF)
        sMeanS = sMean.expand_as(sF)
        sF = sF - sMeanS

        # Compress content feature map to match the matrix size
        compress_content = self.compress(cF)
        b, c, h, w = compress_content.size()
        compress_content = compress_content.reshape(b, c, -1)

        if trans:
            # Compute transformation matrices for content and style
            cMatrix = self.cnet(cF)
            sMatrix = self.snet(sF)

            # Reshape matrices to match dimensions for batch matrix multiplication
            sMatrix = sMatrix.view(sMatrix.size(0), self.matrixSize, self.matrixSize)
            cMatrix = cMatrix.view(cMatrix.size(0), self.matrixSize, self.matrixSize)

            # Compute the transformation matrix by multiplying style and content matrices
            transmatrix = torch.bmm(sMatrix, cMatrix)

            # Apply the transformation to the compressed content
            transfeature = torch.bmm(transmatrix, compress_content).view(b, c, h, w)

            # Decompress and add the mean back
            out = self.unzip(transfeature.view(b, c, h, w))
            out = out + sMeanC
            return out, transmatrix
        else:
            # Decompress without transformation and add the mean back
            out = self.unzip(compress_content.view(b, c, h, w))
            out = out + cMean
            return out
