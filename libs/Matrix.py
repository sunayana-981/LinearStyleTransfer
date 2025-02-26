import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, layer, matrixSize=32):
        super(CNN, self).__init__()
        if layer == 'r31':
            # 256x64x64
            self.convs = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, matrixSize, 3, 1, 1)
            )
        elif layer == 'r41':
            # 512x32x32
            self.convs = nn.Sequential(
                nn.Conv2d(512, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, matrixSize, 3, 1, 1)
            )
        elif layer == 'r21':
            # 128x128x128
            self.convs = nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, matrixSize, 3, 1, 1)
            )
        elif layer == 'r11':
            # 64x256x256
            self.convs = nn.Sequential(
                nn.Conv2d(64, matrixSize, 3, 1, 1)
            )
        else:
            raise ValueError(f"Unsupported layer: {layer}. Must be one of ['r11', 'r21', 'r31', 'r41']")

        # Fully connected layer to produce transformation matrix
        self.fc = nn.Linear(matrixSize * matrixSize, matrixSize * matrixSize)

    def forward(self, x):
        out = self.convs(x)
        # 32x8x8
        b, c, h, w = out.size()
        out = out.view(b, c, -1)
        # Compute Gram matrix and normalize by height * width
        out = torch.bmm(out, out.transpose(1, 2)).div(h * w)
        # Flatten the output
        out = out.view(out.size(0), -1)
        return self.fc(out)

class MulLayer(nn.Module):
    def __init__(self, layer, matrixSize=32):
        super(MulLayer, self).__init__()
        self.snet = CNN(layer, matrixSize)
        self.cnet = CNN(layer, matrixSize)
        self.matrixSize = matrixSize

        # Compression and decompression layers to adjust feature map channels
        if layer == 'r11':
            self.compress = nn.Conv2d(64, matrixSize, 1, 1, 0)
            self.unzip = nn.Conv2d(matrixSize, 64, 1, 1, 0)
        elif layer == 'r21':
            self.compress = nn.Conv2d(128, matrixSize, 1, 1, 0)
            self.unzip = nn.Conv2d(matrixSize, 128, 1, 1, 0)
        elif layer == 'r31':
            self.compress = nn.Conv2d(256, matrixSize, 1, 1, 0)
            self.unzip = nn.Conv2d(matrixSize, 256, 1, 1, 0)
        elif layer == 'r41':
            self.compress = nn.Conv2d(512, matrixSize, 1, 1, 0)
            self.unzip = nn.Conv2d(matrixSize, 512, 1, 1, 0)
        else:
            raise ValueError(f"Unsupported layer: {layer}. Must be one of ['r11', 'r21', 'r31', 'r41']")
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

        # Add debug prints to check dimensions
        #print(f"Content feature dimensions: {cF.size()}")
        #print(f"Style feature dimensions: {sF.size()}")
        
        # Compress content feature map to match the matrix size
        compress_content = self.compress(cF)
        #print(f"Compressed content dimensions: {compress_content.size()}")
        
        b, c, h, w = compress_content.size()
        compress_content = compress_content.view(b, c, -1)

        if trans:
            # Compute transformation matrices for content and style
            cMatrix = self.cnet(cF)  # Make sure cF has correct channel count
            sMatrix = self.snet(sF)  # Make sure sF has correct channel count
            
            #print(f"Content matrix dimensions: {cMatrix.size()}")
            #print(f"Style matrix dimensions: {sMatrix.size()}")
            
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