import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

# Helper function to ensure tensors have matching sizes before addition
def matched_addition(tensor1, tensor2):
    """
    Resize tensor2 to match the dimensions of tensor1 if needed.
    """
    if tensor1.size() != tensor2.size():
        # Get target size from tensor1
        target_size = tensor1.size()[2:]  # Skip batch and channel dimensions
        
        # Resize tensor2 to match tensor1
        tensor2_resized = F.interpolate(tensor2, size=target_size, mode='bilinear', align_corners=False)
        return tensor1 + tensor2_resized
    else:
        return tensor1 + tensor2

class VGG(nn.Module):
    def __init__(self, nf):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(3, nf, 3, padding=1)
        # 256 x 256
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(nf, nf*2, 3, padding=1)
        # 128 x 128
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(nf*2, nf*4, 3, padding=1)
        # 64 x 64
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 32 x 32
        self.conv4 = nn.Conv2d(nf*4, nf*8, 3, padding=1)

    def forward(self, x):
        output = {}
        output['conv1'] = self.conv1(x)
        x = F.relu(output['conv1'])
        x = self.pool1(x)
        output['conv2'] = self.conv2(x)
        # 128 x 128
        x = F.relu(output['conv2'])
        x = self.pool2(x)
        output['conv3'] = self.conv3(x)
        # 64 x 64
        x = F.relu(output['conv3'])
        output['pool3'] = self.pool3(x)
        # 32 x 32
        output['conv4'] = self.conv4(output['pool3'])
        return output

class Decoder(nn.Module):
    def __init__(self, nf=32, spn=1):
        super(Decoder, self).__init__()
        # 32 x 32
        self.layer0 = nn.Conv2d(nf*8, nf*4, 1, 1, 0)  # edge_conv5
        self.layer1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.layer2 = nn.Sequential(nn.Conv2d(nf*4, nf*4, 3, 1, 1),  # edge_conv8
                                   nn.ELU(inplace=True))
        # 64 x 64
        self.layer3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.layer4 = nn.Sequential(nn.Conv2d(nf*4, nf*2, 3, 1, 1),  # edge_conv8
                                   nn.ELU(inplace=True))
        # 128 x 128
        self.layer5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.layer6 = nn.Sequential(nn.Conv2d(nf*2, nf, 3, 1, 1),  # edge_conv8
                                   nn.ELU(inplace=True))
        if spn == 1:
            self.layer7 = nn.Conv2d(nf, nf*12, 3, 1, 1)
        else:
            self.layer7 = nn.Conv2d(nf, nf*24, 3, 1, 1)
        self.spn = spn
        # 256 x 256

    def forward(self, encode_feature):
        output = {}
        output['0'] = self.layer0(encode_feature['conv4'])
        output['1'] = self.layer1(output['0'])

        output['2'] = self.layer2(output['1'])
        # Use matched addition to handle size mismatches
        output['2res'] = matched_addition(output['2'], encode_feature['conv3'])
        # 64 x 64

        output['3'] = self.layer3(output['2res'])
        output['4'] = self.layer4(output['3'])
        # Use matched addition to handle size mismatches
        output['4res'] = matched_addition(output['4'], encode_feature['conv2'])
        # 128 x 128

        output['5'] = self.layer5(output['4res'])
        output['6'] = self.layer6(output['5'])
        # Use matched addition to handle size mismatches
        output['6res'] = matched_addition(output['6'], encode_feature['conv1'])

        output['7'] = self.layer7(output['6res'])

        return output['7']

# Fallback implementation that works entirely on the GPU
class GateRecurrent2dnoind(nn.Module):
    def __init__(self, horizontal, reverse):
        super(GateRecurrent2dnoind, self).__init__()
        self.horizontal = horizontal
        self.reverse = reverse
    
    def forward(self, X, G1, G2, G3):
        # Ensure all tensors are the same size
        _, _, h, w = X.size()
        G1 = F.interpolate(G1, size=(h, w), mode='bilinear', align_corners=False)
        G2 = F.interpolate(G2, size=(h, w), mode='bilinear', align_corners=False)
        G3 = F.interpolate(G3, size=(h, w), mode='bilinear', align_corners=False)
        
        # Simple implementation that mimics the recursive pattern but is fully parallellizable
        # Initialize output as X weighted by G1
        output = G1 * X
        
        # Apply a series of convolutions to simulate the recurrent behavior
        # This is a simplified approximation of the original algorithm
        kernel_size = 3
        padding = 1
        
        # Create a simple recurrent block
        if self.horizontal:
            # Horizontal direction
            kernel = torch.zeros(1, 1, kernel_size, kernel_size, device=X.device)
            if self.reverse:
                # Right to left: use the right neighbor
                kernel[0, 0, 1, 2] = 1.0
            else:
                # Left to right: use the left neighbor
                kernel[0, 0, 1, 0] = 1.0
        else:
            # Vertical direction
            kernel = torch.zeros(1, 1, kernel_size, kernel_size, device=X.device)
            if self.reverse:
                # Bottom to top: use the bottom neighbor
                kernel[0, 0, 2, 1] = 1.0
            else:
                # Top to bottom: use the top neighbor
                kernel[0, 0, 0, 1] = 1.0
                
        # Expand kernel for all input channels
        batch_size, channels, _, _ = X.size()
        kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
        
        # Apply recursive operations
        # This is a simplified approximation - for a more accurate implementation,
        # you'd need to implement the exact recurrent algorithm
        for i in range(max(h, w) // 2):  # Half the image dimension should be enough for propagation
            # Pad properly for convolution
            padded_output = F.pad(output, (padding, padding, padding, padding), mode='replicate')
            
            # Apply the directional kernel
            propagated = F.conv2d(padded_output, kernel, padding=0, groups=channels)
            
            # Weight the propagated values with G2 and add to output
            output = G1 * X + G2 * propagated
            
        return output

class spn_block(nn.Module):
    def __init__(self, horizontal, reverse):
        super(spn_block, self).__init__()
        self.propagator = GateRecurrent2dnoind(horizontal, reverse)

    def forward(self, x, G1, G2, G3):
        # Make sure all tensors are the same size
        _, _, h, w = x.size()
        if G1.size(2) != h or G1.size(3) != w:
            G1 = F.interpolate(G1, size=(h, w), mode='bilinear', align_corners=False)
        if G2.size(2) != h or G2.size(3) != w:
            G2 = F.interpolate(G2, size=(h, w), mode='bilinear', align_corners=False)
        if G3.size(2) != h or G3.size(3) != w:
            G3 = F.interpolate(G3, size=(h, w), mode='bilinear', align_corners=False)
        
        sum_abs = G1.abs() + G2.abs() + G3.abs()
        sum_abs.data[sum_abs.data == 0] = 1e-6
        mask_need_norm = sum_abs.ge(1)
        mask_need_norm = mask_need_norm.float()
        G1_norm = torch.div(G1, sum_abs)
        G2_norm = torch.div(G2, sum_abs)
        G3_norm = torch.div(G3, sum_abs)

        G1 = torch.add(-mask_need_norm, 1) * G1 + mask_need_norm * G1_norm
        G2 = torch.add(-mask_need_norm, 1) * G2 + mask_need_norm * G2_norm
        G3 = torch.add(-mask_need_norm, 1) * G3 + mask_need_norm * G3_norm

        return self.propagator(x, G1, G2, G3)

class SPN(nn.Module):
    def __init__(self, nf=32, spn=1):
        super(SPN, self).__init__()
        # conv for mask
        self.mask_conv = nn.Conv2d(3, nf, 3, 1, 1)

        # guidance network
        self.encoder = VGG(nf)
        self.decoder = Decoder(nf, spn)

        # spn blocks
        self.left_right = spn_block(True, False)
        self.right_left = spn_block(True, True)
        self.top_down = spn_block(False, False)
        self.down_top = spn_block(False, True)

        # post upsample
        self.post = nn.Conv2d(nf, 3, 3, 1, 1)
        self.nf = nf

    def forward(self, x, rgb):
        # Handle different sized inputs - resize if needed
        if x.size() != rgb.size():
            # Get target size from rgb
            _, _, h, w = rgb.size()
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        
        # feature for mask
        X = self.mask_conv(x)

        # guidance
        features = self.encoder(rgb)
        guide = self.decoder(features)

        # Make sure we have enough channels
        if guide.size(1) < self.nf * 12:
            # Pad with zeros if needed
            padding = torch.zeros(guide.size(0), self.nf * 12 - guide.size(1), 
                                guide.size(2), guide.size(3), device=guide.device)
            guide = torch.cat([guide, padding], dim=1)
        
        # Split the channels
        G = torch.split(guide, self.nf, 1)
        
        # Make sure we have enough split tensors
        if len(G) < 12:
            G = list(G)
            while len(G) < 12:
                G.append(torch.zeros_like(G[0]))
            G = tuple(G)
        
        # Apply SPN blocks
        out1 = self.left_right(X, G[0], G[1], G[2])
        out2 = self.right_left(X, G[3], G[4], G[5])
        out3 = self.top_down(X, G[6], G[7], G[8])
        out4 = self.down_top(X, G[9], G[10], G[11])

        # Combine outputs
        out = torch.max(out1, out2)
        out = torch.max(out, out3)
        out = torch.max(out, out4)

        return self.post(out)

if __name__ == '__main__':
    spn = SPN()
    spn = spn.cuda()
    for i in range(100):
        x = Variable(torch.Tensor(1,3,256,256)).cuda()
        rgb = Variable(torch.Tensor(1,3,256,256)).cuda()
        output = spn(x,rgb)
        print(output.size())