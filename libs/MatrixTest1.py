import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.autograd import Variable
import torchvision.utils as vutils

# Safely access masks with index checking
def safe_mask_access(masks, index, default_size=(512, 512)):
    if masks is None:
        # Return a default mask filled with ones
        return torch.ones(1, default_size[0], default_size[1])
    
    # Handle tensor vs list
    if isinstance(masks, list):
        if index >= len(masks) or index < 0:
            # Return first mask or create a default one
            if len(masks) > 0:
                return masks[0].clone()
            else:
                return torch.ones(1, default_size[0], default_size[1])
        return masks[index].clone()
    else:  # Tensor
        if index >= masks.size(0) or index < 0:
            # Return first mask or create a default one
            if masks.size(0) > 0:
                return masks[0].clone()
            else:
                return torch.ones(1, default_size[0], default_size[1])
        return masks[index].clone()

# Safe resize function that handles PyTorch tensors properly
def safe_resize(tensor, size, interpolation=cv2.INTER_NEAREST):
    # Convert tensor to numpy for OpenCV
    if tensor.is_cuda:
        tensor_np = tensor.cpu().detach().numpy()
    else:
        tensor_np = tensor.detach().numpy()
    
    # Ensure mask is 2D for cv2.resize
    if tensor_np.ndim > 2:
        tensor_np = tensor_np.squeeze()
    
    # Resize the mask
    resized = cv2.resize(tensor_np, (size[1], size[0]), interpolation=interpolation)
    
    # Convert back to tensor
    return torch.FloatTensor(resized)

class CNN(nn.Module):
    def __init__(self, layer, matrixSize=32):
        super(CNN, self).__init__()
        # Define convolutional layers based on the input layer name
        if layer == 'r11':
            # For layer r11: 64 channels
            self.convs = nn.Sequential(
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, matrixSize, 3, 1, 1)
            )
        elif layer == 'r21':
            # For layer r21: 128 channels
            self.convs = nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, matrixSize, 3, 1, 1)
            )
        elif layer == 'r31':
            # For layer r31: 256 channels
            self.convs = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, matrixSize, 3, 1, 1)
            )
        elif layer == 'r41':
            # For layer r41: 512 channels
            self.convs = nn.Sequential(
                nn.Conv2d(512, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, matrixSize, 3, 1, 1)
            )
        elif layer == 'r51':
            # For layer r51: 512 channels
            self.convs = nn.Sequential(
                nn.Conv2d(512, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, matrixSize, 3, 1, 1)
            )
        # Fully connected layer
        self.fc = nn.Linear(matrixSize * matrixSize, matrixSize * matrixSize)

    def forward(self, x, masks, style=False):
        color_code_number = 9
        xb, xc, xh, xw = x.size()
        device = x.device
        
        # Create feature_sub_mean on the same device as x
        x_view = x.view(xc, -1)
        feature_sub_mean = x_view.clone()
        
        # Number of masks available
        num_masks = masks.size(0) if isinstance(masks, torch.Tensor) else len(masks)
        
        for i in range(min(color_code_number, num_masks)):
            try:
                # Safely get mask with bounds checking
                mask = safe_mask_access(masks, i, (xh, xw))
                if mask.dim() > 2:
                    mask = mask.squeeze(0)
                
                # Use PyTorch's F.interpolate instead of cv2.resize
                if mask.size(0) != xh or mask.size(1) != xw:
                    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), 
                                        size=(xh, xw), 
                                        mode='nearest').squeeze(0).squeeze(0)
                
                # Convert to long for indexing and move to same device as x
                mask = mask.long().to(device)
                
                if torch.sum(mask) >= 10:
                    mask = mask.view(-1)
                    
                    # Get non-zero indices
                    fgmask = (mask > 0).nonzero().squeeze(1)
                    if fgmask.numel() > 0:  # Check if mask contains any foreground pixels
                        # Select features
                        selectFeature = torch.index_select(x_view, 1, fgmask)
                        
                        # Calculate and subtract mean
                        f_mean = torch.mean(selectFeature, 1, keepdim=True)
                        f_mean = f_mean.expand_as(selectFeature)
                        selectFeature = selectFeature - f_mean
                        
                        # Update feature_sub_mean
                        feature_sub_mean.index_copy_(1, fgmask, selectFeature)
            except Exception as e:
                print(f"Error processing mask {i}: {str(e)}")
                continue

        # Reshape and apply convolutions
        feature = self.convs(feature_sub_mean.view(xb, xc, xh, xw))
        b, c, h, w = feature.size()
        transMatrices = {}
        feature_view = feature.view(c, -1)

        for i in range(min(color_code_number, num_masks)):
            try:
                # Safely get mask with bounds checking
                mask = safe_mask_access(masks, i, (h, w))
                if mask.dim() > 2:
                    mask = mask.squeeze(0)
                
                # Use PyTorch's F.interpolate instead of cv2.resize
                if mask.size(0) != h or mask.size(1) != w:
                    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), 
                                        size=(h, w), 
                                        mode='nearest').squeeze(0).squeeze(0)
                
                # Convert to long for indexing and move to same device
                mask = mask.long().to(device)
                
                if torch.sum(mask) >= 10:
                    mask = mask.view(-1)
                    
                    # Get indices where mask equals 1
                    fgmask = (mask == 1).nonzero().squeeze(1)
                    if fgmask.numel() > 0:  # Check if mask contains any pixels with value 1
                        # Select features
                        selectFeature = torch.index_select(feature_view, 1, fgmask)
                        tc, tN = selectFeature.size()
                        
                        if tN > 0:  # Check if we selected any features
                            # Calculate covariance matrix
                            covMatrix = torch.mm(selectFeature, selectFeature.transpose(0, 1)).div(tN)
                            
                            # Apply fully connected layer to get transformation matrix
                            transmatrix = self.fc(covMatrix.view(-1))
                            transMatrices[i] = transmatrix
            except Exception as e:
                print(f"Error processing mask {i} for transformation: {str(e)}")
                continue
                
        return transMatrices, feature_sub_mean


class MulLayer(nn.Module):
    def __init__(self, layer, matrixSize=32):
        super(MulLayer, self).__init__()
        self.snet = CNN(layer)
        self.cnet = CNN(layer)
        self.matrixSize = matrixSize

        if layer == 'r41':
            self.compress = nn.Conv2d(512, matrixSize, 1, 1, 0)
            self.unzip = nn.Conv2d(matrixSize, 512, 1, 1, 0)
        elif layer == 'r31':
            self.compress = nn.Conv2d(256, matrixSize, 1, 1, 0)
            self.unzip = nn.Conv2d(matrixSize, 256, 1, 1, 0)
        elif layer == 'r51':
            self.compress = nn.Conv2d(512, matrixSize, 1, 1, 0)
            self.unzip = nn.Conv2d(matrixSize, 512, 1, 1, 0)
        elif layer == 'r21':
            self.compress = nn.Conv2d(128, matrixSize, 1, 1, 0)
            self.unzip = nn.Conv2d(matrixSize, 128, 1, 1, 0)
        elif layer == 'r11':
            self.compress = nn.Conv2d(64, matrixSize, 1, 1, 0)
            self.unzip = nn.Conv2d(matrixSize, 64, 1, 1, 0)

    def forward(self, cF, sF, cmasks=None, smasks=None):
        device = cF.device
        
        # Handle case where masks are None
        if cmasks is None:
            cmasks = torch.ones(1, 1, cF.size(2), cF.size(3)).to(device)
        if smasks is None:
            smasks = torch.ones(1, 1, sF.size(2), sF.size(3)).to(device)
            
        # Ensure masks are on the correct device
        if isinstance(cmasks, torch.Tensor) and cmasks.device != device:
            cmasks = cmasks.to(device)
        if isinstance(smasks, torch.Tensor) and smasks.device != device:
            smasks = smasks.to(device)

        # Get feature dimensions
        sb, sc, sh, sw = sF.size()
        cb, cc, ch, cw = cF.size()

        # Forward through style and content networks
        sMatrices, sF_sub_mean = self.snet(sF, smasks, style=True)
        cMatrices, cF_sub_mean = self.cnet(cF, cmasks, style=False)

        # Compress content features
        compress_content = self.compress(cF_sub_mean.view(cF.size()))
        cb, cc, ch, cw = compress_content.size()
        compress_content_view = compress_content.view(cc, -1)
        transfeature = compress_content_view.clone()
        
        # Number of color codes and masks
        color_code_number = 9
        num_cmasks = cmasks.size(0) if isinstance(cmasks, torch.Tensor) else len(cmasks)
        num_smasks = smasks.size(0) if isinstance(smasks, torch.Tensor) else len(smasks)
        
        # Initialize finalSMean
        finalSMean = torch.zeros(cF.size()).to(device)
        finalSMean_view = finalSMean.view(sc, -1)
        
        for i in range(min(color_code_number, num_cmasks, num_smasks)):
            try:
                # Safely get content and style masks with bounds checking
                cmask = safe_mask_access(cmasks, i, (ch, cw))
                smask = safe_mask_access(smasks, i, (sh, sw))
                
                if cmask.dim() > 2:
                    cmask = cmask.squeeze(0)
                if smask.dim() > 2:
                    smask = smask.squeeze(0)
                
                # Use PyTorch's F.interpolate instead of cv2.resize
                if cmask.size(0) != ch or cmask.size(1) != cw:
                    cmask = F.interpolate(cmask.unsqueeze(0).unsqueeze(0).float(), 
                                         size=(ch, cw), 
                                         mode='nearest').squeeze(0).squeeze(0)
                if smask.size(0) != sh or smask.size(1) != sw:
                    smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), 
                                         size=(sh, sw), 
                                         mode='nearest').squeeze(0).squeeze(0)
                
                # Convert to long for indexing and move to same device
                cmask = cmask.long().to(device)
                smask = smask.long().to(device)
                
                # Check if we have enough mask pixels and required matrices
                if (torch.sum(cmask) >= 10 and torch.sum(smask) >= 10 and 
                    (i in sMatrices) and (i in cMatrices)):
                    
                    # Flatten masks
                    cmask_flat = cmask.view(-1)
                    smask_flat = smask.view(-1)
                    
                    # Get mask indices
                    fgcmask = (cmask_flat == 1).nonzero().squeeze(1)
                    fgsmask = (smask_flat == 1).nonzero().squeeze(1)
                    
                    if fgcmask.numel() > 0 and fgsmask.numel() > 0:  # Check if we have non-empty masks
                        # Select style features and compute mean
                        sFF = sF.view(sc, -1)
                        sFF_select = torch.index_select(sFF, 1, fgsmask)
                        sMean = torch.mean(sFF_select, dim=1, keepdim=True)
                        sMean = sMean.view(1, sc, 1, 1)
                        sMean = sMean.expand_as(cF)
                        
                        # Get matrices
                        sMatrix = sMatrices[i]
                        cMatrix = cMatrices[i]
                        
                        # Reshape matrices
                        sMatrix = sMatrix.view(self.matrixSize, self.matrixSize)
                        cMatrix = cMatrix.view(self.matrixSize, self.matrixSize)
                        
                        # Calculate transformation matrix
                        transmatrix = torch.mm(sMatrix, cMatrix)
                        
                        # Select and transform content features
                        compress_content_select = torch.index_select(compress_content_view, 1, fgcmask)
                        transfeatureFG = torch.mm(transmatrix, compress_content_select)
                        transfeature.index_copy_(1, fgcmask, transfeatureFG)
                        
                        # Update finalSMean
                        sMean_view = sMean.contiguous().view(sc, -1)
                        sMean_select = torch.index_select(sMean_view, 1, fgcmask)
                        finalSMean_view.index_copy_(1, fgcmask, sMean_select)
            except Exception as e:
                print(f"Error processing mask {i} for transformation: {str(e)}")
                continue
        
        # Reshape and unzip the transformed features
        out = self.unzip(transfeature.view(cb, cc, ch, cw))
        return out + finalSMean_view.view(out.size())