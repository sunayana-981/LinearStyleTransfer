import os
import torch
import random
from PIL import Image
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

def default_loader(path, fineSize):
    # Normalize path separators
    path = os.path.normpath(path)
    
    # Check if file exists
    if not os.path.exists(path):
        print(f"Warning: File does not exist: {path}")
        # Try to find any image in the directory
        dir_path = os.path.dirname(path)
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                alt_path = os.path.join(dir_path, files[0])
                print(f"Using alternative file: {alt_path}")
                path = alt_path
            else:
                print(f"No image files found in directory: {dir_path}")
                # Create a blank image as fallback
                img = Image.new('RGB', (fineSize, fineSize), color='gray')
                return img
        else:
            print(f"Directory does not exist: {dir_path}")
            # Create a blank image as fallback
            img = Image.new('RGB', (fineSize, fineSize), color='gray')
            return img
    
    try:
        img = Image.open(path).convert('RGB')
        
        # Resize if needed
        if fineSize:
            img = transforms.Resize(fineSize)(img)
            img = transforms.CenterCrop(fineSize)(img)
            
        return img
    except Exception as e:
        print(f"Error loading image {path}: {str(e)}")
        # Create a blank image as fallback
        img = Image.new('RGB', (fineSize, fineSize), color='gray')
        return img

class Dataset(data.Dataset):
    def __init__(self, contentPath, stylePath, contentSegPath, styleSegPath, fineSize=512):
        super(Dataset, self).__init__()
        self.contentPath = contentPath
        self.stylePath = stylePath
        self.contentSegPath = contentSegPath
        self.styleSegPath = styleSegPath
        self.fineSize = fineSize
        
        # Normalize paths
        self.contentPath = os.path.normpath(self.contentPath)
        self.stylePath = os.path.normpath(self.stylePath)
        self.contentSegPath = os.path.normpath(self.contentSegPath)
        self.styleSegPath = os.path.normpath(self.styleSegPath)
        
        # Get content files
        self.contentFiles = []
        if os.path.exists(self.contentPath):
            if os.path.isdir(self.contentPath):
                self.contentFiles = [f for f in os.listdir(self.contentPath) 
                                     if f.endswith(('.png', '.jpg', '.jpeg'))]
            else:
                # If it's a single file
                self.contentFiles = [os.path.basename(self.contentPath)]
                self.contentPath = os.path.dirname(self.contentPath)
        
        # Get style files
        self.styleFiles = []
        if os.path.exists(self.stylePath):
            if os.path.isdir(self.stylePath):
                self.styleFiles = [f for f in os.listdir(self.stylePath) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))]
            else:
                # If it's a single file
                self.styleFiles = [os.path.basename(self.stylePath)]
                self.stylePath = os.path.dirname(self.stylePath)
        
        # Print debug info
        print(f"Content path: {self.contentPath}, found {len(self.contentFiles)} files")
        print(f"Style path: {self.stylePath}, found {len(self.styleFiles)} files")
        
        if not self.contentFiles:
            print(f"Warning: No content files found in {self.contentPath}")
        if not self.styleFiles:
            print(f"Warning: No style files found in {self.stylePath}")
        
        # If either is empty, create dummy data
        if not self.contentFiles:
            self.contentFiles = ['dummy.png']
        if not self.styleFiles:
            self.styleFiles = ['dummy.png']
            
        # Transform for normalization
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        # Get random content and style files
        contentImgName = self.contentFiles[index % len(self.contentFiles)]
        styleImgName = random.choice(self.styleFiles)
        
        # Full paths
        contentImgPath = os.path.join(self.contentPath, contentImgName)
        styleImgPath = os.path.join(self.stylePath, styleImgName)
        
        # Create mask paths
        contentMaskPath = None
        if self.contentSegPath and os.path.exists(self.contentSegPath):
            contentMaskPath = os.path.join(self.contentSegPath, contentImgName)
            # Try different extensions if necessary
            if not os.path.exists(contentMaskPath):
                base = os.path.splitext(contentImgName)[0]
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = os.path.join(self.contentSegPath, base + ext)
                    if os.path.exists(test_path):
                        contentMaskPath = test_path
                        break
        
        styleMaskPath = None
        if self.styleSegPath and os.path.exists(self.styleSegPath):
            styleMaskPath = os.path.join(self.styleSegPath, styleImgName)
            # Try different extensions if necessary
            if not os.path.exists(styleMaskPath):
                base = os.path.splitext(styleImgName)[0]
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = os.path.join(self.styleSegPath, base + ext)
                    if os.path.exists(test_path):
                        styleMaskPath = test_path
                        break
        
        # Debug info
        print(f"Loading content: {contentImgPath}")
        print(f"Loading style: {styleImgPath}")
        print(f"Content mask: {contentMaskPath}")
        print(f"Style mask: {styleMaskPath}")
        
        # Load images
        contentImg = default_loader(contentImgPath, self.fineSize)
        styleImg = default_loader(styleImgPath, self.fineSize)
        
        # Create white noise image for SPN
        whitenImg = contentImg.copy()
        
        # Default masks (all ones) if no mask files
        cmasks = torch.ones(1, self.fineSize, self.fineSize)
        smasks = torch.ones(1, self.fineSize, self.fineSize)
        
        # Load masks if available
        if contentMaskPath and os.path.exists(contentMaskPath):
            try:
                cmask = Image.open(contentMaskPath).convert('L')
                cmask = transforms.Resize(self.fineSize)(cmask)
                cmask = transforms.CenterCrop(self.fineSize)(cmask)
                cmasks = transforms.ToTensor()(cmask)
                # Binarize the mask if needed
                cmasks = (cmasks > 0.5).float()
            except Exception as e:
                print(f"Error loading content mask {contentMaskPath}: {str(e)}")
        
        if styleMaskPath and os.path.exists(styleMaskPath):
            try:
                smask = Image.open(styleMaskPath).convert('L')
                smask = transforms.Resize(self.fineSize)(smask)
                smask = transforms.CenterCrop(self.fineSize)(smask)
                smasks = transforms.ToTensor()(smask)
                # Binarize the mask if needed
                smasks = (smasks > 0.5).float()
            except Exception as e:
                print(f"Error loading style mask {styleMaskPath}: {str(e)}")
        
        # Transform images to tensors
        contentImg = self.normalize(contentImg)
        styleImg = self.normalize(styleImg)
        whitenImg = self.normalize(whitenImg)
        
        return contentImg, styleImg, whitenImg, cmasks, smasks, contentImgName

    def __len__(self):
        return len(self.contentFiles)