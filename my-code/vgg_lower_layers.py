import torchvision.models as models
import torch

# Load the VGG-19 pre-trained model
vgg19 = models.vgg19(pretrained=True)
vgg_features = vgg19.features

# Extract weights for the layers
r11_weights = vgg_features[0].state_dict()  # conv1_1
r21_weights = vgg_features[5].state_dict()  # conv2_1

# Save these weights
torch.save(r11_weights, "models/vgg_r11.pth")
torch.save(r21_weights, "models/vgg_r21.pth")
