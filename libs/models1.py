import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, layer):
        super(Encoder, self).__init__()
        # Dynamic setup based on the chosen layer
        layer_config = {
            'r11': (3, 64, 224),
            'r21': (64, 128, 112),
            'r31': (128, 256, 56),
            'r41': (256, 512, 28),
            'r51': (512, 512, 28)
        }
        in_channels, out_channels, _ = layer_config.get(layer, (256, 128, 56))
        
        self.model = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels, out_channels, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, layer):
        super(Decoder, self).__init__()
        # Dynamic setup based on the chosen layer
        layer_config = {
            'r11': (64, 3, 224),
            'r21': (128, 64, 112),
            'r31': (256, 128, 56),
            'r41': (512, 256, 28),
            'r51': (512, 256, 28)
        }
        in_channels, out_channels, _ = layer_config.get(layer, (256, 128, 56))
        
        self.model = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels, out_channels, 3, 1, 0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)