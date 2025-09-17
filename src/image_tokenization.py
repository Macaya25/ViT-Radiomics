from torch import from_numpy
from torch.nn import Conv2d
import numpy as np


def convolution_3_layers(image):

    if isinstance(image, np.ndarray):
        image = from_numpy(image).float()
        image = image.unsqueeze(0).unsqueeze(0)

    conv_layer1 = Conv2d(in_channels=1, out_channels=64, kernel_size=3)
    conv_layer2 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)
    conv_layer3 = Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2)

    output = conv_layer1(image)
    output = conv_layer2(output)
    output = conv_layer3(output)

    return output
    