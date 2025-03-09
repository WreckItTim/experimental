from torch import nn

def V1():
    return nn.Sequential(
        nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1), 
            nn.GroupNorm(num_groups=32, num_channels=256), 
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        ),
        nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    )

# This is the new model
def ResNet152(bottleneck_channel=512):
    return nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=64),
                nn.SELU(inplace=True),
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=32, num_channels=64),
                nn.SELU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(64, 512, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=512),
                nn.SELU(inplace=True),
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=32, num_channels=512),
                nn.SELU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, bottleneck_channel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=bottleneck_channel, num_channels=bottleneck_channel),
                nn.SELU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(bottleneck_channel, 512, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=512),
                nn.SELU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=256),
                nn.SELU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=128),
                nn.SELU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=64),
                nn.SELU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=32),
                nn.SELU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
                nn.SELU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False),
                nn.Sigmoid()
            ),
        )

# CUSTOM SLIM LAYERS
from torch import nn, Tensor
import torch
class SELU_Hack(nn.SELU):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace)

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input)[:,:,:,:(input.shape[3]-1)]
    
class Conv2d_Hack(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(torch.transpose(input, 2, 3))
    
class Sigmoid_Hack(nn.Sigmoid):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(torch.transpose(input, 2, 3))
    
def IanV1_parent():
    return nn.Sequential(
        ############################################ Depth prediction network
        nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1), 
            nn.GroupNorm(num_groups=32, num_channels=256), 
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        ),
    )
    
def IanV1_student(bn_channels):
    return nn.Sequential(
        ############################################ Depth prediction network
        nn.Sequential(
            Conv2d_Hack(3, 32, 4, stride=3),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=3),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(64, bn_channels, 4, stride=3),
            nn.GroupNorm(num_groups=bn_channels, num_channels=bn_channels),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.ConvTranspose2d(bn_channels, 64, kernel_size=4, stride=2, padding=(2,0)),
            nn.GroupNorm(num_groups=32, num_channels=64),
            SELU_Hack(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=64, num_channels=128),
            nn.SELU(inplace=True),

            nn.Conv2d(128, 256, 3, padding=2, dilation=2), 
            nn.GroupNorm(num_groups=32, num_channels=256), 
            nn.SELU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        ),

        nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            Sigmoid_Hack()
        ),
    )

def get_head_tail(model_name, split_point, compression):
    if model_name == 'V1':
        if split_point == 0:
            head_block = nn.Sequential(
                nn.Conv2d(3, compression, 4, stride=2, padding=1),
                nn.BatchNorm2d(compression),
                nn.SELU(inplace=True),
            )
            tail_block = nn.Sequential(
                nn.Conv2d(compression, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.SELU(inplace=True),
            )
        if split_point == 1:
            head_block = nn.Sequential(
                nn.Conv2d(32, compression, 4, stride=2, padding=1),
                nn.BatchNorm2d(compression),
                nn.SELU(inplace=True)
            )
            tail_block = nn.Sequential(
                nn.Conv2d(compression, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128), 
                nn.SELU(inplace=True)
            )
        if split_point == 2:
            head_block = nn.Sequential(
                nn.Conv2d(64, compression, 4, stride=2, padding=1),
                nn.BatchNorm2d(compression),
                nn.SELU(inplace=True)
            )
            tail_block = nn.Sequential(
                nn.Conv2d(compression, 256, 4, stride=2, padding=1), 
                nn.BatchNorm2d(256), 
                nn.SELU(inplace=True)
            )
    if model_name == 'ResNet152':
        if split_point == 4:
            head_block = nn.Sequential(
                nn.Conv2d(512, compression, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=compression, num_channels=compression),
                nn.SELU(inplace=True),
            )
            tail_block = nn.Sequential(
                nn.ConvTranspose2d(compression, 512, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=512),
                nn.SELU(inplace=True),
            )
    return head_block, tail_block