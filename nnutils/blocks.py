# TODO: Implement basic blocks which can be re used for different models that can be used.
#  eg: Conv block, Convtrans block, resnet block, you can refer the original repo

import torch.nn as nn
import torchvision


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def double_conv(in_planes, out_planes, mid_planes=None, batch_norm=False):
    """double convolution layers and keep dimensions"""
    if batch_norm is False:
        if mid_planes is None:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_planes, out_planes, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, mid_planes, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_planes, out_planes, 3, padding=1),
                nn.ReLU(inplace=True)
            )
    else:
        if mid_planes is None:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 3, padding=1),
                nn.BatchNorm2d(num_features=out_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_planes, out_planes, 3, padding=1),
                nn.BatchNorm2d(num_features=out_planes),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, mid_planes, 3, padding=1),
                nn.BatchNorm2d(num_features=mid_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_planes, out_planes, 3, padding=1),
                nn.BatchNorm2d(num_features=out_planes),
                nn.ReLU(inplace=True)
            )


class ResNetConv(nn.Module):
    """resnet18 architecture with n blocks"""

    def __init__(self, n_blocks=4):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.n_blocks = n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)

        return x


def make_conv(n_in, n_out, n_blocks, kernel=3, normalization=nn.BatchNorm3d, activation=nn.ReLU):
    blocks = []
    for i in range(n_blocks):                                                                                                                                                                                                                                                                                             
        in1 = n_in if i == 0 else n_out
        blocks.append(nn.Sequential(
            nn.Conv3d(in1, n_out, kernel_size=kernel, stride=1, padding=(kernel//2)),
            normalization(n_out),
            activation(inplace=True)
        ))
    return nn.Sequential(*blocks)


def make_conv_2d(n_in, n_out, n_blocks, kernel=3, normalization=nn.BatchNorm2d, activation=nn.ReLU):
    blocks = []
    for i in range(n_blocks):                                                                                                                                                                                                                                                                                             
        in1 = n_in if i == 0 else n_out
        blocks.append(nn.Sequential(
            nn.Conv2d(in1, n_out, kernel_size=kernel, stride=1, padding=(kernel//2)),
            normalization(n_out),
            activation(inplace=True)
        ))
    return nn.Sequential(*blocks)


def make_downscale(n_in, n_out, kernel=4, normalization=nn.BatchNorm3d, activation=nn.ReLU):                                                                                                                                                                                                                              
    block = nn.Sequential(
        nn.Conv3d(n_in, n_out, kernel_size=kernel, stride=2, padding=(kernel-2)//2),
        normalization(n_out),
        activation(inplace=True)
    )
    return block


def make_downscale_2d(n_in, n_out, kernel=4, normalization=nn.BatchNorm2d, activation=nn.ReLU):                                                                                                                                                                                                                              
    block = nn.Sequential(
        nn.Conv2d(n_in, n_out, kernel_size=kernel, stride=2, padding=(kernel-2)//2),
        normalization(n_out),
        activation(inplace=True)
    )
    return block
    

def make_upscale(n_in, n_out, normalization=nn.BatchNorm3d, activation=nn.ReLU):
    block = nn.Sequential(
        nn.ConvTranspose3d(n_in, n_out, kernel_size=6, stride=2, padding=2),
        normalization(n_out),
        activation(inplace=True)
    )
    return block


def make_upscale_2d(n_in, n_out, kernel=4, normalization=nn.BatchNorm2d, activation=nn.ReLU):
    block = nn.Sequential(
        nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel, stride=2, padding=(kernel-2)//2),
        normalization(n_out),
        activation(inplace=True)
    )
    return block


class ResBlock(nn.Module):
    def __init__(self, n_out, kernel=3, normalization=nn.BatchNorm3d, activation=nn.ReLU):
        super().__init__()
        self.block0 = nn.Sequential(
            nn.Conv3d(n_out, n_out, kernel_size=kernel, stride=1, padding=(kernel//2)),
            normalization(n_out),
            activation(inplace=True)
        )
        
        self.block1 = nn.Sequential(
            nn.Conv3d(n_out, n_out, kernel_size=kernel, stride=1, padding=(kernel//2)),
            normalization(n_out),
        )

        self.block2 = nn.ReLU()

    def forward(self, x0):
        x = self.block0(x0)

        x = self.block1(x)
        
        x = self.block2(x + x0)
        return x


class ResBlock2d(nn.Module):
    def __init__(self, n_out, kernel=3, normalization=nn.BatchNorm2d, activation=nn.ReLU):
        super().__init__()
        self.block0 = nn.Sequential(
            nn.Conv2d(n_out, n_out, kernel_size=kernel, stride=1, padding=(kernel//2)),
            normalization(n_out),
            activation(inplace=True)
        )
        
        self.block1 = nn.Sequential(
            nn.Conv2d(n_out, n_out, kernel_size=kernel, stride=1, padding=(kernel//2)),
            normalization(n_out),
        )

        self.block2 = nn.ReLU()

    def forward(self, x0):
        x = self.block0(x0)

        x = self.block1(x)
        
        x = self.block2(x + x0)
        return x


def gaussian3d(l, sigma=1.0):                                                                                                                                                                                                                                                                                             
    """
    Creates gaussian kernel with side length l and a sigma of sigma.
    """

    ax = np.arange(-l//2 + 1.0, l//2 + 1.0)
    xx, yy, zz = np.meshgrid(ax, ax, ax)

    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2.0*sigma**2))

    return np.asarray(kernel, dtype=np.float32)


def unnorm_gaussian2d(l, sigma=1.0):                                                                                                                                                                                                                                                                                             
    """
    Creates an unnormalized gaussian kernel with side length l and a sigma of sigma.
    """

    ax = np.arange(-l//2 + 1.0, l//2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2)/(2.0*sigma**2))

    return np.asarray(kernel, dtype=np.float32)


def gaussian2d(l, sigma=1.0):                                                                                                                                                                                                                                                                                             
    """
    Creates gaussian kernel with side length l and a sigma of sigma.
    """

    ax = np.arange(-l//2 + 1.0, l//2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)

    kernel = (1.0 / math.sqrt(2.0 * math.pi * sigma**2)) * np.exp(-(xx**2 + yy**2)/(2.0*sigma**2))

    return np.asarray(kernel, dtype=np.float32)


def gradient_norm2d(x):
    """
    Computes gradient norm of 2d image.
    4d torch vector is excepted as input, with shape (batch, dim, height, width).
    """
    assert len(x.shape) == 4

    a = torch.Tensor([[1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]])

    a = a.view((1,1,3,3)).to(x.device)
    G_x = nn.functional.conv2d(x, a, padding=1)

    b = torch.Tensor([[1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]])

    b = b.view((1,1,3,3)).to(x.device)
    G_y = nn.functional.conv2d(x, b, padding=1)

    G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
    
    return G