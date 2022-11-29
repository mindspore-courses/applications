"""Modules dense"""

import torch
import torch.nn as nn


__all__ = ['PVCNN', 'DConv3d', 'ConvGRU']


class BasicConvolutionBlock(nn.Module):
    """BasicConvolutionBlock"""
    def __init__(self, inc, outc, ks=3, stride=1, padding=0, dilation=1):
        """init"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(inc,
                      outc,
                      kernel_size=ks,
                      padding=padding,
                      dilation=dilation,
                      stride=stride), nn.BatchNorm3d(outc),
            nn.ReLU(True))

    def forward(self, x):
        """forward"""
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    """BasicDeconvolutionBlock"""
    def __init__(self, inc, outc, ks=3, stride=1, padding=0):
        """init"""
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(inc,
                               outc,
                               kernel_size=ks,
                               stride=stride,
                               padding=padding), nn.BatchNorm3d(outc),
            nn.ReLU(True))

    def forward(self, x):
        """forward"""
        return self.net(x)


class ResidualBlock(nn.Module):
    """ResidualBlock"""
    def __init__(self, inc, outc, ks=3, stride=1, padding=1, dilation=1):
        """init"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(inc,
                      outc,
                      kernel_size=ks,
                      padding=padding,
                      dilation=dilation,
                      stride=stride), nn.BatchNorm3d(outc),
            nn.ReLU(True),
            nn.Conv3d(outc,
                      outc,
                      kernel_size=ks,
                      padding=padding,
                      dilation=dilation,
                      stride=stride), nn.BatchNorm3d(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                nn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                nn.BatchNorm3d(outc)
            )

        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class PVCNN(nn.Module):
    """PVCNN"""
    def __init__(self, **kwargs):
        """init"""
        super().__init__()

        self.dropout = kwargs['dropout']

        cr = kwargs.get('cr', 1.0)
        cs = [32, 64, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.stem = nn.Sequential(
            nn.Conv3d(kwargs['in_channels'], cs[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(cs[0]), nn.ReLU(True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[2], cs[3], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[3] + cs[1], cs[3], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[3], cs[4], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[4] + cs[0], cs[4], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
            )
        ])

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[2]),
                nn.BatchNorm1d(cs[2]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[2], cs[4]),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(True),
            )
        ])

        self.weight_initialization()

        if self.dropout:
            self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        """weight_initialization"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        """forward"""
        # x: DenseTensor z: PointTensor
        # x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = z
        x0 = self.stem(x0)
        # z0 = voxel_to_point(x0, z, nearest=False)
        # z0.F = z0.F

        # x1 = point_to_voxel(x0, z0)
        x1 = x0
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        # z1 = voxel_to_point(x2, z0)
        # z1.F = z1.F + self.point_transforms[0](z0.F)

        # y3 = point_to_voxel(x2, z1)
        y3 = x2
        if self.dropout:
            y3 = self.dropout(y3)
        y3 = self.up1[0](y3)
        y3 = torch.cat([y3, x1], dim=1)
        y3 = self.up1[1](y3)

        y4 = self.up2[0](y3)
        y4 = torch.cat([y4, x0], dim=1)
        y4 = self.up2[1](y4)
        # z3 = voxel_to_point(y4, z1)
        # z3.F = z3.F + self.point_transforms[1](z1.F)

        return y4


class DConv3d(nn.Module):
    """DConv3d"""
    def __init__(self, inc, outc, pres, vres, ks=3, stride=1, dilation=1, padding=1):
        """init"""
        super().__init__()
        self.net = nn.Conv3d(inc,
                             outc,
                             kernel_size=ks,
                             dilation=dilation,
                             stride=stride,
                             padding=padding)
        self.point_transforms = nn.Sequential(
            nn.Linear(inc, outc),
        )
        self.pres = pres
        self.vres = vres

    def forward(self, z):
        """forward"""
        # x = initial_voxelize(z, self.pres, self.vres)
        x = z
        x = self.net(x)
        # out = voxel_to_point(x, z, nearest=False)
        # out.F = out.F + self.point_transforms(z.F)
        out = x
        return out


class ConvGRU(nn.Module):
    """ConvGRU"""
    def __init__(self, hidden_dim=128, input_dim=192 + 128, pres=1, vres=1):
        """init"""
        super(ConvGRU, self).__init__()
        self.convz = DConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        self.convr = DConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        self.convq = DConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)

    def forward(self, h, x):
        '''
        :param h: DenseTensor
        :param x: DenseTensor
        :return: h: Tensor (N, C, H, W, D)
        '''
        # hx = PointTensor(torch.cat([h.F, x.F], dim=1), h.C)
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        x = torch.cat([r * h, x], dim=1)
        q = torch.tanh(self.convq(x))

        h = (1 - z) * h + z * q
        return h
