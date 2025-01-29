import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock

# from models.resnet import ResNetBase


class Encoder3D(nn.Module):
    def __init__(self, in_channels, out_channels, D=3, planes= (32, 64, 96)):
        nn.Module.__init__(self)
        self.BLOCK = BasicBlock 
        self.D = D
        self.PLANES = planes 
        self.LAYERS = (1, 1, 1)
        self.inplanes = 32

        self.conv1 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=3, dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes) 
        self.relu = ME.MinkowskiReLU(inplace=True)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0], stride = 1)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1], stride = [1,2,2])
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2], stride = [1,2,2])

        self.conv2 = ME.MinkowskiConvolution(self.PLANES[2], out_channels, kernel_size=1, dimension=D)

        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D
                )
            )

        return nn.Sequential(*layers)
       
    def forward(self, x):
        # feat = []
        out = self.conv1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)
        
        out_p2 = self.block1(out_p1)
        out_p3 = self.block2(out_p2)
        out_p4 = self.block3(out_p3)

        fin = self.conv2(out_p4)

        return  fin