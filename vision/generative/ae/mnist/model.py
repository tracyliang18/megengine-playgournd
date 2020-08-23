import math

import megengine as mge
import megengine.functional as F
import megengine.hub as hub
import megengine.module as M
import numpy as np

class conv_bn_relu_pool(M.Module):
    def __init__(self, *args, relu=True, **kwargs):
        self.conv_bn = M.conv_bn.ConvBn2d(*args, **kwargs)
        self.pooling = M.pooling.AvgPool2d(2)
        self.relu = relu

    def forward(self, x):
        x = self.conv_bn(x)
        if self.relu:
            x = F.relu(x)
        x = self.pooling(x)
        return x

class transpose_conv_bn_relu(M.Module):
    def __init__(self, *args, relu=True, **kwargs):
        self.conv_bn = M.conv.ConvTranspose2d(*args, **kwargs)
        self.relu = relu

    def forward(self, x):
        x = self.conv_bn(x)
        if self.relu:
            x = F.relu(x)
        return x

class AE(M.Module):
    def __init__(self, layers):
        inp = 1
        self.convs = []
        self.deconvs = []

        self.gap = M.pooling.AvgPool2d(3)

        for layer in layers:
            self.convs.append(conv_bn_relu_pool(inp, layer, 3, padding=1))
            inp = layer

        for layer in layers[::-1]:# + [1,]:
            self.deconvs.append(transpose_conv_bn_relu(inp, layer, 5, padding=1, stride=2, relu=layer!=1))
            inp = layer

        self.predict_layer = M.conv_bn.ConvBn2d(inp, 1, 1)

    def encoding(self, x):
        for conv in self.convs:
            x = conv(x)
            print('encoding', x.shape)

        #code = self.gap(x)

        code = x
        return code

    def decoding(self, x):
        for deconv in self.deconvs:
            x = deconv(x)
            print('decoding', x.shape)


        #x = F.sigmoid(x[:, :, 1:29, 1:29])
        x = F.sigmoid(self.predict_layer(x))
        x = x[:, :, 1:29, 1:29]

        return x

    def forward(self, x):

        code = self.encoding(x)
        print('code size', code.shape)
        decode = self.decoding(code)

        return decode

def get_net():
    net = AE([8, 16, 24])
    return net

if __name__ == '__main__':
    x = mge.tensor(np.random.randn(1, 1, 28, 28).astype(np.float32))
    net = AE([8, 16, 24])
    out = net(x)
    print(out.shape)
    print(out)

    #net = ResNet(Bottleneck, [3,3,3,3]
