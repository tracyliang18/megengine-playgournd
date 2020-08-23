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
        self.layers = layers
        inp = 1
        self.convs = []
        self.deconvs = []

        self.gap = M.pooling.AvgPool2d(3)

        for layer in layers:
            self.convs.append(conv_bn_relu_pool(inp, layer, 3, padding=1))
            inp = layer

        inp += 10
        for layer in layers[::-1]:# + [1,]:
            self.deconvs.append(transpose_conv_bn_relu(inp, layer, 5, padding=1, stride=2, relu=layer!=1))
            inp = layer

        self.predict_layer = M.conv_bn.ConvBn2d(inp, 1, 1)

        self.fc_mean = M.Linear(layers[-1], layers[-1])
        self.fc_var = M.Linear(layers[-1], layers[-1])

    def encoding(self, x):
        for conv in self.convs:
            x = conv(x)
            print('encoding', x.shape)

        print('fmap', x.shape)
        #code = self.gap(x)
        #code = code.reshape(-1, code.shape[1])
        code = x.dimshuffle(0, 2, 3, 1)

        print(code.shape)
        mean = self.fc_mean(code).dimshuffle(0, 3, 1, 2)
        var = 1e-5 + F.exp(self.fc_var(code)).dimshuffle(0, 3, 1, 2)

        return mean, var

    def get_sample_code(self, gaussian, mean, var, onehot):

        #z = mge.random.gaussian(mean.shape, mean=0, std=1)
        #mean = mean.reshape(*mean.shape, 1, 1)
        #mean = F.add_axis(F.add_axis(mean, 2), 3)
        #var = F.add_axis(F.add_axis(var, 2), 3)

        z = gaussian
        z = z * F.sqrt(var) + mean

        print('gaussian, mean, var, z', gaussian.shape, mean.shape, var.shape, z.shape)
        z = F.concat([z,onehot],axis=1)

        return z

    def decoding(self, x):
        for deconv in self.deconvs:
            x = deconv(x)
            print('decoding', x.shape)


        #x = F.sigmoid(x[:, :, 1:29, 1:29])
        x = F.sigmoid(self.predict_layer(x))
        x = x[:, :, 1:29, 1:29]

        return x

    def forward(self, x, gaussian, onehot):

        mean, var = self.encoding(x)
        code = self.get_sample_code(gaussian, mean, var, onehot)
        decode = self.decoding(code)

        return mean, var, decode

def get_net():
    c = 8
    net = AE([8*c, 16*c, 24*c])
    return net

if __name__ == '__main__':
    net = get_net()
    net.eval()
    #x = mge.tensor(np.random.randn(2, 1, 28, 28).astype(np.float32))
    x = np.random.randn(2, 1, 28, 28).astype(np.float32)
    gaussian = np.random.normal(size=(2,net.layers[-1],3,3))
    target = np.array([1,3])
    onehot = np.broadcast_to(np.eye(10)[target][:,:,np.newaxis,np.newaxis], (2,10,3,3))
    print(onehot, onehot.shape)
    t_x = mge.tensor()
    t_x.set_value(x)
    t_gaussian = mge.tensor()
    t_gaussian.set_value(gaussian)
    t_onehot = mge.tensor()
    t_onehot .set_value(onehot)
    _, _, out = net(t_x, t_gaussian, t_onehot)
    print(out.shape)
    print(out)

    #net = ResNet(Bottleneck, [3,3,3,3]
