r"""
generator
"""
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z, filter, c):
        super(Generator, self).__init__()
        self.first = nn.Sequential(nn.Linear(z, 7 * 7 * z), nn.ReLU(True))
        self.main = nn.Sequential(self.block(z, filter * 16), self.block(filter * 16, c, last=True))

    def block(self, i, o, last=False):
        l = []
        l.append(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=True))
        if not last:
            l.append(nn.BatchNorm2d(o))
        l.append(nn.ReLU(True) if not last else nn.Tanh())
        return nn.Sequential(*l)

    def forward(self, input):
        bs = input.shape[0]
        x = self.first(input)
        x = x.reshape(bs, -1, 7, 7)
        return self.main(x)
