r"""
discriminator
"""

from torch import nn


class Discriminator(nn.Module):
    def __init__(
        self, c, filter,
    ):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(self.block(c, filter * 2, first=True), self.block(filter * 2, filter * 4),)
        self.last = nn.Sequential(
            nn.Conv2d(filter * 4, filter * 8, 4, 2, 1, bias=False),
            # nn.Flatten(),
            # nn.Linear(filter * 8 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def block(self, i, o, first=False):
        l = []
        l.append(nn.Conv2d(i, o, 4, 2 if not first else 1, 1 if not first else 2, bias=False))
        if not first:
            l.append(nn.BatchNorm2d(o))
        l.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*l)

    def forward(self, input):
        x = self.main(input)
        x = self.last(x)
        return x
