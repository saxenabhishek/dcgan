r"""
trainer
"""

import code.models.disc as D
import code.models.gen as G

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


class trainer:
    def __init__(self, lr, dataloader) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vector = 512
        self.gen = G.Generator(self.vector, 4, 1).to(self.device)
        self.disc = D.Discriminator(1, 4).to(self.device)

        self.gen.apply(self.weights_init)
        self.disc.apply(self.weights_init)

        beta1 = 0.5
        self.optimD = optim.Adam(self.gen.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimG = optim.Adam(self.gen.parameters(), lr=lr, betas=(beta1, 0.999))

        self.dataloader = dataloader

    def train(self, e):
        for _ in range(e):
            for i, data in enumerate(tqdm(self.dataloader), 0):
                bz = data[0][0]
                sample = data[0].to(self.device)

                realD = self.disc(sample)

                noise = noise = torch.randn(bz, self.vector, device=self.device)

                self.disc.zero_grad()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
