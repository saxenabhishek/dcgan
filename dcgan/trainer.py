r"""
trainer
"""

import dcgan.models.disc as D
import dcgan.models.gen as G
from dcgan.loss.bce import BCE
import dcgan.utils as utl

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt


class trainer:
    def __init__(self, lr, dataloader) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vector = 512
        self.gen = G.Generator(self.vector, 4, 1).to(self.device)
        self.disc = D.Discriminator(1, 4).to(self.device)
        self.testnoise = torch.randn((128, self.vector), device=self.device)

        self.BCE = BCE()

        self.gen.apply(self.weights_init)
        self.disc.apply(self.weights_init)

        beta1 = 0.5
        self.optimD = optim.Adam(self.gen.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimG = optim.Adam(self.gen.parameters(), lr=lr, betas=(beta1, 0.999))

        self.dataloader = dataloader

    def train(self, ep) -> None:
        with torch.autograd.set_detect_anomaly(True):
            for e in range(ep):
                for i, data in enumerate(tqdm(self.dataloader), 0):
                    bz = data[0].size(0)
                    sample = data[0].to(self.device)

                    self.disc.zero_grad()
                    realD = self.disc(sample)

                    noise = torch.randn((bz, self.vector), device=self.device)
                    fake_sample = self.gen(noise).detach()
                    fakeD = self.disc(fake_sample)

                    lossd = self.BCE.disc(realD, fakeD)

                    self.optimD.step()

                    self.disc.zero_grad()
                    noise = torch.randn((bz, self.vector), device=self.device)
                    fake_sample = self.gen(noise)
                    genfakeD = self.disc(fake_sample)

                    lossg = self.BCE.gen(genfakeD)

                    self.optimG.step()

                    if i % 400 == 0:
                        print(f"  {e}   {lossd.mean().item()}\t{lossg.mean().item()}")
                        with torch.no_grad():
                            fake = self.gen(self.testnoise)
                            print(fake[0])
                            utl.show_tensor_images(torch.cat([fake[:8], sample[:8]]))
                            self.show_plots()

    def show_plots(self):
        L = self.BCE.loss_points
        for i in L:
            plt.plot(L[i], label=i)
        plt.legend()
        plt.show()

    def weights_init(self, m) -> None:
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
