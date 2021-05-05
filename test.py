import dcgan.models.disc as D
import dcgan.models.gen as G
from dcgan.trainer import trainer
import dcgan.utils as utl

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms

if __name__ == "__main__":
    # gen = G.Generator(512, 4, 1)
    # print(gen)
    # o = gen(torch.rand(1, 512))
    # print(o.shape)

    # disc = D.Discriminator(1, 4)
    # print(disc)
    # o = disc(torch.rand(1, 1, 28, 28))
    # print(o.mean())

    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

    data = torchvision.datasets.MNIST("Data", train=True, download=True, transform=t)
    dl = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)

    dcgan = trainer(0.0002, dl)

    dcgan.train(5)
    # print(len(dl))
    # r = random.randint(0, len(dl))
    # sample = next(iter(dl))
    # print(sample[0].shape)
    # utl.show_tensor_images(sample[0])

