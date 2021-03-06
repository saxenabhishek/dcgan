r"""
binnary cross entropy loss functions
"""

import torch
from torch import nn


class BCE:
    criterion = nn.BCELoss()
    truth_label = 1
    fake_label = 0
    loss_points = {"d1": [], "d2": [], "g": [], "FakeOpinion": [], "RealOpinion": []}

    def disc(self, realD, fakeD) -> torch.Tensor:
        self.loss_points["FakeOpinion"].append(fakeD.mean().item())
        self.loss_points["RealOpinion"].append(realD.mean().item())
        reallab = torch.ones_like(realD) * self.truth_label

        L_real = self.criterion(realD, reallab)
        self.loss_points["d1"].append(L_real.mean().item())
        L_real.backward()

        fakelab = reallab.fill_(self.fake_label)
        L_fake = self.criterion(fakeD, fakelab)
        self.loss_points["d2"].append(L_fake.mean().item())
        L_fake.backward()

        L = (L_real + L_fake) / 2
        # self.loss_points["d"].append(L.mean().item())
        return L

    def gen(self, GenfakeD):
        reallab = torch.ones_like(GenfakeD) * self.truth_label
        L_Fake = self.criterion(GenfakeD, reallab)
        L_Fake.backward()
        self.loss_points["g"].append(L_Fake.mean().item())
        return L_Fake

