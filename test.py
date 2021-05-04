import code.models.gen as G
import torch

if __name__ == "__main__":
    gen = G.Generator(512, 4)
    print(gen)
    o = gen(torch.rand(1, 512))
    print(o.shape)
