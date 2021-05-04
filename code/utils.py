from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import math

def show_tensor_images(image_tensor):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    plt.figure(figsize=(5, 5))
    numImgs = image_tensor.shape[0]
    edgeNum = int(numImgs / int(math.sqrt(numImgs)))
    image_grid = make_grid(image_unflat, nrow=edgeNum)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis(False)
    plt.show()