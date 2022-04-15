import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_images(images, targets, outputs):

    f, ax = plt.subplots(1, 3)
    ax[0].imshow(np.uint8(torch.permute(images[0], (1,2,0)).squeeze().cpu().detach().numpy()))
    ax[0].set_title('Input', size=10)
    ax[1].imshow(targets[0].squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
    ax[1].set_title('Physician Segmentation', size=10)
    ax[2].imshow(outputs[0].squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
    #ax[2].imshow(np.uint8(torch.permute(images[0], (1,2,0)).squeeze().cpu().detach().numpy()), cmap=plt.cm.bone, alpha=.5)
    ax[2].set_title('Segmentation', size=10)


    plt.show()
